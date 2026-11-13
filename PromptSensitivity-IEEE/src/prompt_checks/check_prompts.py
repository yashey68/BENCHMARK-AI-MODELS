import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import textstat
from rapidfuzz.distance import Levenshtein


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/dev_prompts.jsonl"
OUTPUT_FILE = ROOT / "results/prompt_checks/prompt_metrics.csv"
SUMMARY_FILE = ROOT / "results/prompt_checks/prompt_summary.csv"


def words(text):
    return re.findall(r"\b[\w'-]+\b", text.lower())


def sentences(text):
    parts = re.split(r"[.!?]+", text)
    return [p.strip() for p in parts if p.strip()]


def lexical_diversity(text):
    tokens = words(text)

    if not tokens:
        return 0.0

    return len(set(tokens)) / len(tokens)


def jaccard_similarity(text_a, text_b):
    a = set(words(text_a))
    b = set(words(text_b))

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return len(a & b) / len(a | b)


def normalized_edit_distance(text_a, text_b):
    maximum = max(len(text_a), len(text_b))

    if maximum == 0:
        return 0.0

    return Levenshtein.distance(text_a, text_b) / maximum


def average_word_length(text):
    tokens = words(text)

    if not tokens:
        return 0.0

    return sum(len(token) for token in tokens) / len(tokens)


def average_sentence_length(text):
    token_list = words(text)
    sentence_list = sentences(text)

    if not sentence_list:
        return float(len(token_list))

    return len(token_list) / len(sentence_list)


def instruction_only(prompt, question):
    """
    Remove the benchmark question before calculating prompt complexity.

    This is important because we want to measure the complexity of the
    instruction itself rather than the difficulty/length of the question.
    """
    return prompt.replace(question, "[QUESTION]")


def load_records():
    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    return records


def main():
    records = load_records()

    # Store the P0 instruction for each question.
    baselines = {}

    for record in records:
        if record["prompt_id"] == "P0":
            baselines[record["question_id"]] = instruction_only(
                record["prompt"],
                record["question"]
            )

    output = []

    for record in records:
        prompt = record["prompt"]
        question = record["question"]

        instruction = instruction_only(prompt, question)
        baseline = baselines[record["question_id"]]

        token_list = words(instruction)
        sentence_list = sentences(instruction)

        metrics = {
            "question_id": record["question_id"],
            "task": record["task"],
            "prompt_id": record["prompt_id"],
            "prompt_name": record["prompt_name"],
            "prompt_group": record["prompt_group"],
            "complexity_level": record["complexity_level"],
            "manipulation": record["manipulation"],

            # Surface complexity
            "word_count": len(token_list),
            "character_count": len(instruction),
            "sentence_count": len(sentence_list),
            "avg_word_length": round(
                average_word_length(instruction), 4
            ),
            "avg_sentence_length": round(
                average_sentence_length(instruction), 4
            ),
            "lexical_diversity": round(
                lexical_diversity(instruction), 4
            ),

            # Readability
            "flesch_reading_ease": round(
                textstat.flesch_reading_ease(instruction), 4
            ),
            "flesch_kincaid_grade": round(
                textstat.flesch_kincaid_grade(instruction), 4
            ),

            # Difference from P0
            "lexical_jaccard_vs_p0": round(
                jaccard_similarity(baseline, instruction), 4
            ),
            "normalized_edit_distance_vs_p0": round(
                normalized_edit_distance(baseline, instruction), 4
            ),
        }

        output.append(metrics)

    df = pd.DataFrame(output)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    numeric_columns = [
        "word_count",
        "character_count",
        "sentence_count",
        "avg_word_length",
        "avg_sentence_length",
        "lexical_diversity",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "lexical_jaccard_vs_p0",
        "normalized_edit_distance_vs_p0",
    ]

    summary = (
        df.groupby(
            [
                "prompt_id",
                "prompt_name",
                "complexity_level",
                "prompt_group"
            ],
            as_index=False
        )[numeric_columns]
        .mean()
        .sort_values("complexity_level")
    )

    summary.to_csv(SUMMARY_FILE, index=False)

    print("\n" + "=" * 90)
    print("PROMPT VALIDATION ENGINE — PHASE 1")
    print("=" * 90)

    display_columns = [
        "prompt_id",
        "complexity_level",
        "word_count",
        "sentence_count",
        "avg_word_length",
        "lexical_diversity",
        "flesch_reading_ease",
        "flesch_kincaid_grade",
        "lexical_jaccard_vs_p0",
        "normalized_edit_distance_vs_p0",
    ]

    print(
        summary[display_columns].to_string(
            index=False
        )
    )

    print("\n" + "-" * 90)
    print("Saved detailed metrics:")
    print(OUTPUT_FILE)

    print("\nSaved prompt-level summary:")
    print(SUMMARY_FILE)

    print("=" * 90)


if __name__ == "__main__":
    main()
