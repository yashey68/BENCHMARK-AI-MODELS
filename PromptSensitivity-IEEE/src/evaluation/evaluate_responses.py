import argparse
import json
import re
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "results/raw"
OUTPUT_DIR = ROOT / "results/processed"


def normalize(text):
    text = str(text).strip().lower()

    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("$", "")
    text = text.replace(",", "")

    text = re.sub(r"\s+", " ", text)

    return text.strip(" .!?:;")


def normalize_numeric(text):
    text = normalize(text)

    # Keep numbers, decimal points, minus signs and percentages.
    matches = re.findall(
        r"-?\d+(?:\.\d+)?%?",
        text
    )

    return matches


def extract_yes_no(text):
    text = normalize(text)

    # Prefer explicit final-answer patterns.
    patterns = [
        r"(?:answer|final answer)\s*(?:is|:)?\s*(yes|no)\b",
        r"^\s*(yes|no)\b",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return match.group(1)

    tokens = re.findall(r"\b(?:yes|no)\b", text)

    if tokens:
        return tokens[-1]

    return None


def extract_answer(response, ground_truth, task):
    response_norm = normalize(response)
    truth_norm = normalize(ground_truth)

    # YES / NO questions
    if truth_norm in {"yes", "no"}:
        extracted = extract_yes_no(response)

        if extracted is not None:
            return extracted

    # Numeric / percentage answers
    truth_numbers = normalize_numeric(ground_truth)

    if truth_numbers:
        response_numbers = normalize_numeric(response)

        if response_numbers:
            # Models commonly place the final answer last.
            return response_numbers[-1]

    # Short text answers
    # First look for explicit answer markers.
    patterns = [
        r"(?:final answer|answer)\s*(?:is|:)?\s*(.+)$",
    ]

    for pattern in patterns:
        match = re.search(
            pattern,
            response_norm,
            flags=re.IGNORECASE
        )

        if match:
            candidate = normalize(
                match.group(1)
            )

            # Avoid capturing multiple paragraphs.
            candidate = candidate.split("\n")[0]
            return candidate

    # If the response is already short, treat it as the answer.
    if len(response_norm.split()) <= 8:
        return response_norm

    return response_norm


def score_correctness(extracted, ground_truth):
    extracted = normalize(extracted)
    truth = normalize(ground_truth)

    if extracted == truth:
        return 1

    # Numeric comparison
    extracted_numbers = normalize_numeric(extracted)
    truth_numbers = normalize_numeric(truth)

    if (
        len(extracted_numbers) == 1
        and len(truth_numbers) == 1
        and extracted_numbers[0] == truth_numbers[0]
    ):
        return 1

    return 0


def exact_match(response, ground_truth):
    return int(
        normalize(response)
        == normalize(ground_truth)
    )


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True
    )

    args = parser.parse_args()

    input_file = RAW_DIR / args.input

    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing file: {input_file}"
        )

    rows = load_jsonl(input_file)

    evaluated = []

    for row in rows:
        response = row.get(
            "response",
            ""
        )

        truth = row.get(
            "ground_truth",
            ""
        )

        task = row.get(
            "task",
            ""
        )

        extracted = extract_answer(
            response,
            truth,
            task
        )

        correctness = score_correctness(
            extracted,
            truth
        )

        strict_match = exact_match(
            response,
            truth
        )

        evaluated.append({
            **row,

            "extracted_answer":
                extracted,

            "correct":
                correctness,

            "strict_exact_match":
                strict_match,

            "format_compliant":
                strict_match,
        })

    df = pd.DataFrame(
        evaluated
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    stem = input_file.stem

    csv_file = (
        OUTPUT_DIR /
        f"{stem}_evaluated.csv"
    )

    json_file = (
        OUTPUT_DIR /
        f"{stem}_evaluated.jsonl"
    )

    df.to_csv(
        csv_file,
        index=False
    )

    with json_file.open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in evaluated:
            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    print("\n" + "=" * 100)
    print("PROMPT SENSITIVITY — TASK-AWARE EVALUATION")
    print("=" * 100)

    columns = [
        "question_id",
        "prompt_id",
        "ground_truth",
        "extracted_answer",
        "correct",
        "strict_exact_match",
        "format_compliant",
        "latency_seconds",
    ]

    print(
        df[columns].to_string(
            index=False
        )
    )

    print("\n" + "-" * 100)

    print(
        f"Answer accuracy       : "
        f"{df['correct'].mean():.3f}"
    )

    print(
        f"Strict exact match    : "
        f"{df['strict_exact_match'].mean():.3f}"
    )

    print(
        f"Format compliance     : "
        f"{df['format_compliant'].mean():.3f}"
    )

    print(
        f"Mean latency          : "
        f"{df['latency_seconds'].mean():.3f}s"
    )

    print(
        f"Median latency        : "
        f"{df['latency_seconds'].median():.3f}s"
    )

    print("\nSaved:")
    print(csv_file)
    print(json_file)

    print("=" * 100)


if __name__ == "__main__":
    main()
