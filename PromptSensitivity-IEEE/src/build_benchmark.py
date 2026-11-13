import argparse
import json
import random
import re
from pathlib import Path

from datasets import load_dataset


ROOT = Path(__file__).resolve().parents[1]

OUTPUT_FILE = (
    ROOT /
    "data" /
    "benchmark" /
    "questions.jsonl"
)

SEED = 42


# ============================================================
# MATH — GSM8K
# ============================================================

def extract_gsm8k_answer(answer_text):
    match = re.search(
        r"####\s*(.+)",
        answer_text
    )

    if not match:
        raise ValueError(
            "Could not extract GSM8K answer"
        )

    return (
        match
        .group(1)
        .strip()
        .replace(",", "")
    )


def build_math(n=40):

    print("\nLoading GSM8K...")

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="test"
    )

    print(
        f"GSM8K test examples available: {len(dataset)}"
    )

    indices = list(
        range(len(dataset))
    )

    rng = random.Random(SEED)
    rng.shuffle(indices)

    selected = indices[:n]

    records = []

    for number, index in enumerate(
        selected,
        start=1
    ):

        row = dataset[index]

        records.append({
            "id": f"MATH_{number:03d}",
            "task": "math",
            "source": "gsm8k",
            "source_config": "main",
            "source_split": "test",
            "source_index": index,
            "question": row["question"].strip(),
            "context": None,
            "choices": None,
            "answer": extract_gsm8k_answer(
                row["answer"]
            ),
        })

    return records


# ============================================================
# LOGIC — BIG-BENCH LOGICAL DEDUCTION
# ============================================================

def build_logic(n=40):

    print("\nLoading BIG-bench Logical Deduction...")

    dataset = load_dataset(
        "tasksource/bigbench",
        "logical_deduction",
        split="validation"
    )

    print(
        f"Logical deduction validation examples available: "
        f"{len(dataset)}"
    )

    indices = list(
        range(len(dataset))
    )

    # Separate deterministic seed so categories are independently
    # reproducible.
    rng = random.Random(SEED + 1)
    rng.shuffle(indices)

    selected = indices[:n]

    records = []

    labels = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
    ]

    for number, index in enumerate(
        selected,
        start=1
    ):

        row = dataset[index]

        targets = row[
            "multiple_choice_targets"
        ]

        scores = row[
            "multiple_choice_scores"
        ]

        if len(targets) != len(scores):
            raise ValueError(
                f"Target/score mismatch at logic index {index}"
            )

        correct_positions = [
            i
            for i, score in enumerate(scores)
            if score == 1
        ]

        if len(correct_positions) != 1:
            raise ValueError(
                f"Expected exactly one correct answer "
                f"at logic index {index}; "
                f"found {len(correct_positions)}"
            )

        correct_position = correct_positions[0]

        if len(targets) > len(labels):
            raise ValueError(
                f"Too many choices at logic index {index}"
            )

        choices = {
            labels[i]: target.strip()
            for i, target in enumerate(targets)
        }

        answer_label = labels[
            correct_position
        ]

        records.append({
            "id": f"LOGIC_{number:03d}",
            "task": "logic",
            "source": "bigbench_logical_deduction",
            "source_config": "logical_deduction",
            "source_split": "validation",
            "source_index": index,
            "question": row["inputs"].strip(),
            "context": None,
            "choices": choices,
            "answer": answer_label,
            "answer_text":
                targets[correct_position].strip(),
        })

    return records


# ============================================================
# COMMONSENSE — COMMONSENSEQA
# ============================================================

def build_commonsense(n=40):

    print("\nLoading CommonsenseQA...")

    dataset = load_dataset(
        "tau/commonsense_qa",
        split="validation"
    )

    print(
        f"CommonsenseQA validation examples available: "
        f"{len(dataset)}"
    )

    indices = list(
        range(len(dataset))
    )

    # Independent deterministic seed
    rng = random.Random(SEED + 2)
    rng.shuffle(indices)

    selected = indices[:n]

    records = []

    for number, index in enumerate(
        selected,
        start=1
    ):

        row = dataset[index]

        labels = row["choices"]["label"]
        texts = row["choices"]["text"]

        if len(labels) != len(texts):
            raise ValueError(
                f"Choice label/text mismatch at "
                f"CommonsenseQA index {index}"
            )

        choices = {
            label: text.strip()
            for label, text
            in zip(labels, texts)
        }

        answer = row["answerKey"].strip()

        if not answer:
            raise ValueError(
                f"Missing answer at "
                f"CommonsenseQA index {index}"
            )

        if answer not in choices:
            raise ValueError(
                f"Answer {answer} not present in choices "
                f"at CommonsenseQA index {index}"
            )

        records.append({
            "id": f"COMMON_{number:03d}",
            "task": "commonsense",
            "source": "commonsense_qa",
            "source_config": None,
            "source_split": "validation",
            "source_index": index,
            "source_id": row["id"],
            "question": row["question"].strip(),
            "context": None,
            "choices": choices,
            "answer": answer,
            "answer_text": choices[answer],
            "question_concept":
                row["question_concept"],
        })

    return records



# ============================================================
# KNOWLEDGE — MMLU
# ============================================================

MMLU_DOMAINS = {

    "stem": [
        "abstract_algebra",
        "college_mathematics",
        "conceptual_physics",
        "college_physics",
        "college_chemistry",
        "college_computer_science",
        "machine_learning",
    ],

    "humanities": [
        "world_religions",
        "philosophy",
        "moral_disputes",
        "moral_scenarios",
        "prehistory",
    ],

    "social_sciences": [
        "high_school_psychology",
        "professional_psychology",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_government_and_politics",
        "sociology",
    ],

    "professional_other": [
        "professional_law",
        "professional_accounting",
        "professional_medicine",
        "clinical_knowledge",
        "business_ethics",
        "management",
        "marketing",
    ],
}


def build_knowledge(n=40):

    print("\nLoading MMLU...")

    dataset = load_dataset(
        "cais/mmlu",
        "all",
        split="test"
    )

    print(
        f"MMLU test examples available: {len(dataset)}"
    )

    if n % len(MMLU_DOMAINS) != 0:
        raise ValueError(
            "Knowledge sample count must be divisible "
            "by the number of MMLU domains."
        )

    per_domain = n // len(MMLU_DOMAINS)

    labels = ["A", "B", "C", "D"]

    records = []

    number = 1

    for domain_number, (domain, subjects) in enumerate(
        MMLU_DOMAINS.items()
    ):

        candidates = []

        for index, row in enumerate(dataset):

            if row["subject"] in subjects:
                candidates.append(
                    (index, row)
                )

        print(
            f"{domain}: "
            f"{len(candidates)} candidates"
        )

        if len(candidates) < per_domain:
            raise ValueError(
                f"Not enough MMLU examples "
                f"for domain {domain}"
            )

        rng = random.Random(
            SEED + 10 + domain_number
        )

        rng.shuffle(candidates)

        selected = candidates[:per_domain]

        for index, row in selected:

            raw_choices = row["choices"]

            if len(raw_choices) != 4:
                raise ValueError(
                    f"Expected 4 choices at "
                    f"MMLU index {index}"
                )

            choices = {
                labels[i]: str(choice).strip()
                for i, choice
                in enumerate(raw_choices)
            }

            answer_index = int(
                row["answer"]
            )

            if answer_index < 0 or answer_index >= 4:
                raise ValueError(
                    f"Invalid MMLU answer index "
                    f"at {index}: {answer_index}"
                )

            answer_label = labels[
                answer_index
            ]

            records.append({
                "id": f"KNOW_{number:03d}",
                "task": "knowledge",
                "source": "mmlu",
                "source_config": "all",
                "source_split": "test",
                "source_index": index,
                "subject": row["subject"],
                "knowledge_domain": domain,
                "question":
                    row["question"].strip(),
                "context": None,
                "choices": choices,
                "answer": answer_label,
                "answer_text":
                    choices[answer_label],
            })

            number += 1

    return records



# ============================================================
# READING — SQUAD
# ============================================================

def build_reading(n=40):

    print("\nLoading SQuAD...")

    dataset = load_dataset(
        "rajpurkar/squad",
        split="validation"
    )

    print(
        f"SQuAD validation examples available: "
        f"{len(dataset)}"
    )

    indices = list(
        range(len(dataset))
    )

    # Independent deterministic seed
    rng = random.Random(SEED + 30)
    rng.shuffle(indices)

    records = []

    for index in indices:

        if len(records) >= n:
            break

        row = dataset[index]

        question = row["question"].strip()
        context = row["context"].strip()

        raw_answers = row["answers"]["text"]

        # Remove duplicate answer annotations while
        # preserving their original order.
        gold_answers = []

        for answer in raw_answers:

            answer = answer.strip()

            if (
                answer
                and answer not in gold_answers
            ):
                gold_answers.append(answer)

        if not question:
            continue

        if not context:
            continue

        if not gold_answers:
            continue

        records.append({
            "id":
                f"READ_{len(records) + 1:03d}",
            "task": "reading",
            "source": "squad",
            "source_config": "plain_text",
            "source_split": "validation",
            "source_index": index,
            "source_id": row["id"],
            "title": row["title"],
            "question": question,
            "context": context,
            "choices": None,

            # Primary answer retained for compatibility
            # with the common benchmark schema.
            "answer": gold_answers[0],

            # All valid SQuAD annotations retained
            # for proper EM/F1 evaluation.
            "answers": gold_answers,
        })

    if len(records) != n:
        raise ValueError(
            f"Requested {n} SQuAD questions "
            f"but created {len(records)}."
        )

    return records



# ============================================================
# VALIDATION
# ============================================================

def validate(records):

    print("\nValidating records...")

    ids = [
        row["id"]
        for row in records
    ]

    if len(ids) != len(set(ids)):
        raise ValueError(
            "Duplicate benchmark IDs detected."
        )

    for row in records:

        if not row["question"]:
            raise ValueError(
                f"Empty question: {row['id']}"
            )

        if not str(row["answer"]).strip():
            raise ValueError(
                f"Empty answer: {row['id']}"
            )

        if row["choices"] is not None:

            if row["answer"] not in row["choices"]:
                raise ValueError(
                    f"Answer label {row['answer']} "
                    f"missing from choices for {row['id']}"
                )

    print(
        f"Validated {len(records)} records."
    )


# ============================================================
# SAVE
# ============================================================

def save(records):

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        for row in records:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                )
                + "\n"
            )

    print(
        f"\nSaved benchmark to:\n{OUTPUT_FILE}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--math",
        type=int,
        default=40
    )

    parser.add_argument(
        "--logic",
        type=int,
        default=40
    )

    parser.add_argument(
        "--commonsense",
        type=int,
        default=40
    )

    parser.add_argument(
        "--knowledge",
        type=int,
        default=40
    )

    parser.add_argument(
        "--reading",
        type=int,
        default=40
    )

    args = parser.parse_args()

    print("=" * 70)
    print("PROMPT SENSITIVITY — BENCHMARK BUILDER")
    print("=" * 70)

    print(
        f"Base random seed: {SEED}"
    )

    math_records = build_math(
        args.math
    )

    logic_records = build_logic(
        args.logic
    )

    commonsense_records = build_commonsense(
        args.commonsense
    )

    knowledge_records = build_knowledge(
        args.knowledge
    )

    reading_records = build_reading(
        args.reading
    )

    records = (
        math_records
        + logic_records
        + commonsense_records
        + knowledge_records
        + reading_records
    )

    validate(records)

    save(records)

    print("\n" + "=" * 70)

    print(
        f"Math questions  : {len(math_records)}"
    )

    print(
        f"Logic questions : {len(logic_records)}"
    )

    print(
        f"Commonsense     : {len(commonsense_records)}"
    )

    print(
        f"Knowledge       : {len(knowledge_records)}"
    )

    print(
        f"Reading         : {len(reading_records)}"
    )

    print(
        f"Total questions : {len(records)}"
    )

    print("=" * 70)


if __name__ == "__main__":
    main()
