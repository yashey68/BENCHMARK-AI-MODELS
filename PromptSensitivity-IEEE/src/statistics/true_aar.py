from pathlib import Path
from itertools import combinations
import re
import string

import pandas as pd

INPUT = Path("results/processed/cross_model/all_models_evaluated.csv")
OUTPUT = Path("results/processed/cross_model")
OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

def normalize_answer(x):
    if pd.isna(x):
        return "<EMPTY>"

    x = str(x).lower().strip()

    x = x.translate(
        str.maketrans("", "", string.punctuation)
    )

    x = re.sub(r"\b(a|an|the)\b", " ", x)
    x = re.sub(r"\s+", " ", x).strip()

    return x if x else "<EMPTY>"


# Find the evaluator's prediction/output column.
candidates = [
    "normalized_prediction",
    "prediction",
    "extracted_answer",
    "response",
]

answer_col = None

for col in candidates:
    if col in df.columns:
        answer_col = col
        break

if answer_col is None:
    raise ValueError(
        "Could not locate prediction column.\n"
        f"Available columns:\n{list(df.columns)}"
    )

print("=" * 76)
print("TRUE PAIRWISE ANSWER AGREEMENT RATE")
print("=" * 76)
print("Answer column:", answer_col)

df["_aar_answer"] = df[answer_col].apply(normalize_answer)

question_rows = []

for (model, qid), group in df.groupby(
    ["model", "question_id"]
):
    group = group.sort_values("prompt_id")

    if len(group) != 10:
        raise ValueError(
            f"{model} {qid}: expected 10 prompts, got {len(group)}"
        )

    answers = group["_aar_answer"].tolist()

    pairs = list(combinations(range(10), 2))

    agreements = sum(
        answers[i] == answers[j]
        for i, j in pairs
    )

    aar = agreements / len(pairs)

    question_rows.append({
        "model": model,
        "question_id": qid,
        "task": group["task"].iloc[0],
        "agreeing_pairs": agreements,
        "total_pairs": len(pairs),
        "pairwise_aar": aar,
        "unique_answers": len(set(answers)),
    })

qdf = pd.DataFrame(question_rows)

qdf.to_csv(
    OUTPUT / "question_pairwise_aar.csv",
    index=False
)

summary = (
    qdf
    .groupby("model")
    .agg(
        questions=("question_id", "count"),
        mean_pairwise_aar=("pairwise_aar", "mean"),
        median_pairwise_aar=("pairwise_aar", "median"),
        mean_unique_answers=("unique_answers", "mean"),
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT / "pairwise_aar_summary.csv",
    index=False
)

task_summary = (
    qdf
    .groupby(["model", "task"])
    .agg(
        mean_pairwise_aar=("pairwise_aar", "mean"),
        mean_unique_answers=("unique_answers", "mean"),
    )
    .reset_index()
)

task_summary.to_csv(
    OUTPUT / "pairwise_aar_by_task.csv",
    index=False
)

print("\nMODEL SUMMARY")
print(summary.to_string(index=False))

print("\nMODEL × TASK SUMMARY")
print(task_summary.to_string(index=False))

print("\nSaved:")
print(OUTPUT / "question_pairwise_aar.csv")
print(OUTPUT / "pairwise_aar_summary.csv")
print(OUTPUT / "pairwise_aar_by_task.csv")

print("=" * 76)
