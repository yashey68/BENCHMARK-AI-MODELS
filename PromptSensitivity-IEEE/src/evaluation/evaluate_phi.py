import json
import re
import string
import math
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT = ROOT / "results/raw/phi4-mini_latest_responses.jsonl"
OUTPUT_DIR = ROOT / "results/processed/phi4-mini"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_text(text):
    if text is None:
        return ""

    text = str(text).lower()

    # Remove punctuation
    text = "".join(
        ch for ch in text
        if ch not in string.punctuation
    )

    # Remove English articles
    text = re.sub(r"\b(a|an|the)\b", " ", text)

    # Normalize whitespace
    return " ".join(text.split())


def normalize_prediction(text):
    return normalize_text(text)


# ============================================================
# CONFIDENCE INTERVAL
# ============================================================

def wilson_ci(successes, n, z=1.96):
    if n == 0:
        return np.nan, np.nan

    p = successes / n

    denominator = 1 + (z ** 2 / n)

    centre = (
        p + (z ** 2 / (2 * n))
    ) / denominator

    margin = (
        z
        * math.sqrt(
            (p * (1 - p) / n)
            + (z ** 2 / (4 * n ** 2))
        )
        / denominator
    )

    return centre - margin, centre + margin


# ============================================================
# MCQ EXTRACTION
# ============================================================

def extract_mcq_answer(response, choices):
    if not response:
        return None

    text = str(response).strip()

    valid_labels = set()

    if isinstance(choices, dict):
        valid_labels = {
            str(x).upper()
            for x in choices.keys()
        }

    if not valid_labels:
        valid_labels = set("ABCDEFG")

    # --------------------------------------------------------
    # 1. Response is simply "B", "(B)", "B.", etc.
    # --------------------------------------------------------

    m = re.fullmatch(
        r"\s*[\(\[]?([A-G])[\)\].]?\s*",
        text,
        flags=re.I
    )

    if m:
        label = m.group(1).upper()

        if label in valid_labels:
            return label

    # --------------------------------------------------------
    # 2. Explicit final-answer statements
    # --------------------------------------------------------

    patterns = [
        r"(?:final\s+answer|correct\s+answer|answer)\s*(?:is|:|=)\s*[\(\[]?([A-G])\b",
        r"\boption\s+([A-G])\b",
        r"\bchoice\s+([A-G])\b",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.I
        )

        if matches:
            label = matches[-1].upper()

            if label in valid_labels:
                return label

    # --------------------------------------------------------
    # 3. First line begins with a choice label
    # --------------------------------------------------------

    first_line = text.splitlines()[0].strip()

    m = re.match(
        r"^[\(\[]?([A-G])[\)\].:]\s*",
        first_line,
        flags=re.I
    )

    if m:
        label = m.group(1).upper()

        if label in valid_labels:
            return label

    # --------------------------------------------------------
    # 4. Exact / contained choice-text match
    # --------------------------------------------------------

    if isinstance(choices, dict):
        normalized_response = normalize_text(text)

        exact_matches = []

        for label, choice_text in choices.items():
            norm_choice = normalize_text(choice_text)

            if normalized_response == norm_choice:
                exact_matches.append(
                    str(label).upper()
                )

        if len(exact_matches) == 1:
            return exact_matches[0]

        # For short direct answers such as:
        # "B. see beautiful views"
        first_normalized = normalize_text(first_line)

        text_matches = []

        for label, choice_text in choices.items():
            norm_choice = normalize_text(choice_text)

            if norm_choice and norm_choice in first_normalized:
                text_matches.append(
                    str(label).upper()
                )

        if len(text_matches) == 1:
            return text_matches[0]

    # Ambiguous output is intentionally not guessed.
    return None


# ============================================================
# GSM8K NUMERIC EXTRACTION
# ============================================================

def extract_number(text):
    if text is None:
        return None

    text = str(text).replace(",", "")

    # Explicit answer markers first
    patterns = [
        r"####\s*\$?\s*(-?\d+(?:\.\d+)?)",
        r"(?:final\s+answer|answer)\s*(?:is|:|=)\s*\$?\s*(-?\d+(?:\.\d+)?)",
    ]

    for pattern in patterns:
        matches = re.findall(
            pattern,
            text,
            flags=re.I
        )

        if matches:
            return matches[-1]

    # Otherwise final numeric expression
    numbers = re.findall(
        r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?",
        text
    )

    if numbers:
        return numbers[-1]

    return None


def numeric_equal(pred, gold):
    if pred is None or gold is None:
        return False

    try:
        return math.isclose(
            float(pred),
            float(gold),
            rel_tol=1e-9,
            abs_tol=1e-6
        )
    except Exception:
        return False


# ============================================================
# SQUAD METRICS
# ============================================================

def squad_exact_match(prediction, ground_truth):
    return int(
        normalize_text(prediction)
        ==
        normalize_text(ground_truth)
    )


def squad_f1(prediction, ground_truth):
    prediction_tokens = normalize_text(
        prediction
    ).split()

    ground_truth_tokens = normalize_text(
        ground_truth
    ).split()

    if not prediction_tokens or not ground_truth_tokens:
        return float(
            prediction_tokens == ground_truth_tokens
        )

    common = (
        Counter(prediction_tokens)
        &
        Counter(ground_truth_tokens)
    )

    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = (
        num_same / len(prediction_tokens)
    )

    recall = (
        num_same / len(ground_truth_tokens)
    )

    return (
        2 * precision * recall
        / (precision + recall)
    )


def get_accepted_answers(row):
    accepted = row.get("accepted_answers")

    if isinstance(accepted, list) and accepted:
        return [
            str(x)
            for x in accepted
            if str(x).strip()
        ]

    if isinstance(accepted, str) and accepted.strip():
        return [accepted]

    gold = row.get("ground_truth")

    if isinstance(gold, list):
        return [
            str(x)
            for x in gold
        ]

    if gold is not None:
        return [str(gold)]

    return []


# ============================================================
# FORMAT COMPLIANCE
# ============================================================

def mcq_format_compliance(response, choices):
    if not response:
        return 0

    text = response.strip()

    # A direct option label
    if re.fullmatch(
        r"[\(\[]?[A-G][\)\].]?",
        text,
        flags=re.I
    ):
        return 1

    # Label + choice text on a single concise line
    if (
        len(text.splitlines()) == 1
        and len(text.split()) <= 25
        and re.match(
            r"^[\(\[]?[A-G][\)\].:]?\s+",
            text,
            flags=re.I
        )
    ):
        return 1

    return 0


def math_format_compliance(response):
    if not response:
        return 0

    text = response.strip()

    # Direct numeric answer, optionally with currency/unit
    if (
        len(text.splitlines()) == 1
        and len(text.split()) <= 12
        and extract_number(text) is not None
    ):
        return 1

    return 0


def reading_format_compliance(response):
    if not response:
        return 0

    text = response.strip()

    # Prompt requested final answer only.
    # Short answer or concise sentence is acceptable.
    return int(
        len(text.splitlines()) <= 2
        and len(text.split()) <= 40
    )


# ============================================================
# SCORE ONE RECORD
# ============================================================

def score_record(row):
    task = row["task"]
    response = row.get("response", "")
    status = str(
        row.get("status", "")
    ).lower()

    result = {
        "predicted_answer": None,
        "correct": np.nan,
        "exact_match": np.nan,
        "f1": np.nan,
        "format_compliant": 0,
    }

    # Generation failures remain in the dataset but do not
    # masquerade as successful responses.
    if status != "success":
        if task != "reading":
            result["correct"] = 0

        if task == "reading":
            result["exact_match"] = 0
            result["f1"] = 0.0

        return result

    # --------------------------------------------------------
    # MCQ TASKS
    # --------------------------------------------------------

    if task in {
        "commonsense",
        "knowledge",
        "logic"
    }:
        choices = row.get("choices") or {}

        prediction = extract_mcq_answer(
            response,
            choices
        )

        gold = str(
            row.get("ground_truth", "")
        ).strip().upper()

        result["predicted_answer"] = prediction

        result["correct"] = int(
            prediction == gold
        )

        # MCQ EM means extracted label match.
        result["exact_match"] = result["correct"]

        result["format_compliant"] = (
            mcq_format_compliance(
                response,
                choices
            )
        )

        return result

    # --------------------------------------------------------
    # GSM8K
    # --------------------------------------------------------

    if task == "math":
        prediction = extract_number(
            response
        )

        gold = extract_number(
            row.get("ground_truth", "")
        )

        result["predicted_answer"] = prediction

        result["correct"] = int(
            numeric_equal(
                prediction,
                gold
            )
        )

        # Normalized numeric EM
        result["exact_match"] = result["correct"]

        result["format_compliant"] = (
            math_format_compliance(
                response
            )
        )

        return result

    # --------------------------------------------------------
    # SQUAD
    # --------------------------------------------------------

    if task == "reading":
        accepted = get_accepted_answers(
            row
        )

        em_scores = [
            squad_exact_match(
                response,
                gold
            )
            for gold in accepted
        ]

        f1_scores = [
            squad_f1(
                response,
                gold
            )
            for gold in accepted
        ]

        best_em = (
            max(em_scores)
            if em_scores
            else 0
        )

        best_f1 = (
            max(f1_scores)
            if f1_scores
            else 0.0
        )

        result["predicted_answer"] = (
            response.strip()
        )

        result["exact_match"] = best_em
        result["f1"] = best_f1

        # IMPORTANT:
        # Do not invent binary "accuracy" for SQuAD.
        # Reading is evaluated with EM + token F1.
        result["correct"] = np.nan

        result["format_compliant"] = (
            reading_format_compliance(
                response
            )
        )

        return result

    raise ValueError(
        f"Unknown task: {task}"
    )


# ============================================================
# LOAD
# ============================================================

with INPUT.open(
    encoding="utf-8"
) as f:
    rows = [
        json.loads(line)
        for line in f
        if line.strip()
    ]

assert len(rows) == 500, (
    f"Expected 500 records, "
    f"found {len(rows)}"
)


# ============================================================
# RAW STATUS VALIDATION
# ============================================================

status_counts = Counter(
    str(
        row.get("status", "")
    ).lower()
    for row in rows
)

assert (
    status_counts["success"]
    +
    status_counts["truncated"]
    +
    status_counts["empty"]
    +
    status_counts["error"]
) == 500


# ============================================================
# SCORE
# ============================================================

evaluated = []

for row in rows:
    output = dict(row)
    output.update(
        score_record(row)
    )
    evaluated.append(output)

df = pd.DataFrame(evaluated)


# ============================================================
# SAVE EVALUATED RESPONSES
# ============================================================

df.to_csv(
    OUTPUT_DIR / "evaluated_responses.csv",
    index=False
)


# ============================================================
# OBJECTIVE-TASK ACCURACY
#
# Math + logic + commonsense + knowledge.
# SQuAD remains EM/F1 and is not mixed into this accuracy.
# ============================================================

objective_df = df[
    df["task"] != "reading"
].copy()

objective_accuracy = float(
    objective_df["correct"].mean()
)

objective_correct = int(
    objective_df["correct"].sum()
)

objective_n = len(
    objective_df
)

objective_ci_low, objective_ci_high = (
    wilson_ci(
        objective_correct,
        objective_n
    )
)


# ============================================================
# READING METRICS
# ============================================================

reading_df = df[
    df["task"] == "reading"
].copy()

reading_em = float(
    reading_df["exact_match"].mean()
)

reading_f1 = float(
    reading_df["f1"].mean()
)


# ============================================================
# PROMPT METRICS
# ============================================================

prompt_rows = []

for prompt_id, group in df.groupby(
    "prompt_id"
):
    objective_group = group[
        group["task"] != "reading"
    ]

    reading_group = group[
        group["task"] == "reading"
    ]

    n_objective = len(
        objective_group
    )

    correct_objective = int(
        objective_group["correct"].sum()
    )

    ci_low, ci_high = wilson_ci(
        correct_objective,
        n_objective
    )

    prompt_rows.append({
        "prompt_id": prompt_id,
        "complexity_level": int(
            group["complexity_level"].iloc[0]
        ),
        "prompt_group": group[
            "prompt_group"
        ].iloc[0],

        "n_total": len(group),
        "n_objective": n_objective,
        "n_reading": len(
            reading_group
        ),

        "objective_accuracy":
            objective_group[
                "correct"
            ].mean(),

        "objective_accuracy_ci_low":
            ci_low,

        "objective_accuracy_ci_high":
            ci_high,

        "reading_exact_match":
            reading_group[
                "exact_match"
            ].mean(),

        "reading_f1":
            reading_group[
                "f1"
            ].mean(),

        "format_compliance":
            group[
                "format_compliant"
            ].mean(),

        "mean_latency_seconds":
            group[
                "latency_seconds"
            ].mean(),

        "median_latency_seconds":
            group[
                "latency_seconds"
            ].median(),

        "mean_output_tokens":
            group[
                "eval_count"
            ].mean(),

        "median_output_tokens":
            group[
                "eval_count"
            ].median(),

        "truncation_rate":
            (
                group["status"]
                .str.lower()
                ==
                "truncated"
            ).mean()
    })

prompt_df = pd.DataFrame(
    prompt_rows
).sort_values(
    "complexity_level"
)

prompt_df.to_csv(
    OUTPUT_DIR / "prompt_metrics.csv",
    index=False
)


# ============================================================
# TASK METRICS
# ============================================================

task_rows = []

for task, group in df.groupby(
    "task"
):
    row = {
        "task": task,
        "n": len(group),
        "format_compliance":
            group[
                "format_compliant"
            ].mean(),
        "mean_latency_seconds":
            group[
                "latency_seconds"
            ].mean(),
        "median_latency_seconds":
            group[
                "latency_seconds"
            ].median(),
        "mean_output_tokens":
            group[
                "eval_count"
            ].mean(),
        "truncation_rate":
            (
                group["status"]
                .str.lower()
                ==
                "truncated"
            ).mean()
    }

    if task == "reading":
        row.update({
            "accuracy": np.nan,
            "accuracy_ci_low": np.nan,
            "accuracy_ci_high": np.nan,
            "exact_match":
                group[
                    "exact_match"
                ].mean(),
            "f1":
                group[
                    "f1"
                ].mean()
        })

    else:
        n = len(group)
        successes = int(
            group["correct"].sum()
        )

        low, high = wilson_ci(
            successes,
            n
        )

        row.update({
            "accuracy":
                group[
                    "correct"
                ].mean(),
            "accuracy_ci_low": low,
            "accuracy_ci_high": high,
            "exact_match":
                group[
                    "exact_match"
                ].mean(),
            "f1": np.nan
        })

    task_rows.append(row)

task_df = pd.DataFrame(
    task_rows
)

task_df.to_csv(
    OUTPUT_DIR / "task_metrics.csv",
    index=False
)


# ============================================================
# TASK × PROMPT METRICS
# ============================================================

task_prompt_rows = []

for (task, prompt_id), group in df.groupby(
    ["task", "prompt_id"]
):
    row = {
        "task": task,
        "prompt_id": prompt_id,
        "n": len(group),
        "format_compliance":
            group[
                "format_compliant"
            ].mean(),
        "mean_latency_seconds":
            group[
                "latency_seconds"
            ].mean(),
        "mean_output_tokens":
            group[
                "eval_count"
            ].mean(),
    }

    if task == "reading":
        row.update({
            "accuracy": np.nan,
            "exact_match":
                group[
                    "exact_match"
                ].mean(),
            "f1":
                group[
                    "f1"
                ].mean()
        })

    else:
        row.update({
            "accuracy":
                group[
                    "correct"
                ].mean(),
            "exact_match":
                group[
                    "exact_match"
                ].mean(),
            "f1": np.nan
        })

    task_prompt_rows.append(
        row
    )

task_prompt_df = pd.DataFrame(
    task_prompt_rows
)

task_prompt_df.to_csv(
    OUTPUT_DIR /
    "task_prompt_metrics.csv",
    index=False
)


# ============================================================
# QUESTION STABILITY
# ============================================================

stability_rows = []

for question_id, group in df.groupby(
    "question_id"
):
    group = group.sort_values(
        "complexity_level"
    )

    task = group[
        "task"
    ].iloc[0]

    predictions = [
        normalize_prediction(x)
        for x in group[
            "predicted_answer"
        ].tolist()
        if pd.notna(x)
        and normalize_prediction(x)
    ]

    if predictions:
        counts = Counter(
            predictions
        )

        answer_agreement = (
            max(counts.values())
            /
            len(predictions)
        )

        unique_predictions = len(
            counts
        )

    else:
        answer_agreement = 0.0
        unique_predictions = 0

    row = {
        "question_id":
            question_id,
        "task":
            task,
        "answer_agreement_rate":
            answer_agreement,
        "unique_predictions":
            unique_predictions
    }

    # CFR is binary correctness based, so use only
    # objective tasks.
    if task != "reading":
        correctness = [
            int(x)
            for x in group[
                "correct"
            ].tolist()
        ]

        flips = 0
        pairs = 0

        for i in range(
            len(correctness)
        ):
            for j in range(
                i + 1,
                len(correctness)
            ):
                pairs += 1

                if (
                    correctness[i]
                    !=
                    correctness[j]
                ):
                    flips += 1

        cfr = (
            flips / pairs
            if pairs
            else 0.0
        )

        row.update({
            "accuracy_across_prompts":
                float(
                    np.mean(
                        correctness
                    )
                ),
            "correctness_flip_rate":
                cfr
        })

    else:
        row.update({
            "accuracy_across_prompts":
                np.nan,
            "correctness_flip_rate":
                np.nan,
            "mean_reading_f1":
                float(
                    group[
                        "f1"
                    ].mean()
                ),
            "reading_f1_std":
                float(
                    group[
                        "f1"
                    ].std(
                        ddof=0
                    )
                )
        })

    stability_rows.append(
        row
    )

stability_df = pd.DataFrame(
    stability_rows
)

stability_df.to_csv(
    OUTPUT_DIR /
    "question_stability.csv",
    index=False
)


# ============================================================
# PROMPT-TO-PROMPT ADJACENT FLIPS
# ============================================================

flip_rows = []

objective_only = df[
    df["task"] != "reading"
].copy()

for question_id, group in objective_only.groupby(
    "question_id"
):
    group = group.sort_values(
        "complexity_level"
    )

    records = group.to_dict(
        "records"
    )

    for i in range(
        len(records) - 1
    ):
        a = records[i]
        b = records[i + 1]

        flip_rows.append({
            "question_id":
                question_id,
            "task":
                a["task"],
            "from_prompt":
                a["prompt_id"],
            "to_prompt":
                b["prompt_id"],
            "from_correct":
                int(
                    a["correct"]
                ),
            "to_correct":
                int(
                    b["correct"]
                ),
            "correctness_flip":
                int(
                    a["correct"]
                    !=
                    b["correct"]
                )
        })

flip_df = pd.DataFrame(
    flip_rows
)

flip_df.to_csv(
    OUTPUT_DIR /
    "adjacent_prompt_flips.csv",
    index=False
)


# ============================================================
# P0-P4 VS P5-P9
# ============================================================

control = objective_only[
    objective_only[
        "complexity_level"
    ] <= 4
]

intervention = objective_only[
    objective_only[
        "complexity_level"
    ] >= 5
]

group_comparison = pd.DataFrame([
    {
        "group":
            "P0-P4_control_style",
        "n":
            len(control),
        "accuracy":
            control[
                "correct"
            ].mean(),
        "mean_latency_seconds":
            control[
                "latency_seconds"
            ].mean(),
        "mean_output_tokens":
            control[
                "eval_count"
            ].mean(),
        "format_compliance":
            control[
                "format_compliant"
            ].mean()
    },
    {
        "group":
            "P5-P9_reasoning_intervention",
        "n":
            len(intervention),
        "accuracy":
            intervention[
                "correct"
            ].mean(),
        "mean_latency_seconds":
            intervention[
                "latency_seconds"
            ].mean(),
        "mean_output_tokens":
            intervention[
                "eval_count"
            ].mean(),
        "format_compliance":
            intervention[
                "format_compliant"
            ].mean()
    }
])

group_comparison.to_csv(
    OUTPUT_DIR /
    "prompt_group_comparison.csv",
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

objective_prompt_accuracy = (
    prompt_df[
        "objective_accuracy"
    ]
)

objective_accuracy_delta = float(
    objective_prompt_accuracy.max()
    -
    objective_prompt_accuracy.min()
)

objective_stability = (
    stability_df[
        stability_df[
            "task"
        ] != "reading"
    ]
)

summary = {
    "model":
        "phi4-mini:latest",

    "total_records":
        len(df),

    "successful_generations":
        int(
            status_counts[
                "success"
            ]
        ),

    "truncated_generations":
        int(
            status_counts[
                "truncated"
            ]
        ),

    "empty_generations":
        int(
            status_counts[
                "empty"
            ]
        ),

    "error_generations":
        int(
            status_counts[
                "error"
            ]
        ),

    "objective_task_records":
        objective_n,

    "objective_accuracy":
        objective_accuracy,

    "objective_accuracy_ci_low":
        float(
            objective_ci_low
        ),

    "objective_accuracy_ci_high":
        float(
            objective_ci_high
        ),

    "reading_records":
        len(
            reading_df
        ),

    "reading_exact_match":
        reading_em,

    "reading_f1":
        reading_f1,

    "overall_format_compliance":
        float(
            df[
                "format_compliant"
            ].mean()
        ),

    "mean_latency_seconds":
        float(
            df[
                "latency_seconds"
            ].mean()
        ),

    "median_latency_seconds":
        float(
            df[
                "latency_seconds"
            ].median()
        ),

    "mean_output_tokens":
        float(
            df[
                "eval_count"
            ].mean()
        ),

    "objective_accuracy_delta":
        objective_accuracy_delta,

    "mean_correctness_flip_rate":
        float(
            objective_stability[
                "correctness_flip_rate"
            ].mean()
        ),

    "mean_answer_agreement_rate":
        float(
            stability_df[
                "answer_agreement_rate"
            ].mean()
        ),

    "control_P0_P4_accuracy":
        float(
            control[
                "correct"
            ].mean()
        ),

    "intervention_P5_P9_accuracy":
        float(
            intervention[
                "correct"
            ].mean()
        )
}


with (
    OUTPUT_DIR /
    "summary_metrics.json"
).open(
    "w",
    encoding="utf-8"
) as f:
    json.dump(
        summary,
        f,
        indent=2
    )


# ============================================================
# TERMINAL REPORT
# ============================================================

print("=" * 72)
print("PHI-4 MINI — FINAL TASK-AWARE EVALUATION")
print("=" * 72)

print(
    f"Total records          : "
    f"{summary['total_records']}"
)

print(
    f"Successful             : "
    f"{summary['successful_generations']}"
)

print(
    f"Truncated              : "
    f"{summary['truncated_generations']}"
)

print(
    f"Empty                  : "
    f"{summary['empty_generations']}"
)

print(
    f"Errors                 : "
    f"{summary['error_generations']}"
)

print()
print("OBJECTIVE TASKS")
print("-" * 72)

print(
    f"Records                : "
    f"{summary['objective_task_records']}"
)

print(
    f"Accuracy               : "
    f"{summary['objective_accuracy']:.4f}"
)

print(
    f"95% Wilson CI          : "
    f"{summary['objective_accuracy_ci_low']:.4f}"
    f" – "
    f"{summary['objective_accuracy_ci_high']:.4f}"
)

print(
    f"Prompt accuracy delta  : "
    f"{summary['objective_accuracy_delta']:.4f}"
)

print(
    f"Mean CFR               : "
    f"{summary['mean_correctness_flip_rate']:.4f}"
)

print()
print("READING / SQUAD")
print("-" * 72)

print(
    f"Records                : "
    f"{summary['reading_records']}"
)

print(
    f"Exact Match            : "
    f"{summary['reading_exact_match']:.4f}"
)

print(
    f"Token F1               : "
    f"{summary['reading_f1']:.4f}"
)

print()
print("BEHAVIOR")
print("-" * 72)

print(
    f"Format compliance      : "
    f"{summary['overall_format_compliance']:.4f}"
)

print(
    f"Answer agreement       : "
    f"{summary['mean_answer_agreement_rate']:.4f}"
)

print(
    f"Mean latency           : "
    f"{summary['mean_latency_seconds']:.4f}s"
)

print(
    f"Median latency         : "
    f"{summary['median_latency_seconds']:.4f}s"
)

print(
    f"Mean output tokens     : "
    f"{summary['mean_output_tokens']:.2f}"
)

print()
print("PROMPT CONDITION RESULTS")
print("-" * 72)

display_prompt = prompt_df[
    [
        "prompt_id",
        "objective_accuracy",
        "reading_exact_match",
        "reading_f1",
        "format_compliance",
        "mean_latency_seconds",
        "mean_output_tokens"
    ]
].copy()

print(
    display_prompt.to_string(
        index=False
    )
)

print()
print("TASK RESULTS")
print("-" * 72)

print(
    task_df[
        [
            "task",
            "accuracy",
            "exact_match",
            "f1",
            "format_compliance"
        ]
    ].to_string(
        index=False
    )
)

print()
print("PROMPT GROUP COMPARISON")
print("-" * 72)

print(
    group_comparison.to_string(
        index=False
    )
)

print()
print("Saved:")
print(
    OUTPUT_DIR
)

print("=" * 72)
