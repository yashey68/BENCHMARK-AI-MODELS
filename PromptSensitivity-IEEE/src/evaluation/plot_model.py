from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser(
    description="Generate standardized prompt-sensitivity figures."
)

parser.add_argument(
    "--name",
    required=True,
    help="Processed-results directory name, e.g. gemma3-12b"
)

parser.add_argument(
    "--label",
    required=True,
    help="Human-readable model label used in figure titles."
)

args = parser.parse_args()

DATA_DIR = ROOT / "results/processed" / args.name
FIG_DIR = ROOT / "results/figures" / args.name
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_LABEL = args.label

prompt = pd.read_csv(DATA_DIR / "prompt_metrics.csv")
task = pd.read_csv(DATA_DIR / "task_metrics.csv")
task_prompt = pd.read_csv(DATA_DIR / "task_prompt_metrics.csv")
stability = pd.read_csv(DATA_DIR / "question_stability.csv")
flips = pd.read_csv(DATA_DIR / "adjacent_prompt_flips.csv")

prompt["prompt_num"] = (
    prompt["prompt_id"]
    .str.extract(r"(\d+)")
    .astype(int)
)

prompt = prompt.sort_values("prompt_num")

PROMPTS = prompt["prompt_id"].tolist()


def save(name):
    plt.tight_layout()

    plt.savefig(
        FIG_DIR / f"{name}.pdf",
        bbox_inches="tight"
    )

    plt.savefig(
        FIG_DIR / f"{name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 1. OBJECTIVE ACCURACY VS PROMPT WITH 95% CI
# ============================================================

x = np.arange(len(prompt))

accuracy = prompt["objective_accuracy"].to_numpy()

lower = (
    accuracy
    - prompt["objective_accuracy_ci_low"].to_numpy()
)

upper = (
    prompt["objective_accuracy_ci_high"].to_numpy()
    - accuracy
)

plt.figure(figsize=(8, 4.8))

plt.errorbar(
    x,
    accuracy,
    yerr=[lower, upper],
    marker="o",
    capsize=4,
    linewidth=1.5
)

plt.xticks(x, PROMPTS)
plt.ylim(0, 0.60)

plt.xlabel("Prompt Condition")
plt.ylabel("Objective Accuracy")
plt.title(f"{MODEL_LABEL}: Accuracy Across Prompt Conditions")

plt.grid(
    axis="y",
    alpha=0.25
)

save("accuracy_vs_prompt")


# ============================================================
# 2. TASK × PROMPT PERFORMANCE HEATMAP
#
# Objective tasks = accuracy
# Reading = token F1
# ============================================================

tasks = [
    "math",
    "logic",
    "commonsense",
    "knowledge",
    "reading"
]

matrix = []

for t in tasks:
    row = []

    for p in PROMPTS:
        r = task_prompt[
            (task_prompt["task"] == t)
            &
            (task_prompt["prompt_id"] == p)
        ]

        if r.empty:
            row.append(np.nan)

        elif t == "reading":
            row.append(
                float(r["f1"].iloc[0])
            )

        else:
            row.append(
                float(r["accuracy"].iloc[0])
            )

    matrix.append(row)

matrix = np.array(matrix)

plt.figure(figsize=(9, 4.8))

im = plt.imshow(
    matrix,
    aspect="auto",
    vmin=0,
    vmax=1
)

plt.colorbar(
    im,
    label="Performance"
)

plt.xticks(
    np.arange(len(PROMPTS)),
    PROMPTS
)

plt.yticks(
    np.arange(len(tasks)),
    [
        "Math",
        "Logic",
        "Commonsense",
        "Knowledge",
        "Reading (F1)"
    ]
)

plt.xlabel("Prompt Condition")
plt.ylabel("Task")

plt.title(
    f"{MODEL_LABEL}: Task Performance Across Prompt Conditions"
)

for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        if not np.isnan(matrix[i, j]):
            plt.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8
            )

save("task_prompt_heatmap")


# ============================================================
# 3. READING EM AND F1 VS PROMPT
# ============================================================

plt.figure(figsize=(8, 4.8))

plt.plot(
    x,
    prompt["reading_exact_match"],
    marker="o",
    label="Exact Match"
)

plt.plot(
    x,
    prompt["reading_f1"],
    marker="s",
    label="Token F1"
)

plt.xticks(x, PROMPTS)
plt.ylim(0, 1)

plt.xlabel("Prompt Condition")
plt.ylabel("Score")

plt.title(
    f"{MODEL_LABEL}: Reading Performance Across Prompt Conditions"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)

save("reading_em_f1_vs_prompt")


# ============================================================
# 4. LATENCY VS PROMPT
# ============================================================

plt.figure(figsize=(8, 4.8))

plt.plot(
    x,
    prompt["mean_latency_seconds"],
    marker="o",
    label="Mean"
)

plt.plot(
    x,
    prompt["median_latency_seconds"],
    marker="s",
    label="Median"
)

plt.xticks(x, PROMPTS)

plt.xlabel("Prompt Condition")
plt.ylabel("Latency (seconds)")

plt.title(
    f"{MODEL_LABEL}: Generation Latency Across Prompt Conditions"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)

save("latency_vs_prompt")


# ============================================================
# 5. OUTPUT TOKENS VS PROMPT
# ============================================================

plt.figure(figsize=(8, 4.8))

plt.plot(
    x,
    prompt["mean_output_tokens"],
    marker="o",
    label="Mean"
)

plt.plot(
    x,
    prompt["median_output_tokens"],
    marker="s",
    label="Median"
)

plt.xticks(x, PROMPTS)

plt.xlabel("Prompt Condition")
plt.ylabel("Output Tokens")

plt.title(
    f"{MODEL_LABEL}: Output Length Across Prompt Conditions"
)

plt.legend()

plt.grid(
    axis="y",
    alpha=0.25
)

save("output_tokens_vs_prompt")


# ============================================================
# 6. FORMAT COMPLIANCE VS PROMPT
# ============================================================

plt.figure(figsize=(8, 4.8))

plt.bar(
    x,
    prompt["format_compliance"]
)

plt.xticks(x, PROMPTS)
plt.ylim(0, 1.05)

plt.xlabel("Prompt Condition")
plt.ylabel("Format Compliance Rate")

plt.title(
    f"{MODEL_LABEL}: Instruction-Format Compliance"
)

plt.grid(
    axis="y",
    alpha=0.25
)

save("format_compliance_vs_prompt")


# ============================================================
# 7. ADJACENT CORRECTNESS FLIPS
# ============================================================

flip_summary = (
    flips
    .groupby(
        ["from_prompt", "to_prompt"]
    )["correctness_flip"]
    .mean()
    .reset_index()
)

flip_summary["from_num"] = (
    flip_summary["from_prompt"]
    .str.extract(r"(\d+)")
    .astype(int)
)

flip_summary = flip_summary.sort_values(
    "from_num"
)

labels = (
    flip_summary["from_prompt"]
    + "→"
    + flip_summary["to_prompt"]
)

plt.figure(figsize=(9, 4.8))

plt.bar(
    np.arange(len(flip_summary)),
    flip_summary["correctness_flip"]
)

plt.xticks(
    np.arange(len(flip_summary)),
    labels,
    rotation=35,
    ha="right"
)

plt.ylim(0, 1)

plt.xlabel("Adjacent Prompt Transition")
plt.ylabel("Correctness Flip Rate")

plt.title(
    f"{MODEL_LABEL}: Correctness Instability Between Adjacent Prompts"
)

plt.grid(
    axis="y",
    alpha=0.25
)

save("adjacent_correctness_flips")


# ============================================================
# 8. QUESTION-LEVEL CFR DISTRIBUTION
# ============================================================

objective_stability = stability[
    stability["task"] != "reading"
].dropna(
    subset=["correctness_flip_rate"]
)

plt.figure(figsize=(7, 4.8))

plt.hist(
    objective_stability["correctness_flip_rate"],
    bins=np.linspace(0, 1, 11),
    edgecolor="black"
)

plt.xlabel("Question-Level Pairwise Correctness Flip Rate")
plt.ylabel("Number of Questions")

plt.title(
    f"{MODEL_LABEL}: Distribution of Prompt Sensitivity"
)

plt.grid(
    axis="y",
    alpha=0.25
)

save("correctness_flip_distribution")


# ============================================================
# 9. TASK-LEVEL OBJECTIVE ACCURACY
# ============================================================

objective_tasks = task[
    task["task"] != "reading"
].copy()

objective_tasks = objective_tasks.sort_values(
    "accuracy",
    ascending=False
)

plt.figure(figsize=(7, 4.8))

plt.bar(
    objective_tasks["task"],
    objective_tasks["accuracy"]
)

plt.ylim(0, 1)

plt.xlabel("Task")
plt.ylabel("Accuracy")

plt.title(
    f"{MODEL_LABEL}: Accuracy by Objective Task"
)

plt.grid(
    axis="y",
    alpha=0.25
)

save("task_accuracy")


# ============================================================
# 10. TRUNCATION RATE
# ============================================================

plt.figure(figsize=(8, 4.8))

plt.bar(
    x,
    prompt["truncation_rate"]
)

plt.xticks(x, PROMPTS)

plt.xlabel("Prompt Condition")
plt.ylabel("Truncation Rate")

plt.title(
    f"{MODEL_LABEL}: Generation Truncation by Prompt"
)

plt.grid(
    axis="y",
    alpha=0.25
)

save("truncation_rate_vs_prompt")


print("=" * 68)
print(f"{MODEL_LABEL.upper()} FIGURES GENERATED")
print("=" * 68)

for path in sorted(FIG_DIR.glob("*")):
    print(path.name)

print()
print("PDF : vector figures for IEEE paper")
print("PNG : 300-DPI figures for inspection/presentation")
print()
print("Saved:")
print(FIG_DIR)
print("=" * 68)
