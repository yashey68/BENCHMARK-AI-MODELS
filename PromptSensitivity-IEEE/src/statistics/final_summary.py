from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("results")
CROSS = ROOT / "processed/cross_model"

models = {
    "Phi-4 Mini": "phi4-mini",
    "Gemma 3 12B": "gemma3-12b",
    "Qwen 3.5": "qwen3.5",
    "Mistral Small 3.2": "mistral-small3.2",
}

cochran = pd.read_csv(CROSS / "cochrans_q_results.csv")
friedman = pd.read_csv(CROSS / "reading_friedman_results.csv")
aar = pd.read_csv(CROSS / "pairwise_aar_summary.csv")
mcnemar = pd.read_csv(CROSS / "mcnemar_summary.csv")


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


rows = []

print("=" * 90)
print("BUILDING FINAL CROSS-MODEL SUMMARY")
print("=" * 90)

for model, dirname in models.items():

    evaluated = pd.read_csv(
        ROOT / "processed" / dirname / "evaluated_responses.csv"
    )

    objective = evaluated[
        evaluated["task"].isin(
            ["math", "logic", "commonsense", "knowledge"]
        )
    ].copy()

    reading = evaluated[
        evaluated["task"] == "reading"
    ].copy()

    objective["correct_num"] = pd.to_numeric(
        objective["correct"],
        errors="coerce"
    ).fillna(0)

    accuracy = objective["correct_num"].mean()

    prompt_acc = (
        objective
        .groupby("prompt_id")["correct_num"]
        .mean()
    )

    accuracy_delta = prompt_acc.max() - prompt_acc.min()

    reading_f1 = pd.to_numeric(
        reading["f1"],
        errors="coerce"
    ).fillna(0).mean()

    reading_em = pd.to_numeric(
        reading["exact_match"],
        errors="coerce"
    ).fillna(0).mean()

    format_col = find_column(
        evaluated,
        [
            "format_compliant",
            "format_compliance",
            "instruction_following",
        ]
    )

    latency_col = find_column(
        evaluated,
        [
            "latency_seconds",
            "latency",
            "response_time_seconds",
        ]
    )

    token_col = find_column(
        evaluated,
        [
            "output_tokens",
            "eval_count",
            "response_tokens",
            "completion_tokens",
        ]
    )

    if format_col:
        format_compliance = pd.to_numeric(
            evaluated[format_col],
            errors="coerce"
        ).fillna(0).mean()
    else:
        format_compliance = np.nan

    if latency_col:
        mean_latency = pd.to_numeric(
            evaluated[latency_col],
            errors="coerce"
        ).mean()
    else:
        mean_latency = np.nan

    if token_col:
        mean_tokens = pd.to_numeric(
            evaluated[token_col],
            errors="coerce"
        ).mean()
    else:
        mean_tokens = np.nan

    truncations = int(
        (evaluated["status"] == "truncated").sum()
    )

    cq = cochran[
        cochran["model"] == model
    ].iloc[0]

    fr = friedman[
        friedman["model"] == model
    ].iloc[0]

    ar = aar[
        aar["model"] == model
    ].iloc[0]

    mc = mcnemar[
        mcnemar["model"] == model
    ].iloc[0]

    stability = pd.read_csv(
        ROOT / "processed" / dirname /
        "question_stability.csv"
    )

    cfr_col = find_column(
        stability,
        [
            "correctness_flip_rate",
            "cfr",
            "pairwise_cfr",
        ]
    )

    if cfr_col:
        mean_cfr = pd.to_numeric(
            stability[cfr_col],
            errors="coerce"
        ).mean()
    else:
        # Recalculate pairwise correctness flip rate directly
        cfr_values = []

        for qid, g in objective.groupby("question_id"):
            g = g.sort_values("prompt_id")

            values = g["correct_num"].astype(int).tolist()

            flips = 0
            pairs = 0

            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    pairs += 1
                    if values[i] != values[j]:
                        flips += 1

            if pairs:
                cfr_values.append(flips / pairs)

        mean_cfr = (
            sum(cfr_values) / len(cfr_values)
            if cfr_values else np.nan
        )

    print(f"\n{model}")
    print(f"  token column   : {token_col}")
    print(f"  latency column : {latency_col}")
    print(f"  format column  : {format_col}")
    print(f"  CFR column     : {cfr_col or 'recalculated'}")

    rows.append({
        "Model": model,
        "Objective Accuracy": accuracy,
        "Prompt Accuracy Delta": accuracy_delta,
        "Mean CFR": mean_cfr,
        "Pairwise AAR": ar["mean_pairwise_aar"],
        "Reading EM": reading_em,
        "Reading F1": reading_f1,
        "Format Compliance": format_compliance,
        "Mean Latency (s)": mean_latency,
        "Mean Output Tokens": mean_tokens,
        "Truncations": truncations,
        "Cochran Q": cq["cochrans_q"],
        "Cochran p": cq["p_value"],
        "Reading Friedman Chi2": fr["friedman_chi_square"],
        "Reading p": fr["p_value"],
        "Reading Kendall W": fr["kendalls_w"],
        "Holm-significant pairs": int(
            mc["holm_significant"]
        ),
    })


summary = pd.DataFrame(rows)

summary.to_csv(
    CROSS / "final_model_summary.csv",
    index=False
)

paper = summary.copy()

percent_cols = [
    "Objective Accuracy",
    "Prompt Accuracy Delta",
    "Mean CFR",
    "Pairwise AAR",
    "Reading EM",
    "Reading F1",
    "Format Compliance",
]

for col in percent_cols:
    paper[col] = (paper[col] * 100).round(2)

paper["Mean Latency (s)"] = (
    paper["Mean Latency (s)"].round(3)
)

paper["Mean Output Tokens"] = (
    paper["Mean Output Tokens"].round(2)
)

paper["Cochran Q"] = (
    paper["Cochran Q"].round(3)
)

paper["Cochran p"] = paper[
    "Cochran p"
].map(lambda x: f"{x:.3g}")

paper["Reading Friedman Chi2"] = (
    paper["Reading Friedman Chi2"].round(3)
)

paper["Reading p"] = paper[
    "Reading p"
].map(lambda x: f"{x:.3g}")

paper["Reading Kendall W"] = (
    paper["Reading Kendall W"].round(3)
)

paper.to_csv(
    CROSS / "paper_ready_model_summary.csv",
    index=False
)

print("\n" + "=" * 120)
print("FINAL CROSS-MODEL PAPER SUMMARY")
print("=" * 120)

print(
    paper.to_string(
        index=False
    )
)

print("\nSaved:")
print(CROSS / "final_model_summary.csv")
print(CROSS / "paper_ready_model_summary.csv")
print("=" * 120)
