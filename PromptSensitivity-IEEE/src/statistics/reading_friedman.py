from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare

INPUT = Path(
    "results/processed/cross_model/all_models_evaluated.csv"
)

OUTPUT = Path(
    "results/processed/cross_model"
)

PROMPTS = [f"P{i}" for i in range(10)]

df = pd.read_csv(INPUT)

reading = df[
    df["task"] == "reading"
].copy()

reading["f1"] = pd.to_numeric(
    reading["f1"],
    errors="coerce"
)

results = []

print("=" * 78)
print("FRIEDMAN TEST — PROMPT EFFECT ON READING F1")
print("=" * 78)

for model in sorted(reading["model"].unique()):

    m = reading[
        reading["model"] == model
    ]

    pivot = m.pivot(
        index="question_id",
        columns="prompt_id",
        values="f1"
    ).reindex(columns=PROMPTS)

    if pivot.shape != (10, 10):
        raise ValueError(
            f"{model}: expected 10 questions × 10 prompts, "
            f"got {pivot.shape}"
        )

    if pivot.isna().any().any():
        raise ValueError(
            f"{model}: missing reading F1 values"
        )

    samples = [
        pivot[p].to_numpy()
        for p in PROMPTS
    ]

    stat, p = friedmanchisquare(
        *samples
    )

    # Kendall's W effect size for Friedman:
    # W = chi-square / (n * (k - 1))
    n = pivot.shape[0]
    k = pivot.shape[1]

    kendalls_w = stat / (
        n * (k - 1)
    )

    significant = p < 0.05

    results.append({
        "model": model,
        "n_questions": n,
        "n_prompts": k,
        "friedman_chi_square": stat,
        "df": k - 1,
        "p_value": p,
        "kendalls_w": kendalls_w,
        "significant_0.05": significant,
    })

    print(f"\n{model}")
    print(f"  Questions   : {n}")
    print(f"  Prompts     : {k}")
    print(f"  Chi-square  : {stat:.6f}")
    print(f"  df          : {k - 1}")
    print(f"  p-value     : {p:.10g}")
    print(f"  Kendall's W : {kendalls_w:.6f}")

    print(
        "  Result      :",
        "SIGNIFICANT prompt effect"
        if significant
        else "No significant prompt effect detected"
    )

out = pd.DataFrame(results)

out.to_csv(
    OUTPUT / "reading_friedman_results.csv",
    index=False
)

print("\n" + "=" * 78)
print("Saved:")
print(
    OUTPUT /
    "reading_friedman_results.csv"
)
print("=" * 78)
