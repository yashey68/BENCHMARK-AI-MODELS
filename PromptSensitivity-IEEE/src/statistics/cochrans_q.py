from pathlib import Path

import pandas as pd
from statsmodels.stats.contingency_tables import cochrans_q

INPUT = Path("results/processed/cross_model/all_models_evaluated.csv")
OUTPUT = Path("results/processed/cross_model")
OUTPUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [f"P{i}" for i in range(10)]

df = pd.read_csv(INPUT)

# Objective tasks only.
# Reading is evaluated separately using EM/F1.
obj = df[df["task"].isin(
    ["math", "logic", "commonsense", "knowledge"]
)].copy()

obj["correct"] = pd.to_numeric(
    obj["correct"],
    errors="coerce"
)

results = []

print("=" * 78)
print("COCHRAN'S Q — OMNIBUS PROMPT EFFECT ON OBJECTIVE CORRECTNESS")
print("=" * 78)

for model in sorted(obj["model"].unique()):

    m = obj[obj["model"] == model].copy()

    pivot = m.pivot(
        index="question_id",
        columns="prompt_id",
        values="correct"
    )

    pivot = pivot.reindex(columns=PROMPTS)

    if pivot.isna().any().any():
        raise ValueError(
            f"{model}: missing correctness values in repeated-measures matrix"
        )

    if pivot.shape != (40, 10):
        raise ValueError(
            f"{model}: expected 40 objective questions × 10 prompts, "
            f"got {pivot.shape}"
        )

    matrix = pivot.astype(int).to_numpy()

    result = cochrans_q(matrix)

    q = float(result.statistic)
    p = float(result.pvalue)

    significant = p < 0.05

    results.append({
        "model": model,
        "n_questions": matrix.shape[0],
        "n_prompts": matrix.shape[1],
        "cochrans_q": q,
        "df": len(PROMPTS) - 1,
        "p_value": p,
        "significant_0.05": significant,
    })

    print(f"\n{model}")
    print(f"  Questions : {matrix.shape[0]}")
    print(f"  Prompts   : {matrix.shape[1]}")
    print(f"  Q         : {q:.6f}")
    print(f"  df        : {len(PROMPTS) - 1}")
    print(f"  p-value   : {p:.10g}")
    print(
        "  Result    :",
        "SIGNIFICANT prompt effect"
        if significant
        else "No significant prompt effect detected"
    )

out = pd.DataFrame(results)

out.to_csv(
    OUTPUT / "cochrans_q_results.csv",
    index=False
)

print("\n" + "=" * 78)
print("Saved:")
print(OUTPUT / "cochrans_q_results.csv")
print("=" * 78)
