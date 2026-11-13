from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests

INPUT = Path(
    "results/processed/cross_model/all_models_evaluated.csv"
)
OUTPUT = Path("results/processed/cross_model")
OUTPUT.mkdir(parents=True, exist_ok=True)

PROMPTS = [f"P{i}" for i in range(10)]

df = pd.read_csv(INPUT)

obj = df[
    df["task"].isin(
        ["math", "logic", "commonsense", "knowledge"]
    )
].copy()

obj["correct"] = pd.to_numeric(
    obj["correct"],
    errors="coerce"
)

all_results = []

print("=" * 82)
print("PAIRWISE EXACT McNEMAR TESTS + HOLM CORRECTION")
print("=" * 82)

for model in sorted(obj["model"].unique()):

    m = obj[obj["model"] == model]

    pivot = m.pivot(
        index="question_id",
        columns="prompt_id",
        values="correct"
    ).reindex(columns=PROMPTS)

    if pivot.shape != (40, 10):
        raise ValueError(
            f"{model}: expected 40 × 10, got {pivot.shape}"
        )

    if pivot.isna().any().any():
        raise ValueError(
            f"{model}: missing correctness values"
        )

    model_results = []

    for p1, p2 in combinations(PROMPTS, 2):

        a = pivot[p1].astype(int)
        b = pivot[p2].astype(int)

        # Paired 2x2 table:
        #
        #               p2=0   p2=1
        # p1=0           n00    n01
        # p1=1           n10    n11

        n00 = int(((a == 0) & (b == 0)).sum())
        n01 = int(((a == 0) & (b == 1)).sum())
        n10 = int(((a == 1) & (b == 0)).sum())
        n11 = int(((a == 1) & (b == 1)).sum())

        table = [
            [n00, n01],
            [n10, n11],
        ]

        discordant = n01 + n10

        if discordant == 0:
            p_value = 1.0
        else:
            test = mcnemar(
                table,
                exact=True,
                correction=False
            )
            p_value = float(test.pvalue)

        acc1 = float(a.mean())
        acc2 = float(b.mean())

        row = {
            "model": model,
            "prompt_1": p1,
            "prompt_2": p2,
            "accuracy_1": acc1,
            "accuracy_2": acc2,
            "accuracy_difference_p2_minus_p1": acc2 - acc1,
            "n00_both_incorrect": n00,
            "n01_p1_wrong_p2_correct": n01,
            "n10_p1_correct_p2_wrong": n10,
            "n11_both_correct": n11,
            "discordant_pairs": discordant,
            "mcnemar_p_raw": p_value,
        }

        model_results.append(row)

    raw_p = [r["mcnemar_p_raw"] for r in model_results]

    reject, p_holm, _, _ = multipletests(
        raw_p,
        alpha=0.05,
        method="holm"
    )

    for row, corrected, sig in zip(
        model_results,
        p_holm,
        reject
    ):
        row["mcnemar_p_holm"] = float(corrected)
        row["significant_holm_0.05"] = bool(sig)
        all_results.append(row)

    significant = [
        r for r in model_results
        if r.get("significant_holm_0.05")
    ]

    print(f"\n{model}")
    print(f"  Pairwise tests        : {len(model_results)}")
    print(
        f"  Holm-significant pairs: "
        f"{len(significant)}"
    )

    if significant:
        print("\n  Significant comparisons:")

        significant = sorted(
            significant,
            key=lambda x: x["mcnemar_p_holm"]
        )

        for r in significant:
            print(
                f'    {r["prompt_1"]} vs {r["prompt_2"]} | '
                f'acc={r["accuracy_1"]:.3f} vs '
                f'{r["accuracy_2"]:.3f} | '
                f'Δ={r["accuracy_difference_p2_minus_p1"]:+.3f} | '
                f'raw p={r["mcnemar_p_raw"]:.6g} | '
                f'Holm p={r["mcnemar_p_holm"]:.6g}'
            )
    else:
        print(
            "  No individual pair remains significant "
            "after Holm correction."
        )

results = pd.DataFrame(all_results)

results.to_csv(
    OUTPUT / "mcnemar_pairwise_holm.csv",
    index=False
)

summary = (
    results
    .groupby("model")
    .agg(
        comparisons=("mcnemar_p_raw", "size"),
        raw_significant=(
            "mcnemar_p_raw",
            lambda x: int((x < 0.05).sum())
        ),
        holm_significant=(
            "significant_holm_0.05",
            "sum"
        ),
        minimum_raw_p=("mcnemar_p_raw", "min"),
        minimum_holm_p=("mcnemar_p_holm", "min"),
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT / "mcnemar_summary.csv",
    index=False
)

print("\n" + "=" * 82)
print("SUMMARY")
print("=" * 82)

print(
    summary.to_string(
        index=False
    )
)

print("\nSaved:")
print(OUTPUT / "mcnemar_pairwise_holm.csv")
print(OUTPUT / "mcnemar_summary.csv")
print("=" * 82)
