import json
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/dev_prompts.jsonl"
OUTPUT_FILE = ROOT / "results/prompt_checks/semantic_metrics.csv"
SUMMARY_FILE = ROOT / "results/prompt_checks/semantic_summary.csv"

# Fixed embedding model for the entire experiment.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_records():
    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    return records


def instruction_only(prompt, question):
    """
    Replace the benchmark question so semantic similarity measures
    the instruction rather than the question itself.
    """
    return prompt.replace(question, "[QUESTION]")


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


def main():

    print("=" * 80)
    print("PROMPT SEMANTIC VALIDATION")
    print("=" * 80)

    print(f"Loading embedding model:")
    print(EMBEDDING_MODEL)
    print()

    model = SentenceTransformer(EMBEDDING_MODEL)

    records = load_records()

    # Since prompt templates are identical across questions,
    # obtain one representative instruction for each P0-P9.
    prompts = {}

    metadata = {}

    for record in records:

        pid = record["prompt_id"]

        if pid not in prompts:

            instruction = instruction_only(
                record["prompt"],
                record["question"]
            )

            prompts[pid] = instruction

            metadata[pid] = {
                "prompt_name": record["prompt_name"],
                "prompt_group": record["prompt_group"],
                "complexity_level": record["complexity_level"],
                "manipulation": record["manipulation"],
            }

    ordered_ids = sorted(
        prompts.keys(),
        key=lambda x: int(x[1:])
    )

    texts = [prompts[pid] for pid in ordered_ids]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    baseline_embedding = embeddings[0]

    rows = []

    for pid, embedding in zip(ordered_ids, embeddings):

        similarity = cosine_similarity(
            baseline_embedding,
            embedding
        )

        row = {
            "prompt_id": pid,
            "prompt_name": metadata[pid]["prompt_name"],
            "prompt_group": metadata[pid]["prompt_group"],
            "complexity_level": metadata[pid]["complexity_level"],
            "manipulation": metadata[pid]["manipulation"],
            "semantic_similarity_vs_p0": round(similarity, 6),
        }

        rows.append(row)

    df = pd.DataFrame(rows)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # Full pairwise semantic similarity matrix
    matrix = np.matmul(
        embeddings,
        embeddings.T
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=ordered_ids,
        columns=ordered_ids
    )

    matrix_file = (
        ROOT
        / "results/prompt_checks/semantic_similarity_matrix.csv"
    )

    matrix_df.to_csv(matrix_file)

    # Save same data as summary for later merge
    df.to_csv(
        SUMMARY_FILE,
        index=False
    )

    print(
        df[
            [
                "prompt_id",
                "complexity_level",
                "prompt_group",
                "semantic_similarity_vs_p0",
            ]
        ].to_string(index=False)
    )

    print("\n" + "-" * 80)

    print("Saved semantic metrics:")
    print(OUTPUT_FILE)

    print("\nSaved similarity matrix:")
    print(matrix_file)

    print("=" * 80)


if __name__ == "__main__":
    main()
