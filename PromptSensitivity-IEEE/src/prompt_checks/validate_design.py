import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = ROOT / "data/processed/dev_prompts.jsonl"
OUTPUT_FILE = ROOT / "results/prompt_checks/design_validation.csv"


DESIGN = {
    "P0": {
        "task_preserved": 1,
        "final_answer_required": 1,
        "reasoning_guidance": 0,
        "verification_instruction": 0,
        "expert_persona": 0,
        "lexical_sophistication": 0,
        "dense_instruction": 0,
    },

    "P1": {
        "task_preserved": 1,
        "final_answer_required": 1,
        "reasoning_guidance": 0,
        "verification_instruction": 0,
        "expert_persona": 0,
        "lexical_sophistication": 0,
        "dense_instruction": 0,
    },

    "P2": {
        "task_preserved": 1,
        "final_answer_required": 1,
        "reasoning_guidance": 0,
        "verification_instruction": 0,
        "expert_persona": 0,
        "lexical_sophistication": 0,
        "dense_instruction": 0,
    },

    "P3": {
        "task_preserved": 1,
        "final_answer_required": 1,
        "reasoning_guidance": 0,
        "verification_instruction": 0,
        "expert_persona": 0,
        "lexical_sophistication": 0,
        "dense_instruction": 0,
    },

    "P4": {
        "task_preserved": 1,
        "final_answer_required": 1,
        "reasoning_guidance": 0,
        "verification_instruction": 0,
        "expert_persona": 0,
        "lexical_sophistication": 0,
        "dense_instruction": 0,
    },

    "P5": {
        "task_preserved": 1,
        "final_answer_required": 1,
        "reasoning_guidance": 1,
        "verification_instruction": 0,
        "expert_persona": 0,
        "lexical_sophistication": 0,
        "dense_instruction": 0,
    },

    "P6": {
        "task_preserved": 1,
        "final_answer_required": 1,
        "reasoning_guidance": 1,
        "verification_instruction": 1,
        "expert_persona": 0,
        "lexical_sophistication": 0,
        "dense_instruction": 0,
    },

    "P7": {
        "task_preserved": 1,
        "final_answer_required": 1,
        "reasoning_guidance": 1,
        "verification_instruction": 1,
        "expert_persona": 1,
        "lexical_sophistication": 0,
        "dense_instruction": 0,
    },

    "P8": {
        "task_preserved": 1,
        "final_answer_required": 1,
        "reasoning_guidance": 1,
        "verification_instruction": 1,
        "expert_persona": 0,
        "lexical_sophistication": 1,
        "dense_instruction": 0,
    },

    "P9": {
        "task_preserved": 1,
        "final_answer_required": 1,
        "reasoning_guidance": 1,
        "verification_instruction": 1,
        "expert_persona": 0,
        "lexical_sophistication": 1,
        "dense_instruction": 1,
    },
}


def load_prompt_metadata():

    metadata = {}

    with INPUT_FILE.open("r", encoding="utf-8") as f:

        for line in f:

            row = json.loads(line)

            pid = row["prompt_id"]

            if pid not in metadata:

                metadata[pid] = {
                    "prompt_name": row["prompt_name"],
                    "prompt_group": row["prompt_group"],
                    "complexity_level": row["complexity_level"],
                }

    return metadata


def main():

    metadata = load_prompt_metadata()

    rows = []

    for pid, design in DESIGN.items():

        row = {
            "prompt_id": pid,
            **metadata[pid],
            **design,
        }

        row["intervention_count"] = sum([
            design["reasoning_guidance"],
            design["verification_instruction"],
            design["expert_persona"],
            design["lexical_sophistication"],
            design["dense_instruction"],
        ])

        rows.append(row)

    df = pd.DataFrame(rows)

    df = df.sort_values("complexity_level")

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 105)
    print("EXPERIMENTAL DESIGN VALIDATION")
    print("=" * 105)

    columns = [
        "prompt_id",
        "complexity_level",
        "task_preserved",
        "final_answer_required",
        "reasoning_guidance",
        "verification_instruction",
        "expert_persona",
        "lexical_sophistication",
        "dense_instruction",
        "intervention_count",
    ]

    print(
        df[columns].to_string(
            index=False
        )
    )

    print("\n" + "=" * 105)

    print("CONTROL REGION:")
    print("P0-P4 = linguistic / structural manipulation")

    print("\nINTERVENTION REGION:")
    print("P5-P9 = explicit reasoning / verification / complexity intervention")

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("=" * 105)


if __name__ == "__main__":
    main()
