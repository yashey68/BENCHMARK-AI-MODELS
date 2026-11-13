import json
import hashlib
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]

QUESTION_FILE = ROOT / "data/frozen/questions_v1.jsonl"
TEMPLATE_FILE = ROOT / "prompts/templates.yaml"
OUTPUT_FILE = ROOT / "data/processed/benchmark_prompts.jsonl"
MANIFEST_FILE = ROOT / "data/processed/prompt_manifest.json"


def load_jsonl(path):
    rows = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def load_templates():
    with TEMPLATE_FILE.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_task_content(item):

    question = item["question"]

    # SQuAD reading
    if item["task"] == "reading":

        context = item.get("context")

        if not context:
            raise ValueError(
                f"Missing context: {item['id']}"
            )

        return (
            "CONTEXT:\n"
            f"{context}\n\n"
            "QUESTION:\n"
            f"{question}"
        )

    # MCQ tasks
    choices = item.get("choices")

    if choices:

        options = "\n".join(
            f"{label}. {text}"
            for label, text in choices.items()
        )

        return (
            f"{question}\n\n"
            "OPTIONS:\n"
            f"{options}"
        )

    # GSM8K
    return question


def sha256_text(text):
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def main():

    questions = load_jsonl(QUESTION_FILE)
    templates = load_templates()

    if len(questions) != 200:
        raise ValueError(
            f"Expected 200 questions, found {len(questions)}"
        )

    if len({
        q["id"] for q in questions
    }) != 200:
        raise ValueError(
            "Question IDs are not unique."
        )

    expected_prompts = [
        f"P{i}" for i in range(10)
    ]

    if list(templates.keys()) != expected_prompts:
        raise ValueError(
            f"Expected P0-P9, found "
            f"{list(templates.keys())}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    total = 0

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as out:

        for item in questions:

            task_content = build_task_content(
                item
            )

            task_hash = sha256_text(
                task_content
            )

            for prompt_id, prompt_data in templates.items():

                template = prompt_data["template"]

                if "{question}" not in template:
                    raise ValueError(
                        f"{prompt_id} missing "
                        "{question} placeholder"
                    )

                prompt = template.format(
                    question=task_content
                ).strip()

                record = {
                    "question_id": item["id"],
                    "task": item["task"],
                    "source": item["source"],
                    "source_split":
                        item["source_split"],
                    "source_index":
                        item["source_index"],

                    "prompt_id": prompt_id,
                    "prompt_name":
                        prompt_data["name"],
                    "prompt_group":
                        prompt_data["group"],
                    "complexity_level":
                        prompt_data[
                            "complexity_level"
                        ],
                    "manipulation":
                        prompt_data[
                            "manipulation"
                        ],

                    "question":
                        item["question"],
                    "context":
                        item.get("context"),
                    "choices":
                        item.get("choices"),

                    "task_content":
                        task_content,
                    "task_content_sha256":
                        task_hash,

                    "ground_truth":
                        item["answer"],

                    "ground_truth_text":
                        item.get(
                            "answer_text",
                            item["answer"]
                        ),

                    "accepted_answers":
                        item.get(
                            "answers",
                            [item["answer"]]
                        ),

                    "prompt":
                        prompt
                }

                for key in [
                    "subject",
                    "knowledge_domain",
                    "question_concept",
                    "title",
                    "source_id"
                ]:
                    if key in item:
                        record[key] = item[key]

                out.write(
                    json.dumps(
                        record,
                        ensure_ascii=False
                    ) + "\n"
                )

                total += 1

    if total != 2000:
        raise ValueError(
            f"Expected 2000 prompts, generated {total}"
        )

    benchmark_sha = hashlib.sha256(
        QUESTION_FILE.read_bytes()
    ).hexdigest()

    output_sha = hashlib.sha256(
        OUTPUT_FILE.read_bytes()
    ).hexdigest()

    manifest = {
        "benchmark_name":
            "PromptSensitivity-IEEE-v1",
        "benchmark_sha256":
            benchmark_sha,
        "questions": 200,
        "prompt_conditions": 10,
        "total_prompts": total,
        "prompt_ids":
            expected_prompts,
        "prompt_dataset_sha256":
            output_sha
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            manifest,
            indent=2
        ),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("FINAL PROMPT DATASET")
    print("=" * 70)

    print("Questions         : 200")
    print("Prompt conditions : 10")
    print(f"Generated prompts : {total}")

    print("\nBenchmark SHA256:")
    print(benchmark_sha)

    print("\nPrompt dataset SHA256:")
    print(output_sha)

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("=" * 70)


if __name__ == "__main__":
    main()
