from datasets import load_dataset

print("=" * 70)
print("LOGIC DATASET INSPECTION")
print("=" * 70)

candidates = [
    ("tasksource/bigbench", "logical_deduction"),
    ("lukaemon/bbh", "logical_deduction_three_objects"),
]

for dataset_name, config in candidates:
    print(f"\nTrying dataset: {dataset_name}")
    print(f"Configuration : {config}")

    try:
        ds = load_dataset(
            dataset_name,
            config,
            trust_remote_code=False
        )

        print("\nSUCCESS")
        print(ds)

        print("\nAvailable splits:")
        print(list(ds.keys()))

        for split in ds.keys():
            print(f"\n--- {split.upper()} ---")
            print(f"Rows: {len(ds[split])}")
            print(f"Columns: {ds[split].column_names}")

            if len(ds[split]) > 0:
                print("\nFirst example:")
                print(ds[split][0])

        print("\n" + "=" * 70)
        print("USING THIS DATASET")
        print("=" * 70)

        break

    except Exception as e:
        print("\nFAILED")
        print(f"{type(e).__name__}: {e}")

else:
    print("\n" + "=" * 70)
    print("NO LOGIC DATASET COULD BE LOADED")
    print("=" * 70)
