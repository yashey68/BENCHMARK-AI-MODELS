from datasets import load_dataset

print("=" * 70)
print("COMMONSENSEQA DATASET INSPECTION")
print("=" * 70)

candidates = [
    ("tau/commonsense_qa", None),
    ("commonsense_qa", None),
]

for dataset_name, config in candidates:

    print(f"\nTrying dataset: {dataset_name}")

    try:
        if config is None:
            ds = load_dataset(dataset_name)
        else:
            ds = load_dataset(dataset_name, config)

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
        print(
            f"{type(e).__name__}: {e}"
        )

else:

    print("\n" + "=" * 70)
    print("NO COMMONSENSE DATASET COULD BE LOADED")
    print("=" * 70)
