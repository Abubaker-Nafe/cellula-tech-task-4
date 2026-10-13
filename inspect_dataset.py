from pathlib import Path

import pyarrow.parquet as pq


def main():
    dataset_path = (
        Path(__file__).resolve().parent
        / "data"
        / "test-00000-of-00001.parquet"
    )

    if not dataset_path.is_file():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    parquet_file = pq.ParquetFile(dataset_path)

    print(f"Total rows: {parquet_file.metadata.num_rows}")
    print("\nColumns and types:")
    print(parquet_file.schema_arrow)

    print("\nFirst two records:")
    batch = next(parquet_file.iter_batches(batch_size=2), None)

    if batch is None:
        print("The dataset is empty.")
        return

    for number, record in enumerate(batch.to_pylist(), start=1):
        print(f"\n--- Record {number} ---")
        for column, value in record.items():
            # Limit long fields to keep terminal output manageable.
            print(f"\n{column}:\n{str(value)[:2000]}")


if __name__ == "__main__":
    main()