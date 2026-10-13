import ast
import json
from pathlib import Path

import pyarrow.parquet as pq


def prepare_documents(dataset_path: Path) -> list[dict]:
    """Convert dataset rows into complete retrieval documents."""
    rows = pq.read_table(dataset_path).to_pylist()

    documents = []
    seen_ids = set()

    required_fields = (
        "task_id",
        "prompt",
        "canonical_solution",
        "test",
        "entry_point",
    )

    for row_number, row in enumerate(rows, start=1):
        for field in required_fields:
            value = row.get(field)

            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    f"Row {row_number}: missing or empty '{field}'."
                )

        task_id = row["task_id"]

        if task_id in seen_ids:
            raise ValueError(f"Duplicate task ID: {task_id}")

        seen_ids.add(task_id)

        # Preserve the original indentation in the solution body.
        full_code = row["prompt"] + row["canonical_solution"]

        # Check Python syntax without executing the code.
        ast.parse(full_code, filename=task_id)

        document_text = (
            f"Python programming example\n"
            f"Task ID: {task_id}\n"
            f"Function: {row['entry_point']}\n\n"
            f"Problem description and reference implementation:\n"
            f"```python\n{full_code}\n```"
        )

        documents.append({
            "id": task_id,
            "document": document_text,
            "metadata": {
                "source": "HumanEval",
                "task_id": task_id,
                "entry_point": row["entry_point"],
                "language": "python",
            },
            "test": row["test"],
        })

    return documents


def main():
    project_dir = Path(__file__).resolve().parent
    dataset_path = (
        project_dir / "data" / "test-00000-of-00001.parquet"
    )

    if not dataset_path.is_file():
        raise SystemExit(f"Dataset not found: {dataset_path}")

    documents = prepare_documents(dataset_path)

    if not documents:
        raise SystemExit("The dataset is empty.")

    output_dir = project_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "humaneval_documents.json"
    output_path.write_text(
        json.dumps(documents, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Prepared documents: {len(documents)}")
    print("All reference implementations passed syntax checks.")
    print(f"Saved to: {output_path}")
    print("\nFirst retrieval document:\n")
    print(documents[0]["document"])


if __name__ == "__main__":
    main()