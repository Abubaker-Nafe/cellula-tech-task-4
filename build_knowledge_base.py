import json

from vector_store import CHROMA_DIR, PROJECT_DIR, get_collection


BATCH_SIZE = 32


def main():
    documents_path = (
        PROJECT_DIR / "output" / "humaneval_documents.json"
    )

    if not documents_path.is_file():
        raise SystemExit(
            "Prepared documents not found. Run prepare_dataset.py first."
        )

    records = json.loads(
        documents_path.read_text(encoding="utf-8")
    )

    if not records:
        raise SystemExit("The prepared document list is empty.")

    collection = get_collection()

    print(f"Documents to process: {len(records)}")
    print("Creating embeddings and storing documents...")
    print("The first batch may take longer while the model downloads.")

    for start in range(0, len(records), BATCH_SIZE):
        batch = records[start:start + BATCH_SIZE]

        collection.upsert(
            ids=[record["id"] for record in batch],
            documents=[record["document"] for record in batch],
            metadatas=[record["metadata"] for record in batch],
        )

        processed = start + len(batch)
        print(f"Processed: {processed}/{len(records)}")

    sample = collection.get(
        ids=[records[0]["id"]],
        include=["embeddings"],
    )

    embedding_dimension = len(sample["embeddings"][0])

    print("\nKnowledge base ready.")
    print(f"Stored documents: {collection.count()}")
    print(f"Embedding dimension: {embedding_dimension}")
    print(f"Database folder: {CHROMA_DIR}")


if __name__ == "__main__":
    main()