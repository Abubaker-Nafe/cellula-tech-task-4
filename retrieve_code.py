import argparse

from vector_store import get_collection


def retrieve_examples(user_query: str, top_k: int = 3) -> list[dict]:
    """Retrieve programming examples similar to the user's request."""
    user_query = user_query.strip()

    if not user_query:
        raise ValueError("The query cannot be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    collection = get_collection()
    document_count = collection.count()

    if document_count == 0:
        raise ValueError(
            "The knowledge base is empty. Run build_knowledge_base.py."
        )

    results = collection.query(
        query_texts=[user_query], # Supplies the request that Chroma will embed
        n_results=min(top_k, document_count), # Sets how many examples to retrieve.
        include=["documents", "metadatas", "distances"], # Returns document text, metadata, and distances. 
    )

    # Chroma groups results by input query.
    # We submitted one query, so its results are at index 0.
    examples = []

    for record_id, document, metadata, distance in zip(
        results["ids"][0],
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        examples.append({
            "id": record_id,
            "document": document,
            "metadata": metadata or {},
            "distance": float(distance),
        })

    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Retrieve similar programming examples."
    )
    parser.add_argument("query", help="The coding request.")
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    examples = retrieve_examples(args.query, top_k=args.top_k)

    print(f"Query: {args.query}")
    print(f"Retrieved examples: {len(examples)}")

    for rank, example in enumerate(examples, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Task ID: {example['id']}")
        print(
            "Function:",
            example["metadata"].get("entry_point", "unknown"),
        )
        print(f"Cosine distance: {example['distance']:.4f}")
        print("\nDocument preview:")
        print(example["document"][:1000])

        if len(example["document"]) > 1000:
            print("\n[Preview shortened; full document was retrieved.]")


if __name__ == "__main__":
    main()