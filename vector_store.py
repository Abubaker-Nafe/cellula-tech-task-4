from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction


PROJECT_DIR = Path(__file__).resolve().parent
CHROMA_DIR = PROJECT_DIR / "chroma_db"
COLLECTION_NAME = "coding_knowledge"


def get_collection():
    """Open or create our persistent coding knowledge collection."""
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=DefaultEmbeddingFunction(),
        configuration={
            "hnsw": {
                "space": "cosine",
            }
        },
    )