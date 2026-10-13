import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


def get_llm(max_tokens: int = 1024) -> ChatOpenRouter:
    """Create the LLM connection used by our project."""
    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model_name = os.getenv("OPENROUTER_MODEL", "").strip()

    if not api_key or api_key == "your_actual_key_here":
        raise ValueError("Add your OpenRouter API key to .env.")

    if not model_name:
        raise ValueError("Add OPENROUTER_MODEL to .env.")

    return ChatOpenRouter(
        api_key=api_key,
        model=model_name,
        temperature=0,
        max_tokens=max_tokens,
        timeout=60_000,
        max_retries=0,
    )