import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter


def main():
    # Load configuration from this project's .env file.
    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir / ".env")

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    model_name = os.getenv("OPENROUTER_MODEL", "").strip()

    if not api_key or api_key == "your_actual_key_here":
        raise SystemExit("Add your OpenRouter API key to .env.")

    if not model_name:
        raise SystemExit("Add OPENROUTER_MODEL to .env.")

    llm = ChatOpenRouter(
        api_key=api_key,
        model=model_name,
        temperature=0,
        max_tokens=1024,
        timeout=60_000,
        max_retries=0,
    )

    print(f"Connecting to: {model_name}")

    response = llm.invoke([
        ("system", "Follow the user's instruction exactly."),
        ("human", "Reply with exactly: Connection successful"),
    ])

    if not response.content:
        raise SystemExit("The model returned an empty response.")

    print("Model response:")
    print(response.content)


if __name__ == "__main__":
    main()