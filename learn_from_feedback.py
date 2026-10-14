"""
This file:
- Combines the original request with its supplied implementation.
- Checks Python syntax without executing the code.
- Creates metadata recording the source, language, and function name.
- Generates a stable ID from the request and solution.
- Uses Chroma’s upsert() to store the reference and create embeddings. Repeating identical input updates the same record.
"""

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from vector_store import get_collection

DEMO_REQUEST = (
    "Write a Python function that turns a title into lowercase words "
    "joined by hyphens, ignoring extra whitespace."
)

DEMO_SOLUTION = '''def title_to_slug(title: str) -> str:
    """Lowercase a title and join its words with hyphens."""
    if not isinstance(title, str):
        raise TypeError("title must be a string")
    return "-".join(title.lower().split())
'''


def learn_from_feedback(
    original_request: str,
    solution: str,
    language: str = "python",
) -> dict:
    """Store a supplied solution as a future retrieval reference."""
    original_request = original_request.strip()
    solution = solution.replace("\r\n", "\n").lstrip("\ufeff").strip()
    language = language.strip().lower()

    if not original_request or not solution or not language:
        raise ValueError("The request, solution, and language cannot be empty.")

    entry_point = "user_reference"
    validation = "not_checked"

    if language == "python":
        try:
            tree = ast.parse(solution)
        except SyntaxError as error:
            raise ValueError(
                f"Python syntax error at line {error.lineno}: {error.msg}"
            ) from error

        function_names = [
            node.name for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        if function_names:
            entry_point = function_names[0]
        validation = "syntax_only"

    identity = json.dumps(
        {
            "request": original_request,
            "solution": solution,
            "language": language,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    document_id = "feedback_" + hashlib.sha256(
        identity.encode("utf-8")
    ).hexdigest()

    document = (
        f"Programming reference supplied through user feedback\n"
        f"Request: {original_request}\n"
        f"Function: {entry_point}\n\n"
        f"Supplied implementation:\n```{language}\n{solution}\n```"
    )
    metadata = {
        "source": "user_feedback",
        "language": language,
        "entry_point": entry_point,
        "original_request": original_request,
        "validation": validation,
        "added_at": datetime.now(timezone.utc).isoformat(),
    }

    collection = get_collection()
    collection.upsert(
        ids=[document_id],
        documents=[document],
        metadatas=[metadata],
    )

    return {
        "id": document_id,
        "entry_point": entry_point,
        "validation": validation,
        "document": document,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Save a supplied solution in the coding knowledge base."
    )
    parser.add_argument("request", nargs="?", help="The original coding request.")
    parser.add_argument("--solution-file", type=Path)
    parser.add_argument("--language", default="python")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    if args.demo:
        request = DEMO_REQUEST
        solution = DEMO_SOLUTION
        language = "python"
    else:
        if not args.request or not args.solution_file:
            parser.error("Provide a request and --solution-file, or use --demo.")
        if not args.solution_file.is_file():
            parser.error(f"Solution file not found: {args.solution_file}")
        request = args.request
        solution = args.solution_file.read_text(encoding="utf-8-sig")
        language = args.language

    collection = get_collection()
    before = collection.count()
    result = learn_from_feedback(request, solution, language)
    after = collection.count()

    # Retrieve the saved record to check the text and metadata were stored.
    saved = collection.get(
        ids=[result["id"]],
        include=["documents", "metadatas"],
    )
    if saved["ids"] != [result["id"]] or saved["documents"] != [result["document"]]:
        raise RuntimeError("The saved reference could not be verified.")

    print(f"Saved ID: {result['id']}")
    print(f"Function: {result['entry_point']}")
    print(f"Source: {saved['metadatas'][0]['source']}")
    print(f"Validation: {result['validation']}")
    print(f"Documents before: {before}")
    print(f"Documents after: {after}")
    print("Saved reference retrieved successfully.")


if __name__ == "__main__":
    main()
