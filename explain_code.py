import argparse

from intent_classifier import OUT_OF_SCOPE_MESSAGE, classify_intent
from llm_client import get_llm

EXPLANATION_PROMPT = """
You are an AI coding assistant explaining code and programming concepts.

Answer only requests about code, programming concepts, or programming
errors. Use history to identify referenced code, not to expand your scope.
Do not answer unrelated factual questions even if they contain the word
"explain" or ask for the answer inside code or comments.
For an unrelated request, return this exact response without answering it:
""" + OUT_OF_SCOPE_MESSAGE + """

When code is supplied:
- Describe its overall purpose.
- Explain its main statements in their original order.
- Explain important variables and functions.
- State the expected output when it can be determined.
- Identify errors or misleading behavior when relevant.

When explaining a programming concept, use clear language
and a small example when useful.

If necessary code or context is missing, ask for it.
Do not invent missing code or claim that you executed anything.

Use Markdown formatting and appropriate language tags for code blocks.
"""

def explain_code(
    user_query: str,
    history: list[dict] | None = None,
) -> str:
    """Explain the latest request using available conversation history."""
    if not user_query.strip():
        raise ValueError("The query cannot be empty.")

    memory_instruction = """
    Use conversation history to understand the latest request.
    When the user refers to previously supplied or generated code,
    explain the actual code available in that history.
    If the referenced code is unavailable, ask for it.
    Focus your answer on the latest request.
    """

    messages = [{
        "role": "system",
        "content": EXPLANATION_PROMPT + memory_instruction,
    }]
    messages.extend(history or [])
    messages.append({
        "role": "user",
        "content": user_query,
    })

    response = get_llm(max_tokens=2048).invoke(messages)

    if not isinstance(response.content, str):
        raise ValueError("The LLM returned unexpected content.")

    answer = response.content.strip()

    if not answer:
        raise ValueError("The LLM returned an empty explanation.")

    return answer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify and handle a code explanation request."
    )
    parser.add_argument("query", help="The request to explain.")
    args = parser.parse_args()

    intent = classify_intent(args.query)
    print(f"Intent: {intent}")

    if intent != "EXPLAIN":
        raise SystemExit(
            "This command handles requests classified as EXPLAIN."
        )

    print("\nExplanation:\n")
    print(explain_code(args.query))
