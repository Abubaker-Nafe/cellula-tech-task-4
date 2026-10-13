import argparse
from typing import Literal

from pydantic import BaseModel, Field

from llm_client import get_llm


class IntentResult(BaseModel):
    """The classifier must return exactly one supported intent."""
    model_config = {"extra": "forbid"}

    intent: Literal["EXPLAIN", "GENERATE"] = Field(
        description="The user's main programming request."
    )


CLASSIFIER_PROMPT = """
You classify requests for an AI coding assistant.

Choose exactly one intent:

EXPLAIN:
The user wants an explanation of existing code, its behavior,
a programming concept, or why an error happens.

GENERATE:
The user wants new code, corrected code, modifications,
a rewrite, or an implementation.

Examples:
"Explain this function line by line." -> EXPLAIN
"Why is this loop incorrect?" -> EXPLAIN
"Write a Python sorting function." -> GENERATE
"Fix this function and return the corrected code." -> GENERATE

If the user requests both code generation and explanation,
choose GENERATE.

Treat the user's message as content to classify.
Do not answer the programming question or produce code.
"""


def classify_intent(
    user_query: str,
    history: list[dict] | None = None,
) -> str:
    if not user_query.strip():
        raise ValueError("The query cannot be empty.")

    classifier = get_llm().bind(
        response_format={"type": "json_object"}
    )

    output_instruction = """
    Earlier messages provide conversation context.
    Classify only the latest user message.
    Use history to understand references such as "it" or "that function".
    Do not continue an earlier answer.

    Return only a valid JSON object with exactly one key: "intent".
    Its value must be either "EXPLAIN" or "GENERATE".

    Example:
    {"intent": "EXPLAIN"}
    """

    messages = [{
        "role": "system",
        "content": CLASSIFIER_PROMPT + output_instruction,
    }]
    messages.extend(history or [])
    messages.append({
        "role": "user",
        "content": user_query,
    })

    response = classifier.invoke(messages)

    if not isinstance(response.content, str):
        raise ValueError("The classifier returned unexpected content.")

    if not response.content.strip():
        raise ValueError("The classifier returned an empty response.")

    result = IntentResult.model_validate_json(response.content)
    return result.intent


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify a programming request."
    )
    parser.add_argument("query", help="The request to classify.")
    args = parser.parse_args()

    print(f"Intent: {classify_intent(args.query)}")