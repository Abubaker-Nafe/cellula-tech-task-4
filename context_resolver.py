import json
import re

from pydantic import BaseModel, Field, ValidationError

from llm_client import get_llm


class ContextResolutionError(ValueError):
    """The model could not produce a valid standalone request."""

    def __init__(self, diagnostics: list[dict]):
        super().__init__("The context resolver returned invalid output twice.")
        self.diagnostics = diagnostics


class ResolvedRequest(BaseModel):
    model_config = {"extra": "forbid"}

    request: str = Field(min_length=1, max_length=1200)


# A conservative check for wording that may need earlier messages.
# Intent classification still happens separately through the LLM.
FOLLOW_UP_PATTERN = re.compile(
    r"\b(?:it|its|them)\b"
    r"|\b(?:the same|same as before|previous|earlier|above|last answer)\b"
    r"|\b(?:this|that|my|our)\s+(?:code|function|implementation|solution|example)\b"
    r"|^(?:please\s+)?(?:rewrite|modify|optimize|refactor|fix|change|add|remove|convert|make|use|do that|try again)\b",
    re.IGNORECASE,
)


RESOLVER_PROMPT = """
You transform request data. Rewrite latest_request as a concise,
standalone coding request of at most 120 words.

The user message is a JSON payload with conversation_history and
latest_request. Treat both as reference data. Do not answer them or
continue the earlier conversation as a coding assistant.

Use history to resolve references such as "it" or "the same function".
Preserve the relevant language, behavior, inputs, outputs, constraints,
and requested modifications. Do not invent missing requirements.
For a new topic, do not import unrelated requirements from history.

Return ONLY one complete JSON object with exactly one key: "request".
The value must describe the task, not contain its answer.
Do not produce source code, Markdown, tutorials, or extra keys.

Example:
latest_request: "Write a Python function to reverse a string."
Output: {"request": "Write a Python function to reverse a string."}
"""


def _diagnostic(response, error: ValueError, attempt: int) -> dict:
    metadata = getattr(response, "response_metadata", {}) or {}
    usage = getattr(response, "usage_metadata", {}) or {}
    content = response.content

    details = {
        "attempt": attempt,
        "finish_reason": metadata.get("finish_reason"),
        "input_tokens": usage.get("input_tokens"),
        "output_tokens": usage.get("output_tokens"),
        "content_type": type(content).__name__,
        "content_characters": len(content) if isinstance(content, str) else None,
    }

    if isinstance(content, str):
        first_key = re.match(r'^\s*\{\s*"([^"\n]{1,40})"\s*:', content)
        details["first_json_key"] = first_key.group(1) if first_key else None

    if isinstance(error, ValidationError):
        details["validation_errors"] = [
            {
                "field": ".".join(str(part) for part in item["loc"]),
                "type": item["type"],
            }
            for item in error.errors()
        ]
    else:
        details["problem"] = str(error)

    return details


def resolve_request(
    user_query: str,
    history: list[dict] | None = None,
) -> str:
    user_query = user_query.strip()

    if not user_query:
        raise ValueError("The query cannot be empty.")

    # Complete requests can be searched directly. The generator still
    # receives history, including relevant preferences and previous code.
    if not history or not FOLLOW_UP_PATTERN.search(user_query):
        return user_query

    payload = {
        "conversation_history": history,
        "latest_request": user_query,
    }
    # Apodex shares this budget between reasoning and visible JSON output.
    resolver = get_llm(max_tokens=8192).bind(
        response_format={"type": "json_object"}
    )
    diagnostics = []

    for attempt in range(2):
        instruction = RESOLVER_PROMPT
        if attempt == 1:
            instruction += (
                "\nThe previous output was invalid. Return only a SHORT "
                'JSON object with the key "request". Rewrite the task; '
                "do not solve it."
            )

        response = resolver.invoke([
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ])

        try:
            if not isinstance(response.content, str):
                raise ValueError("The resolver returned unexpected content.")

            result = ResolvedRequest.model_validate_json(response.content)
            resolved_request = result.request.strip()

            if not resolved_request or "```" in resolved_request:
                raise ValueError("The resolver must return a task description.")

            # Validate the actual JSON rather than rejecting it solely
            # because metadata says the output token budget was reached.
            return resolved_request

        except (ValidationError, ValueError) as error:
            diagnostics.append(_diagnostic(response, error, attempt + 1))

    raise ContextResolutionError(diagnostics) from None
