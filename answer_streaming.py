"""Collect complete LLM answers while optionally forwarding visible text chunks."""

from collections.abc import Callable

from llm_client import get_llm


class AnswerStreamError(RuntimeError):
    """The model did not finish a usable answer."""


def visible_text(content) -> str:
    """Read visible text, excluding reasoning and tool-call blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                value = block.get("text")
                if isinstance(value, str):
                    parts.append(value)
        return "".join(parts)
    return ""


def collect_answer(
    messages: list[dict],
    *,
    max_tokens: int,
    on_token: Callable[[str], None] | None = None,
) -> str:
    """Use one model request and return only a completed, non-empty answer."""
    model = get_llm(max_tokens=max_tokens)
    finish_reason = None
    if on_token is None:
        response = model.invoke(messages)
        answer = visible_text(response.content)
        metadata = getattr(response, "response_metadata", None) or {}
        finish_reason = metadata.get("finish_reason")
    else:
        parts = []
        stream = model.stream(messages)
        try:
            for chunk in stream:
                metadata = getattr(chunk, "response_metadata", None) or {}
                if metadata.get("finish_reason"):
                    finish_reason = metadata["finish_reason"]
                text = visible_text(chunk.content)
                if text:
                    parts.append(text)
                    on_token(text)
        finally:
            close_stream = getattr(stream, "close", None)
            if callable(close_stream):
                close_stream()
        answer = "".join(parts)
    if finish_reason in {"length", "max_tokens"}:
        raise AnswerStreamError("The model reached its output limit. Request a shorter answer and retry.")
    if finish_reason == "content_filter":
        raise AnswerStreamError("The provider stopped the answer before completion.")
    answer = answer.strip()
    if not answer:
        raise AnswerStreamError("The model returned no visible answer. Retry or check the configured model.")
    return answer


def stream_explanation(
    user_query: str,
    *,
    history: list[dict] | None = None,
    on_token: Callable[[str], None],
) -> str:
    """Stream a direct explanation using the existing explanation prompt."""
    from explain_code import EXPLANATION_PROMPT

    if not user_query.strip():
        raise ValueError("The query cannot be empty.")
    memory_instruction = """
    Use previous conversation messages to resolve follow-ups and identify
    the code or execution output the latest question refers to.
    Explain the latest requested code and respect relevant user preferences.
    Do not invent missing code or execution results.
    """
    messages = [{"role": "system", "content": EXPLANATION_PROMPT + memory_instruction}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": user_query})
    return collect_answer(messages, max_tokens=2048, on_token=on_token)
