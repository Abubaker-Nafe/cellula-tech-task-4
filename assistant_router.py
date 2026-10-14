import argparse
import json
import math
import re
import subprocess
from collections.abc import Callable
from pathlib import Path

from conversation_memory import ConversationMemory
from answer_streaming import stream_explanation
from execute_code import execute_code
from explain_code import explain_code
from generate_code import generate_code
from intent_classifier import classify_intent
from learn_from_feedback import learn_from_feedback


TRANSIENT_ERROR_STATUSES = {"CONTEXT_ERROR", "FEEDBACK_ERROR", "EVALUATION_ERROR"}


def find_pending_request(turns: list[dict]) -> str | None:
    """Find the most recent unanswered generation request."""
    for turn in reversed(turns):
        result = turn["result"]
        status = result.get("status")
        if status in {"GENERATED", "LEARNED"}:
            return None
        if status == "NEEDS_REFERENCE":
            return result.get("resolved_request") or turn["user_query"]
    return None


def extract_python_solution(message: str) -> str | None:
    """Accept one complete Python code block as a supplied solution."""
    blocks = re.findall(
        r"```(?:python|py)?[ \t]*\r?\n(.*?)```",
        message,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if len(blocks) == 1 and blocks[0].strip():
        return blocks[0].strip()
    return None


def extract_python_blocks(answer: str) -> list[str]:
    """Extract complete blocks explicitly marked as Python."""
    return re.findall(
        r"^[ \t]*```(?:python|py)[ \t]*\r?\n(.*?)^[ \t]*```[ \t]*(?:\r?\n|$)",
        answer,
        flags=re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )


def execute_generated_answer(
    result: dict,
    *,
    timeout_seconds: float = 5.0,
    code_block_index: int | None = None,
) -> dict:
    """Pass a selected generated Python block to the separate runner."""
    skipped = {
        "status": "SKIPPED",
        "stdout": "",
        "stderr": "",
        "exit_code": None,
        "execution_seconds": 0.0,
        "message": "",
    }
    if result.get("status") != "GENERATED":
        skipped["message"] = "Execution is available only for GENERATED answers."
        return skipped

    blocks = extract_python_blocks(result["answer"])
    if not blocks:
        skipped["message"] = "No complete code block marked python or py was found."
        return skipped
    if code_block_index is None:
        if len(blocks) != 1:
            skipped["message"] = (
                f"Found {len(blocks)} Python blocks. Select one with --code-block N "
                "or request one complete program in one Python block."
            )
            return skipped
        code_block_index = 1
    if not 1 <= code_block_index <= len(blocks):
        skipped["message"] = f"Choose a Python block from 1 to {len(blocks)}."
        return skipped

    code = blocks[code_block_index - 1].strip()
    if not code:
        skipped["message"] = "The selected Python block is empty."
        return skipped
    try:
        execution = execute_code(code, timeout_seconds=timeout_seconds)
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        execution = {
            **skipped,
            "status": "RUNNER_ERROR",
            "message": f"{type(error).__name__}: {error}",
        }
    execution["code_block_index"] = code_block_index
    execution["code"] = code
    return execution


def answer_with_execution(result: dict) -> str:
    """Include recorded execution output in later conversation context."""
    answer = result["answer"]
    execution = result.get("execution")
    if execution and execution.get("status") != "SKIPPED":
        report = {key: value for key, value in execution.items() if key != "code"}
        answer += "\n\nCode runner result:\n" + json.dumps(report, ensure_ascii=False)
    return answer


def route_query(
    user_query: str,
    conversation_id: str = "default",
    supplied_solution: str | None = None,
    solution_language: str = "python",
    *,
    execute_generated: bool = False,
    execution_timeout: float = 5.0,
    code_block_index: int | None = None,
    on_token: Callable[[str], None] | None = None,
) -> dict:
    """Load history, route the request, and save the completed turn."""
    user_query = user_query.strip()

    if not user_query:
        raise ValueError("The query cannot be empty.")

    memory = ConversationMemory(conversation_id)
    # Recover useful history even if older runs saved formatting errors.
    recent_turns = memory.get_turns(limit_turns=40)
    usable_turns = [
        turn for turn in recent_turns
        if turn["result"].get("status") not in TRANSIENT_ERROR_STATUSES
    ][-4:]
    history = []
    for turn in usable_turns:
        history.extend([
            {"role": "user", "content": turn["user_query"]},
            {"role": "assistant", "content": answer_with_execution(turn["result"])},
        ])

    intent = classify_intent(user_query, history=history)

    pending_request = find_pending_request(recent_turns)
    detected_solution = supplied_solution
    if detected_solution is None and pending_request and intent == "GENERATE":
        detected_solution = extract_python_solution(user_query)

    saved_query = user_query
    if supplied_solution is not None:
        saved_query += (
            f"\n\nSupplied implementation:\n```{solution_language}\n"
            f"{supplied_solution}\n```"
        )

    if intent == "EXPLAIN":
        if on_token is None:
            answer = explain_code(saved_query, history=history)
        else:
            answer = stream_explanation(saved_query, history=history, on_token=on_token)
        result = {
            "intent": intent,
            "status": "EXPLAINED",
            "answer": answer,
            "source_ids": [],
            "evaluation_reason": None,
        }

    elif intent == "GENERATE" and detected_solution is not None:
        if not pending_request:
            result = {
                "intent": intent,
                "status": "FEEDBACK_ERROR",
                "answer": (
                    "There is no unanswered request to attach this solution to. "
                    "Ask the coding question first."
                ),
                "source_ids": [],
                "evaluation_reason": None,
            }
        else:
            try:
                stored = learn_from_feedback(
                    pending_request,
                    detected_solution,
                    language=solution_language,
                )
            except ValueError as error:
                result = {
                    "intent": intent,
                    "status": "FEEDBACK_ERROR",
                    "answer": f"The supplied reference was not saved: {error}",
                    "source_ids": [],
                    "evaluation_reason": None,
                }
            else:
                result = {
                    "intent": intent,
                    "status": "LEARNED",
                    "answer": (
                        f"I saved your supplied implementation for: {pending_request} "
                        "It is available as a reference for future retrieval. "
                        "Its correctness has not been verified by execution."
                    ),
                    "source_ids": [stored["id"]],
                    "evaluation_reason": None,
                    "resolved_request": pending_request,
                    "learned_request": pending_request,
                    "reference_validation": stored["validation"],
                }

    elif intent == "GENERATE":
        result = {
            "intent": intent,
            **generate_code(user_query, history=history, on_token=on_token),
        }

    else:
        raise ValueError(f"Unsupported intent: {intent}")

    result["history_messages_loaded"] = len(history)

    if execute_generated:
        result["execution"] = execute_generated_answer(
            result,
            timeout_seconds=execution_timeout,
            code_block_index=code_block_index,
        )

    # Formatting errors must not displace useful conversation history.
    if result["status"] not in TRANSIENT_ERROR_STATUSES:
        memory.save_turn(saved_query, result)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run the coding assistant with conversation storage."
    )
    parser.add_argument("query", help="The programming request.")
    parser.add_argument(
        "--conversation",
        default="default",
        help="Conversation ID used to load and save history.",
    )
    parser.add_argument(
        "--debug-context",
        action="store_true",
        help="Show resolver diagnostics without printing history or credentials.",
    )
    parser.add_argument(
        "--solution-file",
        type=Path,
        help="Read a supplied solution for the pending generation request.",
    )
    parser.add_argument("--language", default="python")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Run generated Python with your local account permissions.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Maximum execution time in seconds (default: 5).",
    )
    parser.add_argument(
        "--code-block",
        type=int,
        help="Select a Python block to execute, numbered from 1.",
    )
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and greater than zero.")
    if args.code_block is not None:
        if not args.execute:
            parser.error("--code-block requires --execute.")
        if args.code_block < 1:
            parser.error("--code-block must be at least 1.")

    supplied_solution = None
    if args.solution_file is not None:
        if not args.solution_file.is_file():
            parser.error(f"Solution file not found: {args.solution_file}")
        supplied_solution = args.solution_file.read_text(encoding="utf-8-sig")

    result = route_query(
        args.query,
        conversation_id=args.conversation,
        supplied_solution=supplied_solution,
        solution_language=args.language,
        execute_generated=args.execute,
        execution_timeout=args.timeout,
        code_block_index=args.code_block,
    )

    print(f"Conversation: {args.conversation}")
    print(
        "History messages loaded:",
        result["history_messages_loaded"],
    )
    print(f"Intent: {result['intent']}")
    print(f"Status: {result['status']}")
    print(f"Sources: {result['source_ids']}")

    if result["evaluation_reason"]:
        print(f"Evaluation: {result['evaluation_reason']}")

    if result.get("resolved_request"):
        print(f"Resolved request: {result['resolved_request']}")

    if result.get("reference_validation"):
        print(f"Reference validation: {result['reference_validation']}")

    if args.debug_context and result.get("context_diagnostics"):
        print("\nContext diagnostics:")
        print(json.dumps(result["context_diagnostics"], indent=2))

    if result.get("evaluation_diagnostics"):
        print("\nEvaluation diagnostics:")
        print(json.dumps(result["evaluation_diagnostics"], indent=2))

    print("\nAnswer:\n")
    print(result["answer"])

    if result.get("execution"):
        execution = result["execution"]
        print("\nExecution result:")
        print(f"Execution status: {execution['status']}")
        print(f"Exit code: {execution['exit_code']}")
        if execution.get("code_block_index"):
            print(f"Python block: {execution['code_block_index']}")
        if execution["message"]:
            print(f"Message: {execution['message']}")
        print("Standard output:")
        print(execution["stdout"].rstrip() or "(empty)")
        print("Standard error:")
        print(execution["stderr"].rstrip() or "(empty)")


if __name__ == "__main__":
    main()
