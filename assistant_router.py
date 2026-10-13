import argparse
import json

from conversation_memory import ConversationMemory
from explain_code import explain_code
from generate_code import generate_code
from intent_classifier import classify_intent


def route_query(
    user_query: str,
    conversation_id: str = "default",
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
        if turn["result"].get("status") != "CONTEXT_ERROR"
    ][-4:]
    history = []
    for turn in usable_turns:
        history.extend([
            {"role": "user", "content": turn["user_query"]},
            {"role": "assistant", "content": turn["result"]["answer"]},
        ])

    intent = classify_intent(user_query, history=history)

    if intent == "EXPLAIN":
        result = {
            "intent": intent,
            "status": "EXPLAINED",
            "answer": explain_code(user_query, history=history),
            "source_ids": [],
            "evaluation_reason": None,
        }

    elif intent == "GENERATE":
        result = {
            "intent": intent,
            **generate_code(user_query, history=history),
        }

    else:
        raise ValueError(f"Unsupported intent: {intent}")

    result["history_messages_loaded"] = len(history)

    # Formatting errors must not displace useful conversation history.
    if result["status"] != "CONTEXT_ERROR":
        memory.save_turn(user_query, result)

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
    args = parser.parse_args()

    result = route_query(
        args.query,
        conversation_id=args.conversation,
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

    if args.debug_context and result.get("context_diagnostics"):
        print("\nContext diagnostics:")
        print(json.dumps(result["context_diagnostics"], indent=2))

    print("\nAnswer:\n")
    print(result["answer"])


if __name__ == "__main__":
    main()