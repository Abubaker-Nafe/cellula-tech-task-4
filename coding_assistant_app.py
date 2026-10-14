"""Local Streamlit interface for the Task-4 coding assistant.

Install: python -m pip install streamlit
Run: python -m streamlit run coding_assistant_app.py --server.address 127.0.0.1 --server.fileWatcherType none

The local runner uses your operating-system permissions. Keep this prototype
local; public hosting requires a sandboxed runner and access controls.
Final generation and explanation answers stream directly from the model.
"""

import os
import re
from pathlib import Path
from uuid import uuid4

import streamlit as st
from dotenv import load_dotenv

from assistant_router import (
    TRANSIENT_ERROR_STATUSES,
    execute_generated_answer,
    extract_python_blocks,
    find_pending_request,
    route_query,
)
from conversation_memory import ConversationMemory
from answer_streaming import AnswerStreamError


def new_conversation_id() -> str:
    return "ui_" + uuid4().hex[:12]


def load_turns(conversation_id: str) -> list[dict]:
    """Read the latest saved turns in chronological order."""
    return [
        turn for turn in ConversationMemory(conversation_id).get_turns(limit_turns=100)
        if turn["result"].get("status") not in TRANSIENT_ERROR_STATUSES
    ]


def friendly_error(error: Exception) -> str:
    """Describe API failures without displaying credentials or request data."""
    name = type(error).__name__
    if isinstance(error, AnswerStreamError):
        return str(error)
    if "TooManyRequests" in name or "RateLimit" in name:
        return "The model provider's request limit was reached. Try again when quota is available."
    if "NotFound" in name:
        return "No endpoint was available for the configured model. Check OPENROUTER_MODEL."
    if "Unauthorized" in name or "Authentication" in name:
        return "The model provider rejected authentication. Check your API key configuration."
    return f"The request could not complete ({name}). Check your terminal and project configuration."


def submit_query(
    query: str,
    supplied_solution: str | None = None,
    *,
    stream_answers: bool = True,
) -> dict:
    """Submit exactly once; transient failures stay only in this browser session."""
    st.session_state.pop("latest_failure", None)
    parts = []
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        progress = st.empty()
        answer_area = st.empty()
        progress.caption("Processing your request...")

        def display_chunk(text: str) -> None:
            if not parts:
                progress.caption("Receiving answer...")
            parts.append(text)
            answer_area.markdown("".join(parts))

        try:
            result = route_query(
                query,
                conversation_id=st.session_state.conversation_id,
                supplied_solution=supplied_solution,
                on_token=display_chunk if stream_answers else None,
            )
        except Exception as error:
            result = {
                "intent": "UNKNOWN",
                "status": "REQUEST_ERROR",
                "answer": friendly_error(error),
                "source_ids": [],
                "evaluation_reason": None,
            }
        answer_area.markdown(result["answer"])
        progress.empty()
    if result["status"] in TRANSIENT_ERROR_STATUSES or result["status"] == "REQUEST_ERROR":
        st.session_state.latest_failure = {
            "user_query": query,
            "result": result,
            "partial_answer": "".join(parts) if result["status"] == "REQUEST_ERROR" else "",
        }
    return result


def save_execution(conversation_id: str, source_result: dict, execution: dict) -> None:
    """Record the Run button action so follow-ups receive the code and output."""
    number = execution["code_block_index"]
    code = execution["code"]
    ConversationMemory(conversation_id).save_turn(
        f"Run Python code block {number} from the generated answer.",
        {
            "intent": "GENERATE",
            "status": "EXECUTED",
            "answer": f"Execution of Python block {number}:\n\n```python\n{code}\n```",
            "source_ids": source_result.get("source_ids", []),
            "evaluation_reason": None,
            "execution": execution,
        },
    )


def render_execution(execution: dict) -> None:
    status = execution["status"]
    if status == "SUCCESS":
        st.success("Execution completed successfully.")
    elif status == "SKIPPED":
        st.warning(execution.get("message") or "Execution was skipped.")
    else:
        st.error(f"Execution status: {status}")
    if execution.get("message") and status != "SKIPPED":
        st.caption(execution["message"])
    st.caption(f"Exit code: {execution.get('exit_code')}")
    st.markdown("**Standard output**")
    st.code(execution.get("stdout", "").rstrip() or "(empty)", language="text")
    st.markdown("**Standard error**")
    st.code(execution.get("stderr", "").rstrip() or "(empty)", language="text")


def render_turn(turn: dict, index: int, settings: dict) -> None:
    result = turn["result"]
    with st.chat_message("user"):
        st.markdown(turn["user_query"])
    with st.chat_message("assistant"):
        st.caption(f"{result.get('intent', 'UNKNOWN')} · {result['status']}")
        st.markdown(result["answer"])
        if turn.get("partial_answer"):
            with st.expander("Incomplete response — not saved"):
                st.markdown(turn["partial_answer"])
        sources = result.get("source_ids") or []
        if sources:
            with st.expander("Sources and relevance"):
                st.code("\n".join(sources), language="text")
                if result.get("evaluation_reason"):
                    st.write(result["evaluation_reason"])
        if settings["show_details"]:
            for field, label in (
                ("context_diagnostics", "Context diagnostics"),
                ("evaluation_diagnostics", "Evaluation diagnostics"),
            ):
                if result.get(field):
                    with st.expander(label):
                        st.json(result[field])
        if result.get("execution"):
            render_execution(result["execution"])
        if result["status"] != "GENERATED":
            return
        blocks = extract_python_blocks(result["answer"])
        if not blocks:
            st.caption("No complete Python block is available to run.")
            return
        prefix = f"run_{st.session_state.conversation_id}_{index}"
        selected = 1
        if len(blocks) > 1:
            selected = st.selectbox(
                "Python block to run", range(1, len(blocks) + 1), key=prefix + "_block"
            )
        with st.expander("Review selected Python code"):
            st.code(blocks[selected - 1].strip(), language="python")
        if st.button(
            "Run code",
            key=prefix + "_button",
            disabled=not settings["allow_execution"],
        ):
            with st.spinner("Executing selected Python code..."):
                execution = execute_generated_answer(
                    result,
                    timeout_seconds=settings["timeout"],
                    code_block_index=selected,
                )
            if execution["status"] == "SKIPPED":
                render_execution(execution)
            else:
                try:
                    save_execution(st.session_state.conversation_id, result, execution)
                except Exception as error:
                    render_execution(execution)
                    st.error(f"Execution completed, but saving the result failed ({type(error).__name__}).")
                else:
                    st.session_state.pop("latest_failure", None)
                    st.rerun()


def render_sidebar() -> dict:
    with st.sidebar:
        st.header("Conversations")
        st.caption("Active conversation")
        st.code(st.session_state.conversation_id, language="text")
        target = st.text_input("Conversation ID to load", key="conversation_to_load")
        if st.button("Load conversation", use_container_width=True):
            target = target.strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", target):
                st.error("Use 1–80 letters, digits, underscores, or hyphens.")
            else:
                st.session_state.conversation_id = target
                st.session_state.pop("latest_failure", None)
                st.rerun()
        if st.button("New conversation", use_container_width=True):
            st.session_state.conversation_id = new_conversation_id()
            st.session_state.pop("latest_failure", None)
            st.rerun()
        st.divider()
        st.header("Execution")
        allow_execution = st.checkbox("Enable local code execution", value=False)
        st.caption("Review code before running it. It uses your local account's permissions.")
        timeout = st.number_input(
            "Timeout in seconds", min_value=0.2, max_value=30.0, value=5.0, step=0.2
        )
        st.divider()
        stream_responses = st.checkbox("Stream responses", value=True)
        show_details = st.checkbox("Show diagnostic details", value=False)
        model = os.getenv("OPENROUTER_MODEL", "Not configured")
        st.caption(f"Configured model: {model}")
        if model != "Not configured" and not model.endswith(":free") and model != "openrouter/free":
            st.caption("This model may charge for API requests.")
    return {
        "allow_execution": allow_execution,
        "timeout": timeout,
        "show_details": show_details,
        "stream_responses": stream_responses,
    }


def render_feedback(turns: list[dict], *, stream_answers: bool = True) -> None:
    if not find_pending_request(turns[-40:]):
        return
    with st.expander("Provide a reference for the unanswered request", expanded=True):
        st.write("Upload a Python implementation to save it as a reference for future requests.")
        upload = st.file_uploader(
            "Python solution", type=["py", "txt"], key="reference_" + st.session_state.conversation_id
        )
        if st.button("Save reference", disabled=upload is None):
            try:
                solution = upload.getvalue().decode("utf-8-sig")
            except UnicodeDecodeError:
                st.error("Use a UTF-8 Python or text file.")
            else:
                if not solution.strip():
                    st.error("The uploaded file is empty.")
                else:
                    submit_query(
                        "Use this corrected implementation for the previous programming request.",
                        supplied_solution=solution,
                        stream_answers=stream_answers,
                    )
                    st.rerun()


def main() -> None:
    st.set_page_config(page_title="Cellula Coding Assistant", page_icon="💻", layout="wide")
    load_dotenv(Path(__file__).resolve().parent / ".env")
    if "conversation_id" not in st.session_state:
        st.session_state.conversation_id = new_conversation_id()
    settings = render_sidebar()
    st.title("Cellula Coding Assistant")
    st.caption("Generate code, understand it, and reuse knowledge from your references.")
    try:
        turns = load_turns(st.session_state.conversation_id)
    except Exception as error:
        st.error(f"Conversation history could not be loaded ({type(error).__name__}).")
        st.stop()
    if not turns and not st.session_state.get("latest_failure"):
        st.info("Ask for a Python program, paste code to explain, or load an existing conversation.")
    for index, turn in enumerate(turns):
        render_turn(turn, index, settings)
    failure = st.session_state.get("latest_failure")
    if failure:
        render_turn(failure, len(turns), settings)
    render_feedback(turns, stream_answers=settings["stream_responses"])
    prompt = st.chat_input("Ask a coding question or paste code to explain...")
    if prompt and prompt.strip():
        submit_query(prompt.strip(), stream_answers=settings["stream_responses"])
        st.rerun()


if __name__ == "__main__":
    main()
