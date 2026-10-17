import argparse
import ast
import math
import os
from importlib.util import find_spec
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from dotenv import load_dotenv


PROJECT_DIR = Path(__file__).resolve().parent
MAX_SOURCE_BYTES = 100_000


def get_runner_configuration() -> dict:
    """Describe the selected runner without returning any credentials."""
    load_dotenv(PROJECT_DIR / ".env")
    backend = os.getenv("CODE_RUNNER_BACKEND", "e2b").strip().lower()
    if backend == "e2b":
        api_key = os.getenv("E2B_API_KEY", "").strip()
        if not api_key or api_key in {"your_e2b_api_key_here", "PASTE_YOUR_E2B_KEY_HERE"}:
            message = "Set E2B_API_KEY in Streamlit secrets or your local .env to enable the external runner."
        elif find_spec("e2b_code_interpreter") is None:
            message = "Install the updated requirements.txt, which includes e2b-code-interpreter."
        else:
            message = "Python runs in a fresh external E2B sandbox."
            return {"backend": backend, "available": True, "message": message}
    elif backend == "local":
        if os.getenv("ALLOW_LOCAL_CODE_EXECUTION", "false").strip().lower() == "true":
            return {"backend": backend, "available": True, "message": "Python runs using this computer's operating-system permissions."}
        message = "Local execution requires ALLOW_LOCAL_CODE_EXECUTION=true. Use the E2B backend for deployment."
    else:
        message = "CODE_RUNNER_BACKEND must be e2b or local."
    return {"backend": backend, "available": False, "message": message}


def _empty_result(backend: str, *, message: str = "") -> dict:
    return {
        "status": "RUNNER_ERROR", "backend": backend,
        "stdout": "", "stderr": "", "exit_code": None,
        "execution_seconds": 0.0, "message": message,
    }


class _OutputLimitExceeded(Exception):
    """Stop receiving output and destroy the sandbox after exceeding the cap."""


def _execute_e2b(code: str, *, timeout_seconds: float, max_output_bytes: int) -> dict:
    """Run a script remotely and normalize E2B's command results for the assistant."""
    from e2b import AuthenticationException, CommandExitException, RateLimitException, TimeoutException
    from e2b_code_interpreter import Sandbox

    result = _empty_result("e2b")
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    sandbox = None
    execution_started = False
    started = time.monotonic()

    def capture(stream_name: str, chunk: str) -> None:
        data = chunk.encode("utf-8")
        remaining = max_output_bytes - sum(len(buffer) for buffer in buffers.values())
        buffers[stream_name].extend(data[:remaining])
        if len(data) > remaining:
            raise _OutputLimitExceeded()

    try:
        sandbox = Sandbox.create(
            api_key=os.environ["E2B_API_KEY"].strip(),
            timeout=math.ceil(timeout_seconds) + 30,
            request_timeout=20,
            secure=True,
            allow_internet_access=False,
            network={"allow_public_traffic": False},
            envs={},
        )
        # Send source as file content; the command is fixed and never interpolates code.
        sandbox.files.write("/tmp/assistant_snippet.py", code, request_timeout=15)
        execution_started = True
        try:
            command = sandbox.commands.run(
                "python3 -I -X utf8 -u /tmp/assistant_snippet.py",
                cwd="/tmp",
                stdin=False,
                timeout=timeout_seconds,
                on_stdout=lambda text: capture("stdout", text),
                on_stderr=lambda text: capture("stderr", text),
            )
        except CommandExitException as error:
            command = error
        result.update(
            status="SUCCESS" if command.exit_code == 0 else "ERROR",
            exit_code=command.exit_code,
            message="" if command.exit_code == 0 else "The snippet exited with an error; see stderr.",
        )
    except _OutputLimitExceeded:
        result.update(status="OUTPUT_LIMIT", message=f"Stopped after exceeding {max_output_bytes} output bytes.")
    except TimeoutException:
        if execution_started:
            result.update(status="TIMEOUT", message=f"Execution exceeded the {timeout_seconds:g}-second command limit.")
        else:
            result["message"] = "The sandbox could not be prepared before the service request timed out. Try again."
    except AuthenticationException:
        result["message"] = "E2B rejected authentication. Check E2B_API_KEY in the app's secrets."
    except RateLimitException:
        result["message"] = "E2B's request or sandbox limit was reached. Check your account and retry later."
    except Exception as error:
        # Provider exceptions can contain request data. Never expose their raw payloads.
        result["message"] = f"External runner request failed ({type(error).__name__}). Check E2B service availability and account credits."
    finally:
        if sandbox is not None:
            try:
                sandbox.kill(request_timeout=5)
            except Exception:
                result["cleanup_warning"] = "Sandbox cleanup could not be confirmed. Its short lifetime will expire automatically."
        result["stdout"] = buffers["stdout"].decode("utf-8", errors="ignore")
        result["stderr"] = buffers["stderr"].decode("utf-8", errors="ignore")
        result["execution_seconds"] = round(time.monotonic() - started, 3)
    return result


def execute_code(
    code: str,
    *,
    timeout_seconds: float = 5.0,
    max_output_bytes: int = 16_384,
) -> dict:
    """Tool boundary: accept source text and return captured execution results.

    The assistant/router calls this tool; the LLM never interprets or executes
    Python. External failures do not fall back to running on the Streamlit host.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Provide non-empty Python source code.")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or timeout_seconds > 30:
        raise ValueError("timeout_seconds must be finite, positive, and at most 30 seconds.")
    if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int) or max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be a positive integer.")
    configuration = get_runner_configuration()
    backend = configuration["backend"]
    if len(code.encode("utf-8")) > MAX_SOURCE_BYTES:
        return _empty_result(backend, message=f"Source exceeds the {MAX_SOURCE_BYTES}-byte limit.")
    try:
        ast.parse(code, filename="snippet.py")
    except SyntaxError as error:
        return {
            **_empty_result(backend), "status": "SYNTAX_ERROR",
            "stderr": f"{type(error).__name__}: {error.msg} (line {error.lineno})",
            "message": "The source could not be parsed; nothing was executed.",
        }
    if not configuration["available"]:
        return _empty_result(backend, message=configuration["message"])
    if backend == "e2b":
        try:
            return _execute_e2b(code, timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes)
        except ImportError:
            return _empty_result(backend, message="The E2B SDK could not be imported. Reinstall the updated requirements.txt.")
    return {
        **_execute_local_code(code, timeout_seconds=timeout_seconds, max_output_bytes=max_output_bytes),
        "backend": "local",
    }


def _stop_process_tree(process: subprocess.Popen) -> None:
    """Stop the snippet and its subprocesses where the OS permits it."""
    if os.name == "nt":
        system_root = os.environ.get("SystemRoot", r"C:\Windows")
        taskkill = str(Path(system_root) / "System32" / "taskkill.exe")
        try:
            subprocess.run(
                [taskkill, "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=3,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=3)


def _execute_local_code(
    code: str,
    *,
    timeout_seconds: float = 5.0,
    max_output_bytes: int = 16_384,
) -> dict:
    """Execute Python source and return status, stdout, stderr, and exit code.

    The output limit covers both streams together. The snippet runs with the
    current Python interpreter, an empty temporary working directory, closed
    standard input, and an environment that excludes application secrets.
    Imports use packages available in that interpreter; no packages are added.
    """
    if not isinstance(code, str) or not code.strip():
        raise ValueError("Provide non-empty Python source code.")
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be finite and greater than zero.")
    if isinstance(max_output_bytes, bool) or not isinstance(max_output_bytes, int):
        raise ValueError("max_output_bytes must be a positive integer.")
    if max_output_bytes <= 0:
        raise ValueError("max_output_bytes must be a positive integer.")

    result = {
        "status": "ERROR",
        "stdout": "",
        "stderr": "",
        "exit_code": None,
        "execution_seconds": 0.0,
        "message": "",
    }
    try:
        ast.parse(code, filename="snippet.py")
    except SyntaxError as error:
        result.update(
            status="SYNTAX_ERROR",
            stderr=f"{type(error).__name__}: {error.msg} (line {error.lineno})",
            message="The source could not be parsed; nothing was executed.",
        )
        return result

    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="coding_assistant_run_") as folder:
        script = Path(folder) / "snippet.py"
        script.write_text(code, encoding="utf-8")
        allowed_environment = {"SYSTEMROOT", "WINDIR", "LANG", "LC_ALL", "TZ"}
        environment = {
            key: value for key, value in os.environ.items()
            if key.upper() in allowed_environment
        }
        environment.update({"TMP": folder, "TEMP": folder, "TMPDIR": folder})

        options = {"start_new_session": True} if os.name != "nt" else {
            "creationflags": subprocess.CREATE_NEW_PROCESS_GROUP,
        }
        try:
            process = subprocess.Popen(
                [sys.executable, "-I", "-X", "utf8", "-u", str(script)],
                cwd=folder,
                env=environment,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=False,
                **options,
            )
        except OSError as error:
            result.update(status="RUNNER_ERROR", message=str(error))
            return result

        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        lock = threading.Lock()
        output_exceeded = threading.Event()

        def capture(pipe, stream_name):
            try:
                while chunk := os.read(pipe.fileno(), 4096):
                    with lock:
                        remaining = max_output_bytes - sum(
                            len(buffer) for buffer in buffers.values()
                        )
                        buffers[stream_name].extend(chunk[:remaining])
                        if len(chunk) > remaining:
                            output_exceeded.set()
            finally:
                pipe.close()

        readers = [
            threading.Thread(target=capture, args=(process.stdout, "stdout"), daemon=True),
            threading.Thread(target=capture, args=(process.stderr, "stderr"), daemon=True),
        ]
        for reader in readers:
            reader.start()

        deadline = time.monotonic() + timeout_seconds
        stop_status = None
        try:
            while True:
                if output_exceeded.is_set():
                    stop_status = "OUTPUT_LIMIT"
                    break
                if process.poll() is not None and not any(
                    reader.is_alive() for reader in readers
                ):
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    stop_status = "TIMEOUT"
                    break
                output_exceeded.wait(min(0.02, remaining))
        finally:
            # Also stop lingering descendants after the main snippet exits.
            _stop_process_tree(process)
            for reader in readers:
                reader.join(timeout=1)

        with lock:
            result["stdout"] = buffers["stdout"].decode("utf-8", errors="replace")
            result["stderr"] = buffers["stderr"].decode("utf-8", errors="replace")
        result["exit_code"] = process.returncode
        result["execution_seconds"] = round(time.monotonic() - started, 3)
        result["status"] = stop_status or (
            "SUCCESS" if process.returncode == 0 else "ERROR"
        )
        if stop_status == "TIMEOUT":
            result["message"] = f"Execution exceeded the {timeout_seconds:g}-second limit."
        elif stop_status == "OUTPUT_LIMIT":
            result["message"] = f"Stopped after exceeding {max_output_bytes} output bytes."
        elif process.returncode != 0:
            result["message"] = "The snippet exited with an error; see stderr."
    return result


def _print_result(result: dict) -> None:
    print(f"Runner: {result.get('backend', 'unknown')}")
    print(f"Status: {result['status']}")
    print(f"Exit code: {result['exit_code']}")
    if result["message"]:
        print(f"Message: {result['message']}")
    if result.get("cleanup_warning"):
        print(result["cleanup_warning"])
    print("Standard output:")
    print(result["stdout"].rstrip() or "(empty)")
    print("Standard error:")
    print(result["stderr"].rstrip() or "(empty)")


def run_demo() -> None:
    """Exercise success, errors, and timeout; E2B mode makes three billed API runs."""
    cases = [
        ("Successful execution", 'print("Hello from the code runner!")\nprint(2 + 3)',
         "SUCCESS", 5.0),
        ("Runtime error", 'print("Before the error")\nprint(1 / 0)', "ERROR", 5.0),
        ("Timeout", "while True:\n    pass", "TIMEOUT", 1.0),
    ]
    for name, code, expected_status, timeout in cases:
        print(f"\n--- {name} ---")
        result = execute_code(code, timeout_seconds=timeout)
        _print_result(result)
        if result["status"] != expected_status:
            raise RuntimeError(f"Expected {expected_status}, got {result['status']}")
    print("\nAll three execution checks passed.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute reviewed Python snippets with the configured code runner.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--demo", action="store_true", help="Run three built-in checks.")
    source.add_argument("--code", help="Python source code as a string.")
    source.add_argument("--file", type=Path, help="Read source code from a Python file.")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args()
    if args.demo:
        run_demo()
        return
    try:
        code = args.file.read_text(encoding="utf-8-sig") if args.file else args.code
        _print_result(execute_code(code, timeout_seconds=args.timeout))
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
