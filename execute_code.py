"""Run reviewed Python snippets in a separate process.

This local learning tool is not a security sandbox: executed code retains
the operating-system permissions of the current user. Do not expose this
runner to untrusted users. Deployments need a separate sandbox service.
"""

import argparse
import ast
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path


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


def execute_code(
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
    print(f"Status: {result['status']}")
    print(f"Exit code: {result['exit_code']}")
    if result["message"]:
        print(f"Message: {result['message']}")
    print("Standard output:")
    print(result["stdout"].rstrip() or "(empty)")
    print("Standard error:")
    print(result["stderr"].rstrip() or "(empty)")


def run_demo() -> None:
    """Exercise success, error capture, and timeout without any API calls."""
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
    parser = argparse.ArgumentParser(description="Execute reviewed local Python snippets.")
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
