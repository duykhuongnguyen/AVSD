import subprocess
import time
import traceback

from avsd.code.execution.feedback import build_feedback, truncate_text
from avsd.code.execution.sandbox import run_python_in_sandbox
from avsd.code.schemas import ExecutionResult, TestCase


def run_code_on_tests(
    code: str,
    tests: list[TestCase],
    timeout_s: float = 2.0,
    *,
    max_output_chars: int = 4000,
    memory_limit_mb: int = 512,
    unsafe_subprocess_sandbox: bool = False,
) -> ExecutionResult:
    """Run generated Python code on tests and return the first failure."""

    code = str(code or "")
    if not code.strip():
        result = ExecutionResult(
            status="extraction_failure",
            passed=False,
            feedback="No executable Python code was extracted.",
        )
        result.feedback = build_feedback(result)
        return result

    try:
        compile(code, "<student_solution>", "exec")
    except SyntaxError:
        tb = traceback.format_exc(limit=3)
        result = ExecutionResult(
            status="syntax_error",
            passed=False,
            feedback=tb,
            traceback=tb,
            error_type="SyntaxError",
        )
        result.feedback = build_feedback(result)
        return result

    total_time = 0.0
    for case in tests:
        start = time.perf_counter()
        try:
            proc = run_python_in_sandbox(
                code,
                case.input,
                timeout_s=timeout_s,
                max_output_chars=max_output_chars,
                memory_limit_mb=memory_limit_mb,
                unsafe_subprocess_sandbox=unsafe_subprocess_sandbox,
            )
            elapsed = time.perf_counter() - start
            total_time += elapsed
        except subprocess.TimeoutExpired as exc:
            result = ExecutionResult(
                status="timeout",
                passed=False,
                feedback="Timed out.",
                failing_input=case.input,
                traceback=truncate_text(getattr(exc, "stderr", "") or "", max_output_chars),
                execution_time_s=timeout_s,
            )
            result.feedback = build_feedback(result)
            return result

        if proc.returncode != 0:
            stderr = proc.stderr or proc.stdout or f"Process exited with return code {proc.returncode}."
            result = ExecutionResult(
                status="runtime_error",
                passed=False,
                feedback=stderr,
                failing_input=case.input,
                traceback=stderr,
                error_type=_infer_error_type(stderr, proc.returncode),
                execution_time_s=elapsed,
            )
            result.feedback = build_feedback(result)
            return result

        if _normalize_output(proc.stdout) != _normalize_output(case.output):
            result = ExecutionResult(
                status="wrong_answer",
                passed=False,
                feedback="Wrong answer.",
                failing_input=case.input,
                expected_output=case.output,
                actual_output=proc.stdout,
                execution_time_s=elapsed,
            )
            result.feedback = build_feedback(result)
            return result

    result = ExecutionResult(
        status="passed",
        passed=True,
        feedback="Passed all tests.",
        execution_time_s=total_time,
    )
    result.feedback = build_feedback(result)
    return result


def _normalize_output(text: str) -> str:
    lines = [line.rstrip() for line in str(text).replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines)


def _infer_error_type(stderr: str, returncode: int) -> str:
    for marker in (
        "AssertionError",
        "IndexError",
        "KeyError",
        "ValueError",
        "TypeError",
        "ZeroDivisionError",
        "RecursionError",
        "MemoryError",
        "NameError",
    ):
        if marker in stderr:
            return marker
    return f"returncode_{returncode}"
