from avsd.code.schemas import ExecutionResult


def truncate_text(text: str | None, max_chars: int = 1200) -> str:
    if text is None:
        return ""
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "\n...[truncated]"


def build_feedback(result: ExecutionResult, max_chars: int = 1200) -> str:
    if result.status == "passed":
        return (
            "The student's solution passed all available public tests.\n"
            "Preserve the correct behavior and avoid unnecessary changes."
        )
    if result.status == "syntax_error":
        return (
            "The student's solution failed with a syntax error.\n\n"
            f"Error:\n{truncate_text(result.traceback or result.feedback, max_chars)}\n\n"
            "The solution should be corrected."
        )
    if result.status == "runtime_error":
        return (
            "The student's solution failed with a runtime error.\n\n"
            f"Error type:\n{truncate_text(result.error_type or 'runtime_error', max_chars)}\n\n"
            f"Traceback:\n{truncate_text(result.traceback or result.feedback, max_chars)}\n\n"
            f"Failing input:\n{truncate_text(result.failing_input, max_chars)}"
        )
    if result.status == "wrong_answer":
        return (
            "The student's solution failed on a test case.\n\n"
            f"Failing input:\n{truncate_text(result.failing_input, max_chars)}\n\n"
            f"Expected output:\n{truncate_text(result.expected_output, max_chars)}\n\n"
            f"Student output:\n{truncate_text(result.actual_output, max_chars)}"
        )
    if result.status == "timeout":
        return (
            "The student's solution timed out.\n\n"
            f"Failing input:\n{truncate_text(result.failing_input, max_chars)}\n\n"
            "The solution is likely too slow or has an infinite loop."
        )
    if result.status == "extraction_failure":
        return (
            "The student's response did not contain executable Python code.\n"
            "The solution should be rewritten as a complete Python 3 program."
        )
    return truncate_text(result.feedback, max_chars)
