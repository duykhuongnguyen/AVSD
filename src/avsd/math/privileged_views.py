import logging
import math
import re
from collections import Counter


LOGGER = logging.getLogger(__name__)

VIEW_TYPES = ("full_solution", "partial_solution", "answer_only")
ANSWER_EXTRACTION_FALLBACK_COUNTER = Counter()
_PARTIAL_NOTICE = (
    "[The reference above is incomplete and may omit later steps or the final answer.]"
)


def _find_last_boxed_span(text: str) -> tuple[int, int] | None:
    marker = "\\boxed{"
    start = text.rfind(marker)
    while start != -1:
        depth = 1
        idx = start + len(marker)
        while idx < len(text) and depth > 0:
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
            idx += 1
        if depth == 0:
            return start, idx
        start = text.rfind(marker, 0, start)
    return None


def _remove_last_boxed_span(text: str) -> str:
    span = _find_last_boxed_span(text)
    if span is None:
        return text
    start, end = span
    return (text[:start] + text[end:]).strip()


def _split_by_double_newlines(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"\n\s*\n+", text) if segment.strip()]


def _split_by_step_boundaries(text: str) -> list[str]:
    pattern = r"(?=^\s*(?:[-*•]|\d+[.)]))"
    return [segment.strip() for segment in re.split(pattern, text, flags=re.MULTILINE) if segment.strip()]


def _split_by_sentences(text: str) -> list[str]:
    return [segment.strip() for segment in re.split(r"(?<=[.!?])\s+", text) if segment.strip()]


def _segment_solution(text: str) -> tuple[list[str], str] | None:
    strategies = (
        (_split_by_double_newlines, "\n\n"),
        (_split_by_step_boundaries, "\n"),
        (_split_by_sentences, " "),
    )
    for splitter, joiner in strategies:
        segments = splitter(text)
        if len(segments) > 1:
            return segments, joiner
    return None


def extract_final_answer(solution: str) -> str:
    solution = solution.strip()
    if not solution:
        return ""

    boxed_span = _find_last_boxed_span(solution)
    if boxed_span is not None:
        start, end = boxed_span
        return solution[start + len("\\boxed{") : end - 1].strip()

    lines = [line.strip() for line in solution.splitlines() if line.strip()]
    answer_pattern = re.compile(r"^(?:final answer|answer)\s*:\s*(.+)$", flags=re.IGNORECASE)
    for line in reversed(lines):
        match = answer_pattern.match(line)
        if match:
            return match.group(1).strip()

    if lines:
        ANSWER_EXTRACTION_FALLBACK_COUNTER["last_non_empty_line"] += 1
        LOGGER.warning("Falling back to last non-empty line for answer extraction.")
        return lines[-1]

    ANSWER_EXTRACTION_FALLBACK_COUNTER["tail_128_chars"] += 1
    LOGGER.warning("Falling back to final 128 characters for answer extraction.")
    return solution[-128:].strip()


def build_partial_solution(solution: str, keep_ratio: float = 0.5) -> str:
    keep_ratio = min(max(keep_ratio, 0.0), 1.0)
    base_solution = _remove_last_boxed_span(solution).strip()
    if not base_solution:
        return _PARTIAL_NOTICE

    segmented = _segment_solution(base_solution)
    if segmented is not None:
        segments, joiner = segmented
        keep_count = max(1, math.ceil(keep_ratio * len(segments)))
        if len(segments) > 1:
            keep_count = min(keep_count, len(segments) - 1)
        partial_text = joiner.join(segments[:keep_count]).strip()
    else:
        prefix_len = max(1, math.ceil(len(base_solution) * keep_ratio))
        if len(base_solution) > 1:
            prefix_len = min(prefix_len, len(base_solution) - 1)
        partial_text = base_solution[:prefix_len].rstrip()

    if not partial_text:
        partial_text = base_solution[:1]

    return f"{partial_text}\n\n{_PARTIAL_NOTICE}"


def get_view_payload(
    problem: str,
    solution: str,
    view_type: str,
    partial_solution_ratio: float,
) -> str:
    del problem
    if view_type == "full_solution":
        return solution.strip()
    if view_type == "partial_solution":
        return build_partial_solution(solution, keep_ratio=partial_solution_ratio)
    if view_type == "answer_only":
        answer = extract_final_answer(solution)
        return f"Verified final answer: {answer}"
    raise ValueError(f"Unsupported privileged-information view: {view_type}")


def build_teacher_user_message(problem: str, payload: str, view_type: str) -> str:
    if view_type not in VIEW_TYPES:
        raise ValueError(f"Unsupported privileged-information view: {view_type}")

    payload = payload.strip()
    
    return (
        f"Problem: {problem}\n\n"
        "Here is training-time reference information for this problem. It may contain a final answer, a hint, a partial solution, or a full solution to the problem.\n"
        f"=== Privileged Reference ({view_type}) Begin ===\n"
        f"{payload}\n"
        f"=== Privileged Reference ({view_type}) End ===\n\n"
        "Use the reference only as hidden guidance. Use it only to understand, verify, or correct your solution.\n"
        "Do not mention, quote, copy, cite, or refer to the privileged reference or its existence.\n"
        "Now solve the original problem independently using your own reasoning.\n"
        "Please reason step by step, and put your final answer within \\boxed{}."
    )