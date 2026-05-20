from avsd.code.prompts.student_prompt import FINAL_CODE_FORMAT, format_problem_fields
from avsd.code.prompts.thinking import apply_code_chat_template
from avsd.code.schemas import CodeExample


CODE_VIEW_TYPES = ("reference", "hint", "feedback")


def normalize_code_views(raw_views: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    if isinstance(raw_views, str):
        items = raw_views.split(",")
    else:
        items = raw_views
    views: list[str] = []
    seen = set()
    for item in items:
        view = str(item).strip().lower()
        if not view:
            continue
        if view not in CODE_VIEW_TYPES:
            raise ValueError(f"Unsupported code view '{view}'. Supported: {', '.join(CODE_VIEW_TYPES)}")
        if view not in seen:
            seen.add(view)
            views.append(view)
    if not views:
        raise ValueError("At least one code view is required.")
    return tuple(views)


def build_reference_teacher_user_message(example: CodeExample | dict) -> str:
    if isinstance(example, dict):
        example = CodeExample.from_dict(example)
    payload = (
        "Accepted reference implementation:\n"
        "```python\n"
        f"{example.reference_solution.rstrip()}\n"
        "```"
    )
    return build_privileged_teacher_user_message(example, "reference", payload)


def build_hint_teacher_user_message(example: CodeExample | dict) -> str:
    if isinstance(example, dict):
        example = CodeExample.from_dict(example)
    hint = example.algorithm_hint.strip() or "No algorithm hint is available; rely on the problem statement."
    return build_privileged_teacher_user_message(example, "hint", f"Algorithm hint:\n{hint}")


def build_feedback_teacher_user_message(example: CodeExample | dict, feedback: str) -> str:
    if isinstance(example, dict):
        example = CodeExample.from_dict(example)
    return build_privileged_teacher_user_message(example, "feedback", f"Execution feedback:\n{feedback.strip()}")


def build_privileged_teacher_user_message(example: CodeExample | dict, view: str, payload: str) -> str:
    if isinstance(example, dict):
        example = CodeExample.from_dict(example)
    view = view.strip().lower()
    payload = payload.strip()
    return (
        "You are given a programming problem. Write a complete Python 3 solution.\n\n"
        "Provide the final executable solution directly.\n"
        "You are also given private training-time reference information for this problem.\n"
        "Use the private information only as hidden guidance to understand, verify, or correct your solution.\n"
        "Do not mention, quote, copy, cite, or refer to the private information or its existence in your response.\n"
        "Put the final executable solution between <final_code> and </final_code> tags.\n"
        "Inside <final_code>, include only one complete Python code block and no explanation.\n\n"
        f"=== Privileged Reference ({view}) Begin ===\n"
        f"{payload}\n"
        f"=== Privileged Reference ({view}) End ===\n\n"
        f"{format_problem_fields(example)}\n\n"
        f"{FINAL_CODE_FORMAT}"
    )


def build_teacher_user_message(example: CodeExample | dict, view: str, *, feedback: str | None = None) -> str:
    view = view.strip().lower()
    if view == "reference":
        return build_reference_teacher_user_message(example)
    if view == "hint":
        return build_hint_teacher_user_message(example)
    if view == "feedback":
        if feedback is None:
            raise ValueError("feedback view requires feedback text")
        return build_feedback_teacher_user_message(example, feedback)
    raise ValueError(f"Unsupported code view '{view}'. Supported: {', '.join(CODE_VIEW_TYPES)}")


def build_teacher_prompt(
    tokenizer,
    example: CodeExample | dict,
    view: str,
    *,
    feedback: str | None = None,
    enable_thinking: bool = False,
) -> str:
    messages = [{"role": "user", "content": build_teacher_user_message(example, view, feedback=feedback)}]
    return apply_code_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
