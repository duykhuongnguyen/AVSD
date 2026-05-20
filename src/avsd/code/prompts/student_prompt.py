from avsd.code.schemas import CodeExample, TestCase
from avsd.code.prompts.thinking import apply_code_chat_template


FINAL_CODE_FORMAT = (
    "Final answer format:\n"
    "<final_code>\n"
    "```python\n"
    "# your code\n"
    "```\n"
    "</final_code>"
)


def format_test_cases(cases: list[TestCase]) -> str:
    if not cases:
        return "None provided."
    chunks = []
    for idx, case in enumerate(cases, start=1):
        chunks.append(
            f"Example {idx}\n"
            f"Input:\n{case.input.rstrip()}\n"
            f"Output:\n{case.output.rstrip()}"
        )
    return "\n\n".join(chunks)


def build_student_user_message(example: CodeExample | dict) -> str:
    if isinstance(example, dict):
        example = CodeExample.from_dict(example)
    return (
        "You are given a programming problem. Write a complete Python 3 solution.\n\n"
        "Provide the final executable solution directly.\n"
        "Put the final executable solution between <final_code> and </final_code> tags.\n"
        "Inside <final_code>, include only one complete Python code block and no explanation.\n\n"
        f"{format_problem_fields(example)}\n\n"
        f"{FINAL_CODE_FORMAT}"
    )


def format_problem_fields(example: CodeExample | dict) -> str:
    if isinstance(example, dict):
        example = CodeExample.from_dict(example)
    return (
        f"Problem:\n{example.problem}\n\n"
        f"Input format:\n{example.input_format or 'See problem statement.'}\n\n"
        f"Output format:\n{example.output_format or 'See problem statement.'}\n\n"
        f"Constraints:\n{example.constraints or 'See problem statement.'}\n\n"
        f"Examples:\n{format_test_cases(example.examples)}"
    )


def build_student_prompt(tokenizer, example: CodeExample | dict, *, enable_thinking: bool = False) -> str:
    messages = [{"role": "user", "content": build_student_user_message(example)}]
    return apply_code_chat_template(
        tokenizer,
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
