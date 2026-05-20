import re

from avsd.common.chat_template import apply_chat_template_compat


EMPTY_THINK_BLOCK = "<think></think>"
_EMPTY_THINK_AT_END_RE = re.compile(r"<think>\s*</think>\s*$", flags=re.DOTALL)


def apply_code_chat_template(
    tokenizer,
    messages,
    *,
    tokenize: bool = False,
    add_generation_prompt: bool = False,
    enable_thinking: bool = False,
    **kwargs,
):
    rendered = apply_chat_template_compat(
        tokenizer,
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=enable_thinking,
        **kwargs,
    )
    if tokenize or enable_thinking or not add_generation_prompt:
        return rendered
    return ensure_empty_think_after_generation_prompt(rendered)


def ensure_empty_think_after_generation_prompt(prompt: str) -> str:
    prompt = str(prompt)
    stripped = prompt.rstrip()
    if _EMPTY_THINK_AT_END_RE.search(stripped):
        return prompt
    return stripped + "\n" + EMPTY_THINK_BLOCK + "\n"


def format_final_code_answer(code: str, *, include_empty_think: bool = False) -> str:
    answer = (
        "<final_code>\n"
        "```python\n"
        f"{str(code).rstrip()}\n"
        "```\n"
        "</final_code>"
    )
    if include_empty_think:
        return EMPTY_THINK_BLOCK + "\n\n" + answer
    return answer
