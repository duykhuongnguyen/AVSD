import inspect
from collections.abc import Mapping, Sequence


def _template_contains_name(template, name: str) -> bool:
    if isinstance(template, str):
        return name in template
    if isinstance(template, Mapping):
        return any(_template_contains_name(value, name) for value in template.values())
    if isinstance(template, Sequence) and not isinstance(template, (bytes, bytearray, str)):
        return any(_template_contains_name(value, name) for value in template)
    return False


def supports_chat_template_kwarg(tokenizer, name: str) -> bool:
    """Return whether it is safe and meaningful to pass a chat-template kwarg."""

    apply_chat_template = getattr(tokenizer, "apply_chat_template")
    try:
        parameters = inspect.signature(apply_chat_template).parameters
    except (TypeError, ValueError):
        return _template_contains_name(getattr(tokenizer, "chat_template", None), name)

    if name in parameters:
        return True

    has_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    if not has_var_kwargs:
        return False

    return _template_contains_name(getattr(tokenizer, "chat_template", None), name)


def apply_chat_template_compat(
    tokenizer,
    messages,
    *,
    tokenize=False,
    add_generation_prompt=False,
    enable_thinking=None,
    **kwargs,
):
    template_kwargs = dict(kwargs)
    if enable_thinking is not None and supports_chat_template_kwarg(
        tokenizer, "enable_thinking"
    ):
        template_kwargs["enable_thinking"] = enable_thinking

    return tokenizer.apply_chat_template(
        messages,
        tokenize=tokenize,
        add_generation_prompt=add_generation_prompt,
        **template_kwargs,
    )
