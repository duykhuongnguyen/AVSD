import re


FENCED_BLOCK_RE = re.compile(r"```(?P<lang>[^\n`]*)\n(?P<body>.*?)```", flags=re.DOTALL)
FINAL_CODE_RE = re.compile(r"<final_code>\s*(?P<body>.*?)\s*</final_code>", flags=re.DOTALL | re.IGNORECASE)


def extract_python_code(response: str) -> str:
    """Extract executable Python code from a model response.

    Preference order:
    1. Last <final_code>...</final_code> region, if present.
    2. Last fenced block whose language starts with python/py.
    3. Last generic fenced block.
    4. Full response, with obvious leading prose stripped when code starts later.
    """

    response = str(response or "").strip()
    if not response:
        return ""

    final_code_blocks = list(FINAL_CODE_RE.finditer(response))
    if final_code_blocks:
        return _extract_from_code_region(final_code_blocks[-1].group("body"))

    return _extract_from_code_region(response)


def _extract_from_code_region(text: str) -> str:
    blocks = list(FENCED_BLOCK_RE.finditer(text))
    python_blocks = [
        match
        for match in blocks
        if match.group("lang").strip().lower().replace(" ", "").startswith(("python", "py"))
    ]
    if python_blocks:
        return _clean_code(python_blocks[-1].group("body"))
    if blocks:
        return _clean_code(blocks[-1].group("body"))
    return _clean_code(_strip_leading_prose(text))


def _strip_leading_prose(text: str) -> str:
    lines = text.splitlines()
    code_start_markers = (
        "import ",
        "from ",
        "def ",
        "class ",
        "if __name__",
        "n =",
        "t =",
        "for ",
        "while ",
        "try:",
    )
    for idx, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith(code_start_markers) or stripped in {"# your code", "#!/usr/bin/env python3"}:
            return "\n".join(lines[idx:])
    return text


def _clean_code(code: str) -> str:
    code = code.replace("\r\n", "\n").replace("\r", "\n")
    lines = code.splitlines()
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)
