import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from avsd.code.schemas import CodeExample, TestCase


PYTHON_LANGUAGES = {"py", "python", "python3", "python 3", "pypy", "pypy3"}


@dataclass(frozen=True)
class FilterConfig:
    require_hidden_tests: bool = False
    max_problem_chars: int | None = None
    max_reference_chars: int | None = None
    allow_special_judge: bool = False
    allow_interactive: bool = False


@dataclass
class FilterStats:
    total: int = 0
    kept: int = 0
    reasons: Counter[str] | None = None

    def __post_init__(self) -> None:
        if self.reasons is None:
            self.reasons = Counter()

    def reject(self, reason: str) -> None:
        self.reasons[reason] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "kept": self.kept,
            "reasons": dict(self.reasons or {}),
        }


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_code_example(row: dict[str, Any]) -> CodeExample:
    normalized = dict(row)
    if "id" not in normalized:
        for key in ("problem_id", "name", "task_id"):
            if key in normalized:
                normalized["id"] = normalized[key]
                break
    if "reference_solution" not in normalized:
        for key in ("solution", "accepted_solution", "canonical_solution"):
            if key in normalized:
                normalized["reference_solution"] = normalized[key]
                break
    if "public_tests" not in normalized and "tests" in normalized:
        normalized["public_tests"] = normalized["tests"]
    if "hidden_tests" not in normalized and "private_tests" in normalized:
        normalized["hidden_tests"] = normalized["private_tests"]
    if "language" not in normalized:
        normalized["language"] = "python"
    return CodeExample.from_dict(normalized)


def validation_errors(example: CodeExample, config: FilterConfig | None = None) -> list[str]:
    config = config or FilterConfig()
    errors: list[str] = []

    if example.language.strip().lower() not in PYTHON_LANGUAGES:
        errors.append("non_python")
    if not example.reference_solution.strip():
        errors.append("missing_reference_solution")
    if not example.public_tests:
        errors.append("missing_public_tests")
    if config.require_hidden_tests and not example.hidden_tests:
        errors.append("missing_hidden_tests")
    if config.max_problem_chars is not None and len(example.problem) > config.max_problem_chars:
        errors.append("problem_too_long")
    if config.max_reference_chars is not None and len(example.reference_solution) > config.max_reference_chars:
        errors.append("reference_too_long")
    if not config.allow_interactive and _looks_interactive(example):
        errors.append("interactive")
    if not config.allow_special_judge and _looks_special_judge(example):
        errors.append("special_judge")
    if _looks_multifile(example):
        errors.append("multi_file")
    return errors


def filter_code_examples(
    rows: Iterable[dict[str, Any]],
    config: FilterConfig | None = None,
) -> tuple[list[CodeExample], FilterStats]:
    config = config or FilterConfig()
    stats = FilterStats()
    kept: list[CodeExample] = []

    for row in rows:
        stats.total += 1
        try:
            example = normalize_code_example(row)
        except ValueError as exc:
            stats.reject(str(exc))
            continue
        errors = validation_errors(example, config)
        if errors:
            for error in errors:
                stats.reject(error)
            continue
        kept.append(example)
        stats.kept += 1
    return kept, stats


def tests_for_split(example: CodeExample | dict[str, Any], split: str) -> list[TestCase]:
    if isinstance(example, dict):
        example = normalize_code_example(example)
    split = split.lower()
    if split == "public":
        return list(example.public_tests)
    if split == "hidden":
        return list(example.hidden_tests)
    if split == "examples":
        return list(example.examples)
    if split == "all":
        return list(example.public_tests) + list(example.hidden_tests)
    raise ValueError("split must be one of: public, hidden, examples, all")


def examples_to_dataset_rows(examples: Iterable[CodeExample]) -> list[dict[str, Any]]:
    return [example.to_dict() for example in examples]


def _looks_interactive(example: CodeExample) -> bool:
    text = " ".join(
        [
            example.problem,
            example.input_format,
            example.output_format,
            str(example.metadata.get("interactive") or ""),
        ]
    ).lower()
    return "interactive" in text or bool(example.metadata.get("interactive") is True)


def _looks_special_judge(example: CodeExample) -> bool:
    text = " ".join(
        [
            example.problem,
            example.output_format,
            str(example.metadata.get("special_judge") or ""),
            str(example.metadata.get("checker") or ""),
        ]
    ).lower()
    markers = ("special judge", "custom checker", "checker", "any valid")
    return any(marker in text for marker in markers) or bool(example.metadata.get("special_judge") is True)


def _looks_multifile(example: CodeExample) -> bool:
    metadata = example.metadata
    if metadata.get("multi_file") is True or metadata.get("files"):
        return True
    text = example.problem.lower()
    return "multiple files" in text or "multi-file" in text
