from dataclasses import dataclass, field
from typing import Any, Literal


ExecutionStatus = Literal[
    "passed",
    "wrong_answer",
    "runtime_error",
    "syntax_error",
    "timeout",
    "extraction_failure",
]


@dataclass(frozen=True)
class TestCase:
    input: str
    output: str

    @classmethod
    def from_obj(cls, obj: dict[str, Any]) -> "TestCase":
        if not isinstance(obj, dict):
            raise ValueError("test case must be an object")
        if "input" not in obj or "output" not in obj:
            raise ValueError("test case must contain input and output")
        return cls(input=str(obj["input"]), output=str(obj["output"]))

    def to_dict(self) -> dict[str, str]:
        return {"input": self.input, "output": self.output}


@dataclass
class CodeExample:
    id: str
    source: str
    language: str
    problem: str
    input_format: str = ""
    output_format: str = ""
    constraints: str = ""
    examples: list[TestCase] = field(default_factory=list)
    public_tests: list[TestCase] = field(default_factory=list)
    hidden_tests: list[TestCase] = field(default_factory=list)
    reference_solution: str = ""
    algorithm_hint: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "CodeExample":
        if not isinstance(row, dict):
            raise ValueError("example must be an object")
        example_id = str(row.get("id") or "").strip()
        if not example_id:
            raise ValueError("example is missing id")
        problem = str(row.get("problem") or "").strip()
        if not problem:
            raise ValueError(f"{example_id}: problem is empty")
        reference = str(row.get("reference_solution") or "").strip()
        if not reference:
            raise ValueError(f"{example_id}: reference_solution is empty")

        return cls(
            id=example_id,
            source=str(row.get("source") or ""),
            language=str(row.get("language") or ""),
            problem=problem,
            input_format=str(row.get("input_format") or ""),
            output_format=str(row.get("output_format") or ""),
            constraints=str(row.get("constraints") or ""),
            examples=_parse_tests(row.get("examples") or []),
            public_tests=_parse_tests(row.get("public_tests") or []),
            hidden_tests=_parse_tests(row.get("hidden_tests") or []),
            reference_solution=reference,
            algorithm_hint=str(row.get("algorithm_hint") or ""),
            metadata=dict(row.get("metadata") or {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "language": self.language,
            "problem": self.problem,
            "input_format": self.input_format,
            "output_format": self.output_format,
            "constraints": self.constraints,
            "examples": [case.to_dict() for case in self.examples],
            "public_tests": [case.to_dict() for case in self.public_tests],
            "hidden_tests": [case.to_dict() for case in self.hidden_tests],
            "reference_solution": self.reference_solution,
            "algorithm_hint": self.algorithm_hint,
            "metadata": self.metadata,
        }


@dataclass
class ExecutionResult:
    status: ExecutionStatus
    passed: bool
    feedback: str
    failing_input: str | None = None
    expected_output: str | None = None
    actual_output: str | None = None
    traceback: str | None = None
    error_type: str | None = None
    execution_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "passed": self.passed,
            "feedback": self.feedback,
            "failing_input": self.failing_input,
            "expected_output": self.expected_output,
            "actual_output": self.actual_output,
            "traceback": self.traceback,
            "error_type": self.error_type,
            "execution_time_s": self.execution_time_s,
        }


def _parse_tests(raw_tests: Any) -> list[TestCase]:
    if raw_tests is None:
        return []
    if not isinstance(raw_tests, list):
        raise ValueError("tests must be a list")
    return [TestCase.from_obj(item) for item in raw_tests]
