import argparse
import json
import random
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from avsd.code.data.code_dataset import normalize_code_example, write_jsonl
from avsd.code.execution.extract_code import extract_python_code
from avsd.code.execution.run_tests import run_code_on_tests
from avsd.code.execution.sandbox import check_sandbox_available
from avsd.code.schemas import TestCase


DEFAULT_COTS_DATASET = "open-r1/codeforces-cots"
DEFAULT_COTS_CONFIG = "solutions_py"
DEFAULT_PROBLEMS_DATASET = "open-r1/codeforces"
DEFAULT_PROBLEMS_CONFIG = "default"


@dataclass(frozen=True)
class DownloadConfig:
    train_output: Path = Path("data/codeforces_cots_py_train.jsonl")
    eval_output: Path = Path("data/codeforces_cots_py_eval.jsonl")
    train_size: int = 1000
    eval_size: int = 100
    seed: int = 1337
    cots_dataset: str = DEFAULT_COTS_DATASET
    cots_config: str = DEFAULT_COTS_CONFIG
    cots_split: str = "train"
    problems_dataset: str = DEFAULT_PROBLEMS_DATASET
    problems_config: str = DEFAULT_PROBLEMS_CONFIG
    problems_split: str = "train"
    cache_dir: str | None = None
    verify_reference: str = "public"
    timeout_s: float = 2.0
    unsafe_subprocess_sandbox: bool = False
    show_progress: bool = True


@dataclass
class DownloadStats:
    total_candidates: int = 0
    kept: int = 0
    reasons: Counter[str] = field(default_factory=Counter)
    verification_failures: Counter[str] = field(default_factory=Counter)

    def reject(self, reason: str) -> None:
        self.reasons[reason] += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_candidates": self.total_candidates,
            "kept": self.kept,
            "reasons": dict(self.reasons),
            "verification_failures": dict(self.verification_failures),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Python Codeforces-CoTs records from Hugging Face for code AVSD."
    )
    parser.add_argument("--train-output", default=str(DownloadConfig.train_output))
    parser.add_argument("--eval-output", default=str(DownloadConfig.eval_output))
    parser.add_argument("--train-size", type=int, default=DownloadConfig.train_size)
    parser.add_argument("--eval-size", type=int, default=DownloadConfig.eval_size)
    parser.add_argument("--seed", type=int, default=DownloadConfig.seed)
    parser.add_argument("--cots-dataset", default=DEFAULT_COTS_DATASET)
    parser.add_argument("--cots-config", default=DEFAULT_COTS_CONFIG)
    parser.add_argument("--cots-split", default="train")
    parser.add_argument("--problems-dataset", default=DEFAULT_PROBLEMS_DATASET)
    parser.add_argument("--problems-config", default=DEFAULT_PROBLEMS_CONFIG)
    parser.add_argument("--problems-split", default="train")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--verify-reference", choices=("none", "public", "all"), default="public")
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--disable-progress", action="store_true", help="Disable the candidate scan progress bar.")
    parser.add_argument(
        "--unsafe-subprocess-sandbox",
        action="store_true",
        help="Use a plain subprocess instead of bubblewrap. Intended only for local tests.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> DownloadConfig:
    if args.train_size < 0 or args.eval_size < 0:
        raise ValueError("--train-size and --eval-size must be non-negative")
    if args.train_size + args.eval_size <= 0:
        raise ValueError("at least one example must be requested")
    return DownloadConfig(
        train_output=Path(args.train_output),
        eval_output=Path(args.eval_output),
        train_size=args.train_size,
        eval_size=args.eval_size,
        seed=args.seed,
        cots_dataset=args.cots_dataset,
        cots_config=args.cots_config,
        cots_split=args.cots_split,
        problems_dataset=args.problems_dataset,
        problems_config=args.problems_config,
        problems_split=args.problems_split,
        cache_dir=args.cache_dir,
        verify_reference=args.verify_reference,
        timeout_s=args.timeout_s,
        unsafe_subprocess_sandbox=args.unsafe_subprocess_sandbox,
        show_progress=not args.disable_progress,
    )


def download_codeforces_cots_py(config: DownloadConfig) -> dict[str, Any]:
    if config.verify_reference != "none" and not config.unsafe_subprocess_sandbox:
        check_sandbox_available(raise_on_error=True)

    problem_rows = _load_hf_dataset(
        config.problems_dataset,
        config.problems_config,
        split=config.problems_split,
        cache_dir=config.cache_dir,
    )
    problem_index = build_problem_index(problem_rows)

    solution_rows = _load_hf_dataset(
        config.cots_dataset,
        config.cots_config,
        split=config.cots_split,
        cache_dir=config.cache_dir,
    )

    total_needed = config.train_size + config.eval_size
    stats = DownloadStats()
    selected: list[dict[str, Any]] = []
    seen_problem_ids: set[str] = set()
    progress = _make_progress_bar(
        total=_safe_len(solution_rows),
        target=total_needed,
        enabled=config.show_progress,
    )

    try:
        for solution_row in iter_shuffled(solution_rows, config.seed):
            stats.total_candidates += 1
            try:
                problem_id = _clean_str(solution_row.get("id"))
                if not problem_id:
                    stats.reject("missing_problem_id")
                    continue
                if problem_id in seen_problem_ids:
                    stats.reject("duplicate_problem_id")
                    continue

                problem_row = problem_index.get(problem_id)
                if problem_row is None:
                    stats.reject("missing_problem_metadata")
                    continue

                converted, reject_reason = convert_cots_py_row(
                    solution_row,
                    problem_row,
                    cots_dataset=config.cots_dataset,
                    cots_config=config.cots_config,
                    problems_dataset=config.problems_dataset,
                    problems_config=config.problems_config,
                )
                if converted is None:
                    stats.reject(reject_reason or "conversion_failed")
                    continue

                if config.verify_reference != "none":
                    example = normalize_code_example(converted)
                    tests = list(example.public_tests)
                    if config.verify_reference == "all":
                        tests += list(example.hidden_tests)
                    result = run_code_on_tests(
                        example.reference_solution,
                        tests,
                        timeout_s=config.timeout_s,
                        unsafe_subprocess_sandbox=config.unsafe_subprocess_sandbox,
                    )
                    if not result.passed:
                        stats.verification_failures[result.status] += 1
                        stats.reject(f"reference_{result.status}")
                        continue

                selected.append(converted)
                seen_problem_ids.add(problem_id)
                stats.kept += 1
                if len(selected) >= total_needed:
                    break
            finally:
                progress.update(stats)
    finally:
        progress.close()

    if len(selected) < total_needed:
        raise RuntimeError(
            "Not enough usable Python Codeforces-CoTs examples. "
            f"Requested {total_needed}, found {len(selected)}. "
            f"Stats: {json.dumps(stats.to_dict(), sort_keys=True)}"
        )

    train_rows = selected[: config.train_size]
    eval_rows = selected[config.train_size : config.train_size + config.eval_size]
    write_jsonl(config.train_output, train_rows)
    write_jsonl(config.eval_output, eval_rows)

    return {
        "train_output": str(config.train_output),
        "eval_output": str(config.eval_output),
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "seed": config.seed,
        "verify_reference": config.verify_reference,
        "stats": stats.to_dict(),
    }


def build_problem_index(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        problem_id = _clean_str(row.get("id"))
        if problem_id and problem_id not in index:
            index[problem_id] = dict(row)
    return index


def iter_shuffled(rows: Iterable[dict[str, Any]], seed: int) -> Iterator[dict[str, Any]]:
    if hasattr(rows, "shuffle"):
        yield from rows.shuffle(seed=seed)
        return

    materialized = list(rows)
    random.Random(seed).shuffle(materialized)
    yield from materialized


def _safe_len(rows: Any) -> int | None:
    try:
        return len(rows)
    except TypeError:
        return None


class _NoOpProgress:
    def update(self, stats: DownloadStats) -> None:
        return

    def close(self) -> None:
        return


class _TqdmProgress:
    def __init__(self, bar: Any, target: int) -> None:
        self.bar = bar
        self.target = target

    def update(self, stats: DownloadStats) -> None:
        self.bar.update(1)
        self.bar.set_postfix(
            kept=f"{stats.kept}/{self.target}",
            rejected=sum(stats.reasons.values()),
            refresh=False,
        )

    def close(self) -> None:
        self.bar.close()


class _StderrProgress:
    def __init__(self, total: int | None, target: int) -> None:
        self.total = total
        self.target = target
        self.last_reported = 0

    def update(self, stats: DownloadStats) -> None:
        should_report = (
            stats.total_candidates == 1
            or stats.total_candidates - self.last_reported >= 100
            or stats.kept >= self.target
        )
        if not should_report:
            return
        self.last_reported = stats.total_candidates
        total_text = f"/{self.total}" if self.total is not None else ""
        rejected = sum(stats.reasons.values())
        print(
            "Codeforces-CoTs scan: "
            f"seen={stats.total_candidates}{total_text} "
            f"kept={stats.kept}/{self.target} rejected={rejected}",
            file=sys.stderr,
        )

    def close(self) -> None:
        return


def _make_progress_bar(total: int | None, target: int, enabled: bool):
    if not enabled:
        return _NoOpProgress()
    try:
        from tqdm.auto import tqdm
    except ImportError:
        return _StderrProgress(total=total, target=target)

    bar = tqdm(
        total=total,
        desc="Codeforces-CoTs scan",
        unit="row",
        dynamic_ncols=True,
        leave=True,
    )
    return _TqdmProgress(bar=bar, target=target)


def convert_cots_py_row(
    solution_row: dict[str, Any],
    problem_row: dict[str, Any],
    *,
    cots_dataset: str = DEFAULT_COTS_DATASET,
    cots_config: str = DEFAULT_COTS_CONFIG,
    problems_dataset: str = DEFAULT_PROBLEMS_DATASET,
    problems_config: str = DEFAULT_PROBLEMS_CONFIG,
) -> tuple[dict[str, Any] | None, str | None]:
    problem_id = _clean_str(solution_row.get("id") or problem_row.get("id"))
    if not problem_id:
        return None, "missing_problem_id"

    problem_filter_reason = _problem_filter_reason(problem_row)
    if problem_filter_reason:
        return None, problem_filter_reason

    reference_solution, code_reason = extract_reference_solution(solution_row.get("generation"))
    if not reference_solution:
        return None, code_reason

    public_tests = _coerce_test_cases(problem_row.get("examples") or solution_row.get("examples"))
    if not public_tests:
        return None, "missing_public_tests"

    hidden_tests = _coerce_test_cases(problem_row.get("official_tests"))
    if not hidden_tests:
        return None, "missing_hidden_tests"

    problem_text = _build_problem_text(problem_row, solution_row)
    if not problem_text:
        return None, "missing_problem_text"

    row = {
        "id": problem_id,
        "source": f"{cots_dataset}/{cots_config}",
        "language": "python",
        "problem": problem_text,
        "input_format": _clean_str(problem_row.get("input_format") or solution_row.get("input_format")),
        "output_format": _clean_str(problem_row.get("output_format") or solution_row.get("output_format")),
        "constraints": "",
        "examples": [case.to_dict() for case in public_tests],
        "public_tests": [case.to_dict() for case in public_tests],
        "hidden_tests": [case.to_dict() for case in hidden_tests],
        "reference_solution": reference_solution,
        "algorithm_hint": _clean_str(problem_row.get("editorial") or solution_row.get("editorial")),
        "metadata": _json_safe(
            {
                "hf_cots_dataset": cots_dataset,
                "hf_cots_config": cots_config,
                "hf_problems_dataset": problems_dataset,
                "hf_problems_config": problems_config,
                "prompt": solution_row.get("prompt"),
                "finish_reason": solution_row.get("finish_reason"),
                "api_metadata": solution_row.get("api_metadata"),
                "contest_id": problem_row.get("contest_id") or solution_row.get("contest_id"),
                "contest_name": problem_row.get("contest_name") or solution_row.get("contest_name"),
                "contest_type": problem_row.get("contest_type") or solution_row.get("contest_type"),
                "contest_start_year": problem_row.get("contest_start_year")
                or solution_row.get("contest_start_year"),
                "index": problem_row.get("index") or solution_row.get("index"),
                "title": problem_row.get("title") or solution_row.get("title"),
                "rating": problem_row.get("rating") or solution_row.get("rating"),
                "tags": problem_row.get("tags") or solution_row.get("tags"),
                "time_limit": problem_row.get("time_limit") or solution_row.get("time_limit"),
                "memory_limit": problem_row.get("memory_limit") or solution_row.get("memory_limit"),
                "official_tests_complete": problem_row.get("official_tests_complete"),
                "testset_size": problem_row.get("testset_size"),
                "input_mode": problem_row.get("input_mode"),
            }
        ),
    }
    return row, None


def extract_reference_solution(generation: Any) -> tuple[str, str | None]:
    code = extract_python_code(_clean_str(generation))
    if not code.strip():
        return "", "missing_reference_code"
    try:
        compile(code, "<codeforces_cots_reference>", "exec")
    except SyntaxError:
        return "", "reference_syntax_error"
    return code, None


def _load_hf_dataset(
    dataset: str,
    config: str | None,
    *,
    split: str,
    cache_dir: str | None,
):
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The Hugging Face `datasets` package is required to download Codeforces-CoTs. "
            "Install it in the active environment before running this command."
        ) from exc

    kwargs: dict[str, Any] = {"split": split}
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    if config and str(config).lower() not in {"none", "null"}:
        return load_dataset(dataset, config, **kwargs)
    return load_dataset(dataset, **kwargs)


def _problem_filter_reason(problem_row: dict[str, Any]) -> str | None:
    if _clean_str(problem_row.get("interaction_format")):
        return "interactive"
    if _clean_str(problem_row.get("generated_checker")):
        return "custom_checker"
    input_mode = _clean_str(problem_row.get("input_mode") or "stdio").lower()
    if input_mode not in {"", "stdio", "stdin", "standard input"}:
        return "non_stdio"
    if problem_row.get("executable") is False:
        return "non_executable"
    return None


def _coerce_test_cases(raw_tests: Any) -> list[TestCase]:
    if not raw_tests:
        return []
    cases: list[TestCase] = []
    for item in raw_tests:
        if not isinstance(item, dict):
            continue
        raw_inputs = item.get("input")
        raw_outputs = item.get("output")
        if isinstance(raw_inputs, list) and isinstance(raw_outputs, list):
            for input_text, output_text in zip(raw_inputs, raw_outputs):
                cases.append(TestCase(input=str(input_text), output=str(output_text)))
        elif raw_inputs is not None and raw_outputs is not None:
            cases.append(TestCase(input=str(raw_inputs), output=str(raw_outputs)))
    return cases


def _build_problem_text(problem_row: dict[str, Any], solution_row: dict[str, Any]) -> str:
    title = _clean_str(problem_row.get("title") or solution_row.get("title"))
    description = _clean_str(problem_row.get("description") or solution_row.get("description"))
    note = _clean_str(problem_row.get("note") or solution_row.get("note"))
    parts = []
    if title:
        parts.append(title)
    if description:
        parts.append(description)
    if note:
        parts.append(f"Note:\n{note}")
    return "\n\n".join(parts).strip()


def _clean_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    return str(value)


def main() -> None:
    config = config_from_args(parse_args())
    report = download_codeforces_cots_py(config)
    print(json.dumps(report, indent=2, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
