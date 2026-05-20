from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, load_dataset


PUBLIC_MATH_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
OPENR1_MATH_DATASET = "open-r1/OpenR1-Math-220k"
OPENR1_MATH_CONFIG = "default"
OPENR1_MATH_SPLIT = "train"
DEFAULT_OPENR1_MATH_SOURCES = ("olympiads", "cn_contest", "aops_forum")
TRAINING_DATASET_OPENTHOUGHT = "openthought"
TRAINING_DATASET_OPENR1_MATH_15K = "openr1_math_15k"
TRAINING_DATASET_CHOICES = (
    TRAINING_DATASET_OPENTHOUGHT,
    TRAINING_DATASET_OPENR1_MATH_15K,
)


@dataclass(frozen=True)
class TrainingDatasetMetadata:
    training_dataset: str
    openthought_dataset: str
    openthought_num_rows: int
    openr1_math_dataset: str
    openr1_math_config: str
    openr1_math_split: str
    openr1_math_num_requested: int
    openr1_math_num_selected: int
    openr1_math_num_eligible: int
    openr1_math_seed: int
    openr1_math_sources: tuple[str, ...]
    train_num_rows: int


def parse_training_dataset_choice(raw_choice: str) -> str:
    choice = str(raw_choice).strip().lower()
    if choice not in TRAINING_DATASET_CHOICES:
        raise ValueError(
            "training_dataset must be one of: " + ", ".join(TRAINING_DATASET_CHOICES)
        )
    return choice


def parse_openr1_math_sources(raw_sources: str | Iterable[str]) -> tuple[str, ...]:
    if isinstance(raw_sources, str):
        source_items = raw_sources.split(",")
    else:
        source_items = raw_sources

    sources: list[str] = []
    seen = set()
    for source in source_items:
        normalized = str(source).strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        sources.append(normalized)

    if not sources:
        raise ValueError("openr1_math_sources must contain at least one source.")
    return tuple(sources)


def _sequence_value(values: Any, index: int) -> Any:
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
        if 0 <= index < len(values):
            return values[index]
    return None


def _is_true(value: Any) -> bool:
    return value is True or str(value).strip().lower() == "true"


def select_verified_complete_generation(example: dict[str, Any]) -> str | None:
    generations = example.get("generations") or []
    if not isinstance(generations, Sequence) or isinstance(generations, (str, bytes)):
        return None

    complete_flags = example.get("is_reasoning_complete") or []
    math_verify_flags = example.get("correctness_math_verify") or []
    llama_flags = example.get("correctness_llama") or []

    for idx, generation in enumerate(generations):
        if not _is_true(_sequence_value(complete_flags, idx)):
            continue
        is_verified = _is_true(_sequence_value(math_verify_flags, idx)) or _is_true(
            _sequence_value(llama_flags, idx)
        )
        if not is_verified:
            continue
        if isinstance(generation, str) and generation.strip():
            return generation.strip()
    return None


def convert_openr1_math_row(
    example: dict[str, Any],
    allowed_sources: Iterable[str] = DEFAULT_OPENR1_MATH_SOURCES,
) -> dict[str, str] | None:
    allowed_source_set = set(allowed_sources)
    if example.get("source") not in allowed_source_set:
        return None

    problem = example.get("problem")
    if not isinstance(problem, str) or not problem.strip():
        return None

    solution = select_verified_complete_generation(example)
    if solution is None:
        return None

    return {"problem": problem.strip(), "solution": solution}


def build_openr1_math_dataset(
    raw_dataset: Dataset,
    *,
    num_samples: int,
    seed: int,
    sources: str | Iterable[str],
) -> tuple[Dataset, dict[str, Any]]:
    if num_samples <= 0:
        raise ValueError("openr1_math_num_samples must be positive.")

    allowed_sources = parse_openr1_math_sources(sources)
    source_counts: Counter[str] = Counter()
    eligible_source_counts: Counter[str] = Counter()
    converted_rows: list[dict[str, str]] = []

    for example in raw_dataset:
        source = str(example.get("source") or "")
        source_counts[source] += 1
        converted = convert_openr1_math_row(example, allowed_sources=allowed_sources)
        if converted is None:
            continue
        converted_rows.append(converted)
        eligible_source_counts[source] += 1

    if len(converted_rows) < num_samples:
        raise ValueError(
            "OpenR1 Math filter produced "
            f"{len(converted_rows)} eligible rows, fewer than requested {num_samples}. "
            f"Allowed sources: {', '.join(allowed_sources)}."
        )

    selected = Dataset.from_list(converted_rows).shuffle(seed=seed).select(range(num_samples))
    stats = {
        "num_eligible": len(converted_rows),
        "source_counts": dict(source_counts),
        "eligible_source_counts": dict(eligible_source_counts),
        "sources": allowed_sources,
    }
    return selected, stats


def _train_split(dataset_or_dict: Any) -> Dataset:
    if isinstance(dataset_or_dict, Dataset):
        return dataset_or_dict
    return dataset_or_dict["train"]


def _project_problem_solution(dataset: Dataset, dataset_name: str) -> Dataset:
    required_columns = ("problem", "solution")
    missing_columns = [column for column in required_columns if column not in dataset.column_names]
    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: {', '.join(missing_columns)}"
        )

    extra_columns = [column for column in dataset.column_names if column not in required_columns]
    if extra_columns:
        return dataset.remove_columns(extra_columns)
    return dataset


def load_training_dataset(
    *,
    training_dataset: str = TRAINING_DATASET_OPENTHOUGHT,
    openr1_math_num_samples: int = 15_000,
    openr1_math_seed: int = 42,
    openr1_math_sources: str | Iterable[str] = DEFAULT_OPENR1_MATH_SOURCES,
    load_dataset_fn: Callable[..., Any] = load_dataset,
) -> tuple[Dataset, TrainingDatasetMetadata]:
    training_dataset = parse_training_dataset_choice(training_dataset)
    sources = DEFAULT_OPENR1_MATH_SOURCES
    openthought_dataset = ""
    openthought_num_rows = 0
    openr1_num_selected = 0
    openr1_num_eligible = 0

    if training_dataset == TRAINING_DATASET_OPENTHOUGHT:
        openthought_dataset = PUBLIC_MATH_DATASET
        train_dataset = _train_split(load_dataset_fn(openthought_dataset))
        train_dataset = _project_problem_solution(train_dataset, openthought_dataset)
        openthought_num_rows = len(train_dataset)
    elif training_dataset == TRAINING_DATASET_OPENR1_MATH_15K:
        openr1_raw = load_dataset_fn(
            OPENR1_MATH_DATASET,
            OPENR1_MATH_CONFIG,
            split=OPENR1_MATH_SPLIT,
        )
        train_dataset, openr1_stats = build_openr1_math_dataset(
            openr1_raw,
            num_samples=openr1_math_num_samples,
            seed=openr1_math_seed,
            sources=openr1_math_sources,
        )
        sources = openr1_stats["sources"]
        openr1_num_eligible = openr1_stats["num_eligible"]
        openr1_num_selected = len(train_dataset)
    else:
        raise AssertionError(f"Unhandled training dataset: {training_dataset}")

    metadata = TrainingDatasetMetadata(
        training_dataset=training_dataset,
        openthought_dataset=openthought_dataset or "",
        openthought_num_rows=openthought_num_rows,
        openr1_math_dataset=OPENR1_MATH_DATASET,
        openr1_math_config=OPENR1_MATH_CONFIG,
        openr1_math_split=OPENR1_MATH_SPLIT,
        openr1_math_num_requested=(
            openr1_math_num_samples
            if training_dataset == TRAINING_DATASET_OPENR1_MATH_15K
            else 0
        ),
        openr1_math_num_selected=openr1_num_selected,
        openr1_math_num_eligible=openr1_num_eligible,
        openr1_math_seed=openr1_math_seed,
        openr1_math_sources=sources,
        train_num_rows=len(train_dataset),
    )
    return train_dataset, metadata
