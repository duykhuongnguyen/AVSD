import argparse
import json
import re
import sys
from pathlib import Path

from avsd.code.data.code_dataset import normalize_code_example, read_jsonl, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cached code-domain privileged views.")
    parser.add_argument("--input", required=True, help="Input filtered JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL with algorithm_hint.")
    parser.add_argument("--model", default=None, help="Reserved for model-generated hints; static hints are used now.")
    parser.add_argument("--max-ref-chars", type=int, default=12000)
    parser.add_argument("--max-hint-chars", type=int, default=2000)
    parser.add_argument(
        "--require-existing-hints",
        action="store_true",
        help="Drop examples that do not already include algorithm_hint.",
    )
    return parser.parse_args()


def build_static_algorithm_hint(problem: str, reference_solution: str, max_chars: int = 2000) -> str:
    """Build a deterministic fallback hint from shallow code cues.

    This is intentionally conservative. It gives the training pipeline a stable
    hint field without pretending to be a model-generated algorithm summary.
    """

    imports = sorted(set(re.findall(r"^\s*(?:from\s+(\w+)|import\s+(\w+))", reference_solution, flags=re.M)))
    flat_imports = sorted({name for pair in imports for name in pair if name})
    code_lower = reference_solution.lower()
    cues = []
    for label, markers in [
        ("sorting", ("sort(", "sorted(")),
        ("heap/priority queue", ("heapq", "heappush", "heappop")),
        ("binary search", ("bisect", "while lo", "while l < r")),
        ("dynamic programming", ("dp", "memo")),
        ("graph traversal", ("dfs", "bfs", "deque", "adj")),
        ("prefix/suffix aggregation", ("prefix", "suffix", "accumulate")),
        ("hashing/counting", ("counter", "defaultdict", "dict", "set(")),
        ("modular arithmetic", ("%", "mod")),
    ]:
        if any(marker in code_lower for marker in markers):
            cues.append(label)

    problem_excerpt = " ".join(problem.split())[:500]
    import_text = ", ".join(flat_imports) if flat_imports else "standard Python built-ins"
    cue_text = ", ".join(cues) if cues else "implementation details from the accepted solution"
    hint = (
        "Key observation: infer the intended algorithm from the problem statement and the accepted solution.\n"
        f"Problem focus: {problem_excerpt}\n"
        f"Data structures and libraries indicated by the reference: {import_text}.\n"
        f"Likely techniques: {cue_text}.\n"
        "Main algorithm: implement the same input parsing, case handling, and output contract as the reference while avoiding copying its exact style.\n"
        "Edge cases: preserve behavior on the sample/public tests, empty or minimum-size inputs, duplicate values, and boundary constraints.\n"
        "Complexity: keep the same asymptotic structure as the accepted implementation."
    )
    return hint[:max_chars].rstrip()


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input)
    output_rows = []
    dropped_missing_hint = 0
    generated_static = 0

    for row in rows:
        example = normalize_code_example(row)
        out = example.to_dict()
        if len(out["reference_solution"]) > args.max_ref_chars:
            out["reference_solution"] = out["reference_solution"][: args.max_ref_chars].rstrip()
        hint = (out.get("algorithm_hint") or "").strip()
        if not hint:
            if args.require_existing_hints:
                dropped_missing_hint += 1
                continue
            hint = build_static_algorithm_hint(
                out["problem"],
                out["reference_solution"],
                max_chars=args.max_hint_chars,
            )
            generated_static += 1
        out["algorithm_hint"] = hint[: args.max_hint_chars].rstrip()
        output_rows.append(out)

    write_jsonl(args.output, output_rows)
    report = {
        "input": str(Path(args.input)),
        "output": str(Path(args.output)),
        "written": len(output_rows),
        "generated_static_hints": generated_static,
        "dropped_missing_hint": dropped_missing_hint,
        "model": args.model,
    }
    print(json.dumps(report, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
