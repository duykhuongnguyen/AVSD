from collections import Counter
from typing import Any


def compute_code_metrics(problem_records: list[dict[str, Any]], num_samples: int) -> dict[str, float]:
    if not problem_records:
        return {
            "num_problems": 0,
            "num_samples": num_samples,
            "pass@1": 0.0,
            f"avg@{num_samples}": 0.0,
            f"pass@{num_samples}": 0.0,
        }

    total_problem = len(problem_records)
    total_samples = 0
    total_passed = 0
    pass_at_1 = 0
    pass_at_k = 0
    status_counts: Counter[str] = Counter()
    token_total = 0
    exec_time_total = 0.0
    public_total = 0
    public_passed = 0

    for problem in problem_records:
        samples = problem.get("samples") or []
        if samples and samples[0].get("passed"):
            pass_at_1 += 1
        if any(sample.get("passed") for sample in samples):
            pass_at_k += 1
        for sample in samples:
            total_samples += 1
            if sample.get("passed"):
                total_passed += 1
            status_counts[str(sample.get("status") or "unknown")] += 1
            token_total += int(sample.get("generated_tokens") or 0)
            exec_time_total += float(sample.get("execution_time_s") or 0.0)
            public_status = sample.get("public_status")
            if public_status is not None:
                public_total += 1
                if public_status == "passed":
                    public_passed += 1

    denom_samples = max(1, total_samples)
    hidden_pass_rate = total_passed / denom_samples
    public_pass_rate = public_passed / public_total if public_total else 0.0
    metrics = {
        "num_problems": float(total_problem),
        "num_samples": float(num_samples),
        "pass@1": pass_at_1 / total_problem,
        f"avg@{num_samples}": hidden_pass_rate,
        f"pass@{num_samples}": pass_at_k / total_problem,
        "accepted_rate": hidden_pass_rate,
        "average_generated_tokens": token_total / denom_samples,
        "average_execution_time_s": exec_time_total / denom_samples,
        "public_pass_rate": public_pass_rate,
        "public_hidden_gap": public_pass_rate - hidden_pass_rate if public_total else 0.0,
    }
    for status in ("wrong_answer", "runtime_error", "syntax_error", "timeout", "extraction_failure", "passed"):
        metrics[f"{status}_rate"] = status_counts.get(status, 0) / denom_samples
    return metrics
