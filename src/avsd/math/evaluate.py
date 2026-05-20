import argparse
import json
import re
from pathlib import Path

from avsd.common.chat_template import apply_chat_template_compat


DATASETS = {
    "math500": ("HuggingFaceH4/MATH-500", "test"),
    "amo-bench": ("meituan-longcat/AMO-Bench", "test"),
    "minerva": ("math-ai/minervamath", "test"),
    "amc23": ("math-ai/amc23", "test"),
    "aime24": ("HuggingFaceH4/aime_2024", "train"),
    "aime25": ("yentinglin/aime_2025", "train"),
    "hmmt25": ("MathArena/hmmt_feb_2025", "train"),
}


def extract_boxed_answer(text: str | None) -> str | None:
    if not text:
        return None

    match_start = text.rfind("\\boxed")
    if match_start < 0:
        return None

    brace_start = text.find("{", match_start)
    if brace_start < 0:
        return None

    depth = 0
    for idx in range(brace_start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                return text[brace_start + 1 : idx].strip()
    return None


def grade_answer(predicted: str | None, ground_truth: str) -> bool:
    if predicted is None:
        return False

    try:
        from math_verify import parse, verify

        predicted_expr = predicted if "$" in predicted else f"${predicted}$"
        ground_truth_expr = ground_truth if "$" in ground_truth else f"${ground_truth}$"
        return verify(
            parse(ground_truth_expr, fallback_mode="no_fallback"),
            parse(predicted_expr, fallback_mode="no_fallback"),
            timeout_seconds=5,
        )
    except Exception:
        pred_norm = re.sub(r"[\s$]", "", predicted).lower()
        gt_norm = re.sub(r"[\s$]", "", ground_truth).lower()
        return pred_norm == gt_norm


def load_eval_dataset(name: str):
    from datasets import load_dataset

    dataset_id, split = DATASETS[name]
    dataset = load_dataset(dataset_id, split=split, trust_remote_code=name in {"aime25", "hmmt25"})
    return dataset


def example_to_problem_answer(example: dict, dataset_name: str) -> tuple[str, str, str | int | None]:
    if dataset_name == "math500":
        problem = example["problem"]
        answer = extract_boxed_answer(example["solution"]) or example["solution"]
        return problem, answer, None
    if dataset_name == "amo-bench":
        return example["prompt"], str(example["answer"]), example.get("question_id")
    if dataset_name in {"minerva", "amc23"}:
        return example["question"], str(example["answer"]), example.get("id")
    if dataset_name == "aime24":
        return example["problem"], str(example["answer"]), example.get("id")
    return example["problem"], str(example["answer"]), example.get("problem_idx")


def build_prompt(tokenizer, problem: str, enable_thinking: bool) -> str:
    message = {
        "role": "user",
        "content": f"{problem}\n\nPlease reason step by step, and put your final answer within \\boxed{{}}.",
    }
    return apply_chat_template_compat(
        tokenizer,
        [message],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def require_lora_checkpoint(checkpoint_dir: str) -> Path:
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

    if not (checkpoint_path / "adapter_model.safetensors").exists() and not (
        checkpoint_path / "adapter_model.bin"
    ).exists():
        raise FileNotFoundError(
            "Checkpoint directory must contain adapter_model.safetensors or adapter_model.bin."
        )
    return checkpoint_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a base model with one AVSD checkpoint.")
    parser.add_argument("--base_model", required=True, help="Base model name or local path.")
    parser.add_argument("--checkpoint_dir", required=True, help="LoRA checkpoint directory.")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS), help="Evaluation dataset.")
    parser.add_argument("--output_file", default=None, help="Path for the JSON result file.")
    parser.add_argument("--num_samples", type=int, default=None, help="Optional number of examples.")
    parser.add_argument("--val_n", type=int, default=1, help="Samples per problem.")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=-1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--no_thinking", action="store_true", help="Disable thinking-template hints.")
    args = parser.parse_args()

    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    checkpoint_path = require_lora_checkpoint(args.checkpoint_dir)
    enable_thinking = not args.no_thinking

    dataset = load_eval_dataset(args.dataset)
    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    prompts = []
    examples = []
    for example in dataset:
        problem, answer, problem_id = example_to_problem_answer(example, args.dataset)
        prompts.append(build_prompt(tokenizer, problem, enable_thinking))
        examples.append({"problem_id": problem_id, "problem": problem, "ground_truth": answer})

    llm_kwargs = {
        "model": args.base_model,
        "enable_lora": True,
        "max_lora_rank": 64,
        "max_loras": 1,
        "max_cpu_loras": 1,
        "trust_remote_code": True,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "distributed_executor_backend": "mp",
    }
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    sampling_params = SamplingParams(
        n=args.val_n,
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
    )
    lora_request = LoRARequest("avsd_checkpoint", 1, str(checkpoint_path))

    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request, use_tqdm=True)

    correct_total = 0
    formatted_total = 0
    results = []
    for idx, (example, output) in enumerate(tqdm(list(zip(examples, outputs)), desc="grading")):
        generations = []
        for sample in output.outputs:
            predicted = extract_boxed_answer(sample.text)
            correct = grade_answer(predicted, example["ground_truth"])
            formatted = predicted is not None
            correct_total += int(correct)
            formatted_total += int(formatted)
            generations.append(
                {
                    "predicted_answer": predicted,
                    "correct": correct,
                    "formatted": formatted,
                    "full_generation": sample.text,
                }
            )

        results.append(
            {
                "problem_id": example["problem_id"] if example["problem_id"] is not None else idx,
                "problem": example["problem"],
                "ground_truth": example["ground_truth"],
                "generations": generations,
                "num_correct": sum(item["correct"] for item in generations),
            }
        )

    total_generations = len(results) * args.val_n
    summary = {
        "base_model": args.base_model,
        "checkpoint_dir": str(checkpoint_path),
        "dataset": args.dataset,
        "num_problems": len(results),
        "val_n": args.val_n,
        "average_at_n_pct": 100.0 * correct_total / total_generations if total_generations else 0.0,
        "format_rate_pct": 100.0 * formatted_total / total_generations if total_generations else 0.0,
        "results": results,
    }

    output_file = Path(args.output_file or f"eval_results/{args.dataset}_avsd.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    print(f"Average@{args.val_n}: {summary['average_at_n_pct']:.2f}%")
    print(f"Format rate: {summary['format_rate_pct']:.2f}%")
    print(f"Saved results to: {output_file}")


if __name__ == "__main__":
    main()
