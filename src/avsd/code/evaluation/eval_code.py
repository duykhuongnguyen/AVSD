import argparse
import json
import sys
from pathlib import Path

from avsd.code.data.code_dataset import normalize_code_example, read_jsonl, tests_for_split, write_jsonl
from avsd.code.evaluation.compute_metrics import compute_code_metrics
from avsd.code.execution.extract_code import extract_python_code
from avsd.code.execution.run_tests import run_code_on_tests
from avsd.code.execution.sandbox import check_sandbox_available
from avsd.code.prompts.student_prompt import build_student_prompt
from avsd.code.training.vllm_utils import CodeVLLMSamplingConfig, load_vllm_model_for_eval, make_sampling_params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate code-domain models on JSONL tests.")
    parser.add_argument("--model", "--base_model", dest="model", required=True)
    parser.add_argument("--checkpoint-dir", "--checkpoint_dir", dest="checkpoint_dir", default=None)
    parser.add_argument("--dataset", required=True, help="Prepared JSONL dataset.")
    parser.add_argument("--output", "--output_file", dest="output", required=True)
    parser.add_argument("--tests", choices=["hidden", "public", "all"], default="hidden")
    parser.add_argument("--num-samples", "--num_samples", dest="num_samples", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", "--top_p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--top-k", "--top_k", dest="top_k", type=int, default=0)
    parser.add_argument("--min-p", "--min_p", dest="min_p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", "--presence_penalty", dest="presence_penalty", type=float, default=0.0)
    parser.add_argument(
        "--repetition-penalty",
        "--repetition_penalty",
        dest="repetition_penalty",
        type=float,
        default=1.0,
    )
    parser.add_argument("--max-tokens", "--max_tokens", dest="max_tokens", type=int, default=4096)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=None)
    parser.add_argument("--enable-lora", action="store_true")
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--distributed-executor-backend", default="mp")
    parser.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false", default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-s", type=float, default=2.0)
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--compute-public-hidden-gap", action="store_true", default=True)
    parser.add_argument(
        "--unsafe-subprocess-sandbox",
        action="store_true",
        help="Use a plain subprocess instead of bubblewrap. Intended only for local tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.unsafe_subprocess_sandbox:
        check_sandbox_available(raise_on_error=True)

    llm, tokenizer, lora_request = load_vllm_model_for_eval(
        model_name_or_path=args.model,
        checkpoint_dir=args.checkpoint_dir,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        enable_lora=args.enable_lora,
        max_lora_rank=args.max_lora_rank,
        distributed_executor_backend=args.distributed_executor_backend,
        enforce_eager=args.enforce_eager,
    )

    examples = [normalize_code_example(row) for row in read_jsonl(args.dataset)]
    if args.limit is not None:
        examples = examples[: args.limit]

    prompts = [
        build_student_prompt(tokenizer, example, enable_thinking=args.enable_thinking)
        for example in examples
    ]
    sampling = CodeVLLMSamplingConfig(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        repetition_penalty=args.repetition_penalty,
        n=args.num_samples,
    )
    generate_kwargs = {"sampling_params": make_sampling_params(sampling), "use_tqdm": True}
    if lora_request is not None:
        generate_kwargs["lora_request"] = lora_request
    outputs = llm.generate(prompts, **generate_kwargs)

    problem_records = []
    for problem_idx, (example, request_output) in enumerate(zip(examples, outputs), start=1):
        eval_tests = tests_for_split(example, args.tests)
        if not eval_tests:
            raise ValueError(f"{example.id}: no tests available for split '{args.tests}'")
        samples = []
        for sample_idx, generated in enumerate(request_output.outputs):
            completion_ids = list(generated.token_ids)
            completion = generated.text or tokenizer.decode(completion_ids, skip_special_tokens=False)
            code = extract_python_code(completion)
            result = run_code_on_tests(
                code,
                eval_tests,
                timeout_s=args.timeout_s,
                unsafe_subprocess_sandbox=args.unsafe_subprocess_sandbox,
            )
            public_status = None
            if args.compute_public_hidden_gap and args.tests == "hidden" and example.public_tests:
                public_result = run_code_on_tests(
                    code,
                    list(example.public_tests),
                    timeout_s=args.timeout_s,
                    unsafe_subprocess_sandbox=args.unsafe_subprocess_sandbox,
                )
                public_status = public_result.status
            samples.append(
                {
                    "sample_index": sample_idx,
                    "status": result.status,
                    "passed": result.passed,
                    "public_status": public_status,
                    "generated_tokens": len(completion_ids),
                    "execution_time_s": result.execution_time_s,
                    "completion": completion,
                    "extracted_code": code,
                    "execution": result.to_dict(),
                }
            )
        record = {"id": example.id, "source": example.source, "samples": samples}
        problem_records.append(record)
        print(
            f"[{problem_idx}/{len(examples)}] {example.id}: "
            f"{sum(sample['passed'] for sample in samples)}/{len(samples)} passed",
            file=sys.stderr,
        )

    write_jsonl(args.output, problem_records)
    metrics = compute_code_metrics(problem_records, args.num_samples)
    summary_path = Path(args.output).with_suffix(Path(args.output).suffix + ".summary.json")
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
