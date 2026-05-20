import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import AutoTokenizer
from trl import ModelConfig, TrlParser, get_kbit_device_map, get_peft_config, get_quantization_config

try:
    from trl.experimental.gold import GOLDConfig
except ImportError:
    from avsd.math.trainer import GOLDConfig

from avsd.code.data.code_dataset import read_jsonl
from avsd.code.training.code_trainer import CodeAVSDTrainer


@dataclass
class CodeAVSDScriptArguments:
    train_file: str = field(metadata={"help": "Prepared code-domain JSONL training file."})
    run_config: str | None = field(
        default=None,
        metadata={"help": "Run name. Appended to output_dir and used as run_name when set."},
    )
    views: str = field(default="reference,hint,feedback")
    multi_view_mode: str = field(default="avsd", metadata={"help": "single | consensus | arithmetic | avsd"})
    single_view: str = field(default="reference", metadata={"help": "View to use when multi_view_mode=single."})
    max_prompt_length: int = field(default=8192)
    max_response_tokens: int | None = field(
        default=None,
        metadata={"help": "Backward-compatible alias for GOLDConfig.max_completion_length."},
    )
    avsd_gate_mode: str = field(default="avsd", metadata={"help": "sigmoid | consistency_exp | avsd"})
    avsd_gate_alpha: float = field(default=6.0)
    avsd_gate_var_coef: float = field(default=1.0)
    avsd_gate_gap_coef: float = field(default=1.0)
    avsd_sign_threshold: float = field(default=0.8)
    avsd_consistency_exp_variance_mode: str = field(default="on")
    avsd_consistency_exp_var_coef: float = field(default=1.0)
    use_tinker_loss: bool = field(
        default=True,
        metadata={"help": "Compatibility flag. Code AVSD always uses sampled-token Tinker loss."},
    )
    fixed_teacher: bool = field(default=False)
    student_enable_thinking: bool = field(default=False)
    teacher_enable_thinking: bool = field(default=True)
    execution_timeout_s: float = field(default=2.0)
    unsafe_subprocess_sandbox: bool = field(default=True)
    generation_min_p: float = field(default=0.0)
    presence_penalty: float = field(default=0.0)
    generation_presence_penalty: float | None = field(default=None)
    generation_repetition_penalty: float = field(default=1.0)


def main() -> None:
    parser = TrlParser((CodeAVSDScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    _normalize_model_config(model_args)
    _resolve_code_aliases(script_args, training_args)
    _force_numeric_gradient_accumulation_steps(training_args)
    _configure_run_metadata(script_args, training_args, model_args)
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError("fixed_teacher=True requires use_peft=True.")

    model_dtype = _resolve_model_dtype(model_args)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rows = read_jsonl(script_args.train_file)
    train_dataset = Dataset.from_list(rows)
    trainer = CodeAVSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        views=script_args.views,
        multi_view_mode=script_args.multi_view_mode,
        single_view=script_args.single_view,
        max_prompt_length=script_args.max_prompt_length,
        max_response_tokens=training_args.max_completion_length,
        generation_min_p=script_args.generation_min_p,
        generation_presence_penalty=script_args.presence_penalty,
        generation_repetition_penalty=script_args.generation_repetition_penalty,
        student_enable_thinking=script_args.student_enable_thinking,
        teacher_enable_thinking=script_args.teacher_enable_thinking,
        fixed_teacher=script_args.fixed_teacher,
        execution_timeout_s=script_args.execution_timeout_s,
        unsafe_subprocess_sandbox=script_args.unsafe_subprocess_sandbox,
        avsd_gate_mode=script_args.avsd_gate_mode,
        avsd_gate_alpha=script_args.avsd_gate_alpha,
        avsd_gate_var_coef=script_args.avsd_gate_var_coef,
        avsd_gate_gap_coef=script_args.avsd_gate_gap_coef,
        avsd_sign_threshold=script_args.avsd_sign_threshold,
        avsd_consistency_exp_variance_mode=script_args.avsd_consistency_exp_variance_mode,
        avsd_consistency_exp_var_coef=script_args.avsd_consistency_exp_var_coef,
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


def _resolve_model_dtype(model_args):
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        return _resolve_dtype(model_args.torch_dtype)
    if hasattr(model_args, "dtype") and model_args.dtype is not None:
        return _resolve_dtype(model_args.dtype)
    return torch.bfloat16


def _normalize_model_config(model_args: ModelConfig) -> None:
    targets = getattr(model_args, "lora_target_modules", None)
    if isinstance(targets, str):
        model_args.lora_target_modules = [item.strip() for item in targets.split(",") if item.strip()]
    elif isinstance(targets, list) and len(targets) == 1 and isinstance(targets[0], str) and "," in targets[0]:
        model_args.lora_target_modules = [item.strip() for item in targets[0].split(",") if item.strip()]


def _resolve_dtype(raw_dtype):
    if not isinstance(raw_dtype, str):
        return raw_dtype
    normalized = raw_dtype.lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    if normalized in {"fp32", "float32"}:
        return torch.float32
    if normalized in {"auto", "none"}:
        return "auto"
    raise ValueError("dtype must be one of: bfloat16, float16, float32, auto")


def _resolve_code_aliases(script_args: CodeAVSDScriptArguments, training_args: Any) -> None:
    if script_args.max_response_tokens is not None:
        if script_args.max_response_tokens < 1:
            raise ValueError("--max_response_tokens must be >= 1.")
        training_args.max_completion_length = int(script_args.max_response_tokens)
    if training_args.max_completion_length < 1:
        raise ValueError("--max_completion_length must be >= 1.")
    if training_args.max_length is not None:
        if training_args.max_length <= training_args.max_completion_length:
            raise ValueError("--max_length must be greater than --max_completion_length.")
        script_args.max_prompt_length = int(training_args.max_length) - int(training_args.max_completion_length)

    presence_penalty = (
        script_args.generation_presence_penalty
        if script_args.generation_presence_penalty is not None
        else script_args.presence_penalty
    )
    training_args.presence_penalty = float(presence_penalty)
    training_args.repetition_penalty = float(script_args.generation_repetition_penalty)
    training_args.min_p = float(script_args.generation_min_p)
    training_args.use_vllm = True
    if getattr(training_args, "vllm_mode", "server") == "server":
        training_args.vllm_mode = "colocate"
    training_args.remove_unused_columns = False


def _force_numeric_gradient_accumulation_steps(training_args: Any) -> int:
    raw_steps = getattr(training_args, "gradient_accumulation_steps", 1)
    try:
        steps = int(raw_steps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "--gradient_accumulation_steps must be an integer for code AVSD training; "
            f"got {raw_steps!r}."
        ) from exc
    if steps < 1:
        raise ValueError("--gradient_accumulation_steps must be >= 1.")

    training_args.gradient_accumulation_steps = steps
    os.environ["ACCELERATE_GRADIENT_ACCUMULATION_STEPS"] = str(steps)

    deepspeed_plugin = getattr(training_args, "deepspeed_plugin", None)
    for plugin in _iter_deepspeed_plugins(deepspeed_plugin):
        if hasattr(plugin, "gradient_accumulation_steps"):
            plugin.gradient_accumulation_steps = steps
        hf_ds_config = getattr(plugin, "hf_ds_config", None)
        config = getattr(hf_ds_config, "config", None)
        if isinstance(config, dict):
            config["gradient_accumulation_steps"] = steps
        deepspeed_config = getattr(plugin, "deepspeed_config", None)
        if isinstance(deepspeed_config, dict):
            deepspeed_config["gradient_accumulation_steps"] = steps
    return steps


def _configure_run_metadata(script_args: CodeAVSDScriptArguments, training_args: Any, model_args: ModelConfig) -> None:
    if getattr(training_args, "wandb_project", None):
        os.environ["WANDB_PROJECT"] = training_args.wandb_project

    lr_str = f"{float(training_args.learning_rate):.0e}".replace("e-0", "e-")
    world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
    effective_batch_size = (
        int(training_args.per_device_train_batch_size)
        * int(training_args.gradient_accumulation_steps)
        * world_size
    )
    if script_args.run_config:
        run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        if not str(training_args.output_dir).endswith(script_args.run_config):
            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.rstrip("/").split("/")[-1]
        run_name = f"code_avsd_{model_name}_lr{lr_str}_bs{effective_batch_size}_tok{training_args.max_completion_length}"

    training_args.run_name = run_name
    print(f"\n{'=' * 80}")
    print("CODE AVSD RUN CONFIGURATION")
    print(f"{'=' * 80}")
    print(f"Run Name: {run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Effective Batch Size: {effective_batch_size}")
    print(f"Max Prompt Tokens: {script_args.max_prompt_length}")
    print(f"Max Completion Tokens: {training_args.max_completion_length}")
    print(f"{'=' * 80}\n")


def _iter_deepspeed_plugins(plugin_or_plugins: Any) -> list[Any]:
    if plugin_or_plugins is None:
        return []
    if isinstance(plugin_or_plugins, dict):
        return list(plugin_or_plugins.values())
    return [plugin_or_plugins]


if __name__ == "__main__":
    main()
