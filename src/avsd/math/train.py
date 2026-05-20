import os

from transformers import AutoTokenizer

from trl import (
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
try:
    from trl.experimental.gold import GOLDConfig
except ImportError:
    from avsd.math.trainer import GOLDConfig
from avsd.math.trainer import AVSDTrainer, validate_multi_view_mode_settings, validate_avsd_gate_mode_settings
from dataclasses import dataclass, field
from avsd.math.privileged_views import VIEW_TYPES
from avsd.math.training_datasets import (
    DEFAULT_OPENR1_MATH_SOURCES,
    TRAINING_DATASET_OPENR1_MATH_15K,
    TRAINING_DATASET_OPENTHOUGHT,
    load_training_dataset,
    parse_openr1_math_sources,
    parse_training_dataset_choice,
)


@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with Thinking Machines loss option."""

    use_tinker_loss: bool = field(
        default=False,
        metadata={
            "help": "Use Thinking Machines style on-policy reverse KL loss instead of GKD's full-vocab JSD loss. "
            "This is much more memory efficient (O(1) vs O(vocab_size) per token)."
        },
    )
    student_enable_thinking: bool = field(
        default=False,
        metadata={
            "help": "Pass enable_thinking=True to compatible chat templates for student prompts. "
            "Teacher prompts keep thinking mode enabled regardless of this setting."
        },
    )
    teacher_enable_thinking: bool = field(
        default=True,
        metadata={
            "help": "Pass enable_thinking to compatible chat templates for teacher prompts. "
            "Set false to make Qwen templates insert an empty <think></think> block before teacher-scored outputs."
        },
    )
    fixed_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use the initial policy (step 0) as a fixed teacher. Only works with use_peft=True. "
            "The teacher will use the base model without LoRA adapters, while the student updates."
        },
    )
    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir). If not specified, an automatic name is generated."
        },
    )
    training_dataset: str = field(
        default=TRAINING_DATASET_OPENTHOUGHT,
        metadata={
            "help": "Training dataset selector: openthought | openr1_math_15k. "
            "Defaults to openthought; no setting combines datasets."
        },
    )
    openr1_math_num_samples: int = field(
        default=15_000,
        metadata={
            "help": "Number of filtered OpenR1-Math-220k rows to use when "
            "training_dataset=openr1_math_15k."
        },
    )
    openr1_math_seed: int = field(
        default=42,
        metadata={"help": "Shuffle seed used before selecting the OpenR1-Math-220k slice."},
    )
    openr1_math_sources: str = field(
        default=",".join(DEFAULT_OPENR1_MATH_SOURCES),
        metadata={
            "help": "Comma-separated OpenR1 source labels treated as competition/olympiad-level."
        },
    )
    presence_penalty: float = field(
        default=0.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the generated text so far. "
            "Values > 0 encourage the model to use new tokens, while values < 0 encourage the model to repeat tokens."
        },
    )
    reason_first: bool = field(
        default=False,
        metadata={
            "help": "Let the teacher model first rationalize (generate rationalization explictly) about the given reasoning first then act as teacher."
        },
    )
    top_k_loss: int = field(
        default=0,
        metadata={
            "help": "Restrict the JSD loss to only the top-k tokens of the teacher distribution. Both student and "
            "teacher distributions are renormalized over these k tokens before computing JSD. "
            "Set to 0 (default) to use the full vocabulary."
        },
    )
    jsd_token_clip: float = field(
        default=0.05,
        metadata={
            "help": "Clip the JSD loss for each token to a maximum value. This can improve stability by preventing "
            "extremely high-loss stylistic tokens from dominating the training signal. Set to 0 for no clipping."
        },
    )

    use_ema_teacher: bool = field(
        default=False,
        metadata={
            "help": "Use an exponential moving average (EMA) of student weights as the teacher. "
            "The EMA teacher is a smoothly-lagged version of the student, avoiding the teacher "
            "collapsing to the current policy (dynamic) or staying frozen (fixed_teacher). "
            "Mutually exclusive with fixed_teacher."
        },
    )
    ema_decay: float = field(
        default=0.999,
        metadata={
            "help": "EMA decay factor. Higher values make the teacher change more slowly. "
            "Typical range: 0.99–0.9999. Only used when use_ema_teacher=True."
        },
    )
    multi_view_mode: str = field(
        default="single",
        metadata={"help": "Teacher-view routing mode: single | consensus | arithmetic | avsd."},
    )
    single_view_pi: str = field(
        default="full_solution",
        metadata={"help": "Privileged-information view for legacy single-view self-distillation."},
    )
    pi_views: str = field(
        default="full_solution,partial_solution,answer_only",
        metadata={
            "help": "Comma-separated ordered privileged-information views for AVSD."
        },
    )
    partial_solution_ratio: float = field(
        default=0.5,
        metadata={"help": "Fraction of the privileged solution prefix to keep for partial_solution."},
    )
    multi_view_teacher_topk: int = field(
        default=64,
        metadata={"help": "Per-view top-k used to build the multi-view candidate token union."},
    )
    view_weight_mode: str = field(
        default="uniform",
        metadata={"help": "Multi-view weighting mode: uniform | agreement_centrality."},
    )
    view_agreement_eta: float = field(
        default=5.0,
        metadata={"help": "Softmax sharpness for agreement-centrality view weighting."},
    )
    avsd_gate_alpha: float = field(
        default=6.0,
        metadata={"help": "Alpha coefficient for the AVSD gate."},
    )
    avsd_gate_var_coef: float = field(
        default=1.0,
        metadata={"help": "Variance penalty coefficient for the AVSD gate."},
    )
    avsd_gate_gap_coef: float = field(
        default=1.0,
        metadata={"help": "Jensen-gap penalty coefficient for the AVSD gate."},
    )
    avsd_sign_threshold: float = field(
        default=0.8,
        metadata={"help": "Sign-consistency threshold for the AVSD gate."},
    )
    avsd_gate_mode: str = field(
        default="sigmoid",
        metadata={"help": "AVSD gate mode: sigmoid | consistency_exp | avsd."},
    )
    avsd_consistency_exp_variance_mode: str = field(
        default="on",
        metadata={"help": "Whether the consistency_exp AVSD gate includes the variance term: on | off."},
    )
    avsd_consistency_exp_var_coef: float = field(
        default=1.0,
        metadata={"help": "Variance coefficient gamma for the consistency_exp AVSD gate."},
    )
    use_epistemic_preservation: bool = field(
        default=False,
        metadata={"help": "Add the optional epistemic-preservation regularizer in AVSD mode."},
    )
    epistemic_tau: float = field(
        default=0.01,
        metadata={"help": "Strength of the epistemic-preservation regularizer."},
    )


def _normalize_pi_views(raw_pi_views: str) -> tuple[str, ...]:
    normalized = []
    seen = set()
    for view in (item.strip() for item in raw_pi_views.split(",")):
        if not view:
            continue
        if view not in VIEW_TYPES:
            raise ValueError(
                f"Unknown pi view '{view}'. Supported views: {', '.join(VIEW_TYPES)}"
            )
        if view not in seen:
            seen.add(view)
            normalized.append(view)
    if not normalized:
        raise ValueError("pi_views must contain at least one valid privileged-information view.")
    return tuple(normalized)


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GOLDConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    training_dataset = parse_training_dataset_choice(script_args.training_dataset)
    openr1_math_sources = (
        parse_openr1_math_sources(script_args.openr1_math_sources)
        if training_dataset == TRAINING_DATASET_OPENR1_MATH_15K
        else DEFAULT_OPENR1_MATH_SOURCES
    )
    script_args.pi_views = _normalize_pi_views(script_args.pi_views)
    if script_args.single_view_pi not in VIEW_TYPES:
        raise ValueError(
            f"single_view_pi must be one of: {', '.join(VIEW_TYPES)}"
        )

    validate_multi_view_mode_settings(
        multi_view_mode=script_args.multi_view_mode,
        use_epistemic_preservation=script_args.use_epistemic_preservation,
    )
    if script_args.view_weight_mode not in {"uniform", "agreement_centrality"}:
        raise ValueError("view_weight_mode must be one of: uniform, agreement_centrality")
    validate_avsd_gate_mode_settings(
        multi_view_mode=script_args.multi_view_mode,
        view_weight_mode=script_args.view_weight_mode,
        avsd_gate_mode=script_args.avsd_gate_mode,
        avsd_consistency_exp_variance_mode=script_args.avsd_consistency_exp_variance_mode,
        use_tinker_loss=script_args.use_tinker_loss and script_args.multi_view_mode != "single",
    )
    if not 0.0 < script_args.partial_solution_ratio < 1.0:
        raise ValueError("partial_solution_ratio must be strictly between 0 and 1.")
    tinker_multi_view = script_args.use_tinker_loss and script_args.multi_view_mode != "single"
    if script_args.multi_view_teacher_topk <= 0 and not tinker_multi_view:
        raise ValueError("multi_view_teacher_topk must be positive.")
    if script_args.multi_view_mode != "single" and script_args.reason_first:
        raise ValueError("reason_first is only supported for single-view self-distillation in v1")
    if script_args.multi_view_mode != "single" and script_args.top_k_loss > 0:
        raise ValueError("top_k_loss is only supported for single-view self-distillation in v1")

    training_args.report_to = []
    training_args.log_completions = False

    ################
    # Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-4 -> "2e-4" or 0.0002 -> "2e-4")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_run_config = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_run_config = (
            f"avsd_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"tok{training_args.max_completion_length}"
        )

        if script_args.fixed_teacher:
            full_run_config += "_fixteach"

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"Run Name: {full_run_config}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"{'='*80}\n")

    # Validate fixed_teacher argument
    if script_args.fixed_teacher and not model_args.use_peft:
        raise ValueError(
            "fixed_teacher=True requires use_peft=True. As the fixed teacher is implemented by disabling LoRA adapters."
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype - handle both old torch_dtype and new dtype attributes
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        # Passing None would not be treated the same as omitting the argument, so we include it only when valid.
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    # No separate teacher model needed - we use the same model with privileged info

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # Load the math dataset with ground truth solutions
    ################
    # Training
    ################
    # Add presence_penalty to training_args so it can be accessed in the trainer
    training_args.presence_penalty = script_args.presence_penalty

    train_dataset, dataset_metadata = load_training_dataset(
        training_dataset=training_dataset,
        openr1_math_num_samples=script_args.openr1_math_num_samples,
        openr1_math_seed=script_args.openr1_math_seed,
        openr1_math_sources=openr1_math_sources,
    )
    print(f"\n{'='*80}")
    print("TRAINING DATASET")
    print(f"Selected dataset: {dataset_metadata.training_dataset}")
    print(f"OpenThought rows: {dataset_metadata.openthought_num_rows}")
    print(f"OpenR1 rows selected: {dataset_metadata.openr1_math_num_selected}")
    print(f"OpenR1 eligible rows: {dataset_metadata.openr1_math_num_eligible}")
    print(f"OpenR1 sources: {', '.join(dataset_metadata.openr1_math_sources)}")
    print(f"Total training rows: {dataset_metadata.train_num_rows}")
    print(f"{'='*80}\n")

    trainer = AVSDTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        use_thinking_machines_loss=script_args.use_tinker_loss,
        student_enable_thinking=script_args.student_enable_thinking,
        teacher_enable_thinking=script_args.teacher_enable_thinking,
        fixed_teacher=script_args.fixed_teacher,
        reason_first=script_args.reason_first,
        top_k_loss=script_args.top_k_loss if script_args.top_k_loss > 0 else None,
        jsd_token_clip=script_args.jsd_token_clip if script_args.jsd_token_clip > 0 else None,
        use_ema_teacher=script_args.use_ema_teacher,
        ema_decay=script_args.ema_decay,
        multi_view_mode=script_args.multi_view_mode,
        single_view_pi=script_args.single_view_pi,
        pi_views=script_args.pi_views,
        partial_solution_ratio=script_args.partial_solution_ratio,
        multi_view_teacher_topk=script_args.multi_view_teacher_topk,
        view_weight_mode=script_args.view_weight_mode,
        view_agreement_eta=script_args.view_agreement_eta,
        avsd_gate_alpha=script_args.avsd_gate_alpha,
        avsd_gate_var_coef=script_args.avsd_gate_var_coef,
        avsd_gate_gap_coef=script_args.avsd_gate_gap_coef,
        avsd_sign_threshold=script_args.avsd_sign_threshold,
        avsd_gate_mode=script_args.avsd_gate_mode,
        avsd_consistency_exp_variance_mode=script_args.avsd_consistency_exp_variance_mode,
        avsd_consistency_exp_var_coef=script_args.avsd_consistency_exp_var_coef,
        use_epistemic_preservation=script_args.use_epistemic_preservation,
        epistemic_tau=script_args.epistemic_tau,
    )

    trainer.train()

    trainer.save_model(training_args.output_dir)
