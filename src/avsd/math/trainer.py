import os
import random
import textwrap
import warnings
from collections import defaultdict, deque
from collections.abc import Callable
from contextlib import contextmanager, nullcontext
from typing import Any, Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from accelerate.utils import DistributedType, broadcast_object_list, gather_object, is_peft_model
from datasets import Dataset, IterableDataset
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers.data.data_collator import DataCollator
from transformers.feature_extraction_utils import FeatureExtractionMixin
from transformers.generation.configuration_utils import GenerationConfig
from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import EvalPrediction
from transformers.utils import (
    is_flash_attn_2_available,
    is_liger_kernel_available,
    is_peft_available,
    is_rich_available,
)

try:
    from trl.data_utils import is_conversational, maybe_convert_to_chatml, pack_dataset, truncate_dataset
except ImportError:
    from trl.data_utils import is_conversational

    def maybe_convert_to_chatml(example):
        return example

    def pack_dataset(dataset, *args, **kwargs):
        return dataset

    def truncate_dataset(dataset, *args, **kwargs):
        return dataset
try:
    from trl.extras.profiling import profiling_decorator
except ImportError:
    def profiling_decorator(func):
        return func


try:
    from trl.extras.vllm_client import VLLMClient
except ImportError:
    class VLLMClient:  # pragma: no cover - compatibility fallback for older TRL installs
        def __init__(self, *args, **kwargs):
            raise ImportError("VLLMClient is unavailable in the installed TRL version.")
try:
    from trl.import_utils import is_vllm_available
except ImportError:
    def is_vllm_available():
        return False


try:
    from trl.models import prepare_deepspeed
except ImportError:
    def prepare_deepspeed(model, *args, **kwargs):
        return model

from trl.models.utils import unwrap_model_for_generation
from trl.trainer.sft_trainer import SFTTrainer
try:
    from trl.trainer.utils import (
        DataCollatorForChatML,
        disable_dropout_in_model,
        empty_cache,
        ensure_master_addr_port,
        pad,
    )
except ImportError:
    from trl.trainer.utils import DataCollatorForChatML, disable_dropout_in_model, empty_cache, pad

    def ensure_master_addr_port():
        return None


try:
    from trl.experimental.gold.gold_config import GOLDConfig
except ImportError:
    class GOLDConfig:  # pragma: no cover - compatibility fallback for local smoke tests
        pass
from avsd.math.data_collator import SelfDistillationDataCollator
from avsd.common.multiview_distill import (
    build_arithmetic_target,
    build_candidate_union,
    build_consensus_target,
    build_uniform_sampled_tinker_target,
    build_avsd_target,
    compute_view_weights,
    epistemic_preservation_loss,
    generalized_jsd_from_log_probs,
    masked_log_softmax,
    sampled_epistemic_preservation_loss,
    sampled_token_tinker_loss_from_log_probs,
)
from avsd.math.privileged_views import VIEW_TYPES


MULTI_VIEW_MODES = frozenset({"single", "consensus", "arithmetic", "avsd"})


if is_peft_available():
    from peft import PeftConfig

if is_vllm_available():
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import GuidedDecodingParams

if is_rich_available():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text


def _normalize_pi_views(pi_views: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(pi_views, str):
        raw_views = [view.strip() for view in pi_views.split(",")]
    else:
        raw_views = [str(view).strip() for view in pi_views]

    normalized = []
    seen = set()
    for view in raw_views:
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


def force_skip_sft_dataset_preparation(args) -> None:
    args.dataset_kwargs = dict(args.dataset_kwargs or {})
    args.dataset_kwargs["skip_prepare_dataset"] = True


def validate_multi_view_mode_settings(*, multi_view_mode: str, use_epistemic_preservation: bool = False):
    if multi_view_mode not in MULTI_VIEW_MODES:
        raise ValueError("multi_view_mode must be one of: single, consensus, arithmetic, avsd")
    if use_epistemic_preservation and multi_view_mode != "avsd":
        raise ValueError("epistemic preservation is only defined for multi_view_mode=avsd")


def validate_avsd_gate_mode_settings(
    *,
    multi_view_mode: str,
    view_weight_mode: str,
    avsd_gate_mode: str,
    avsd_consistency_exp_variance_mode: str,
    use_tinker_loss: bool = False,
):
    if avsd_gate_mode not in {"sigmoid", "consistency_exp", "avsd"}:
        raise ValueError("avsd_gate_mode must be one of: sigmoid, consistency_exp, avsd")
    if avsd_consistency_exp_variance_mode not in {"on", "off"}:
        raise ValueError("avsd_consistency_exp_variance_mode must be one of: on, off")
    if avsd_gate_mode == "avsd":
        if multi_view_mode != "avsd":
            raise ValueError("avsd gate is only defined for multi_view_mode=avsd")
        if view_weight_mode != "uniform" and not use_tinker_loss:
            raise ValueError("avsd gate requires view_weight_mode=uniform in v1")


class EMAUpdateCallback(TrainerCallback):
    """Update EMA teacher weights after each optimizer step."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        # Only update when the optimizer actually stepped (end of a gradient accumulation cycle)
        if self.trainer.use_ema_teacher and self.trainer.accelerator.sync_gradients:
            self.trainer._update_ema()


class GOLDVLLMSyncCallback(TrainerCallback):
    """Sync the model weights to vLLM after training steps when it's safe to do so."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Sync weights after training step when DeepSpeed is stable."""
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
        ):
            # Check if this is a step where gradients are synchronized
            # This happens at the end of gradient accumulation cycles
            if (
                hasattr(self.trainer.accelerator, "sync_gradients")
                and self.trainer.accelerator.sync_gradients
            ):
                self.trainer._move_model_to_vllm()
                self.trainer._last_vllm_sync_step = state.global_step


class AVSDTrainer(SFTTrainer):
    _tag_names = ["trl", "avsd"]
    _name = "AVSD"

    def __init__(
        self,
        model: PreTrainedModel | nn.Module | str | None = None,
        args: GOLDConfig | None = None,
        data_collator: DataCollator | None = None,  # type: ignore
        train_dataset: Dataset | None = None,
        eval_dataset: Dataset | dict[str, Dataset] | None = None,
        processing_class: (
            PreTrainedTokenizerBase | BaseImageProcessor | FeatureExtractionMixin | ProcessorMixin | None
        ) = None,
        compute_metrics: Callable[[EvalPrediction], dict] | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
        peft_config: Optional["PeftConfig"] = None,
        use_thinking_machines_loss: bool = False,
        student_enable_thinking: bool = False,
        teacher_enable_thinking: bool = True,
        fixed_teacher: bool = False,
        reason_first: bool = False,
        top_k_loss: int | None = None,
        jsd_token_clip: float | None = None,
        use_ema_teacher: bool = False,
        ema_decay: float = 0.999,
        multi_view_mode: str = "single",
        single_view_pi: str = "full_solution",
        pi_views: tuple[str, ...] = ("full_solution", "partial_solution", "answer_only"),
        partial_solution_ratio: float = 0.5,
        multi_view_teacher_topk: int = 64,
        view_weight_mode: str = "uniform",
        view_agreement_eta: float = 5.0,
        avsd_gate_alpha: float = 6.0,
        avsd_gate_var_coef: float = 1.0,
        avsd_gate_gap_coef: float = 1.0,
        avsd_sign_threshold: float = 0.8,
        avsd_gate_mode: str = "sigmoid",
        avsd_consistency_exp_variance_mode: str = "on",
        avsd_consistency_exp_var_coef: float = 1.0,
        use_epistemic_preservation: bool = False,
        epistemic_tau: float = 0.01,
    ):
        normalized_pi_views = _normalize_pi_views(pi_views)
        force_skip_sft_dataset_preparation(args)
        self.model_name_or_path = model if isinstance(model, str) else model.config._name_or_path
        self.model_revision = getattr(args, "student_model_revision", None)
        if isinstance(model, str) and self.model_revision is not None:
            args.model_init_kwargs = args.model_init_kwargs or {}
            args.model_init_kwargs.setdefault("revision", self.model_revision)

        # Custom data collator for self-distillation
        if data_collator is None:
            data_collator = SelfDistillationDataCollator(
                tokenizer=processing_class,
                max_length=args.max_length,
                reason_first=reason_first,
                multi_view_mode=multi_view_mode,
                single_view_pi=single_view_pi,
                pi_views=normalized_pi_views,
                partial_solution_ratio=partial_solution_ratio,
                student_enable_thinking=student_enable_thinking,
                teacher_enable_thinking=teacher_enable_thinking,
            )

        super().__init__(
            model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            peft_config=peft_config,
        )

        if args.disable_dropout:
            disable_dropout_in_model(self.model)

        self.lmbda = args.lmbda
        self.beta = args.beta
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.seq_kd = args.seq_kd
        self.use_thinking_machines_loss = use_thinking_machines_loss
        self.student_enable_thinking = student_enable_thinking
        self.teacher_enable_thinking = teacher_enable_thinking
        self.fixed_teacher = fixed_teacher
        self.reason_first = reason_first
        self.top_k_loss = top_k_loss
        self.jsd_token_clip = jsd_token_clip
        self.use_ema_teacher = use_ema_teacher
        self.ema_decay = ema_decay
        self.multi_view_mode = multi_view_mode
        self.single_view_pi = single_view_pi
        self.pi_views = normalized_pi_views
        self.partial_solution_ratio = partial_solution_ratio
        self.multi_view_teacher_topk = multi_view_teacher_topk
        self.view_weight_mode = view_weight_mode
        self.view_agreement_eta = view_agreement_eta
        self.avsd_gate_alpha = avsd_gate_alpha
        self.avsd_gate_var_coef = avsd_gate_var_coef
        self.avsd_gate_gap_coef = avsd_gate_gap_coef
        self.avsd_sign_threshold = avsd_sign_threshold
        self.avsd_gate_mode = avsd_gate_mode
        self.avsd_consistency_exp_variance_mode = avsd_consistency_exp_variance_mode
        self.avsd_consistency_exp_var_coef = avsd_consistency_exp_var_coef
        self.use_epistemic_preservation = use_epistemic_preservation
        self.epistemic_tau = epistemic_tau
        self._ema_params = None  # lazily initialized on first optimizer step

        validate_multi_view_mode_settings(
            multi_view_mode=self.multi_view_mode,
            use_epistemic_preservation=self.use_epistemic_preservation,
        )
        if self.single_view_pi not in VIEW_TYPES:
            raise ValueError(f"single_view_pi must be one of: {', '.join(VIEW_TYPES)}")
        if self.view_weight_mode not in {"uniform", "agreement_centrality"}:
            raise ValueError("view_weight_mode must be one of: uniform, agreement_centrality")
        validate_avsd_gate_mode_settings(
            multi_view_mode=self.multi_view_mode,
            view_weight_mode=self.view_weight_mode,
            avsd_gate_mode=self.avsd_gate_mode,
            avsd_consistency_exp_variance_mode=self.avsd_consistency_exp_variance_mode,
            use_tinker_loss=self.use_thinking_machines_loss and self.multi_view_mode != "single",
        )
        if not 0.0 < self.partial_solution_ratio < 1.0:
            raise ValueError("partial_solution_ratio must be strictly between 0 and 1.")
        tinker_multi_view = self.use_thinking_machines_loss and self.multi_view_mode != "single"
        if self.multi_view_teacher_topk <= 0 and not tinker_multi_view:
            raise ValueError("multi_view_teacher_topk must be positive.")
        if self.multi_view_mode != "single" and self.reason_first:
            raise ValueError("reason_first is only supported for single-view self-distillation in v1")
        if self.multi_view_mode != "single" and self.top_k_loss is not None:
            raise ValueError("top_k_loss is only supported for single-view self-distillation in v1")

        # Validate fixed_teacher option
        if self.fixed_teacher and peft_config is None:
            raise ValueError(
                "fixed_teacher=True requires a PEFT config (use_peft=True). "
                "The fixed teacher is implemented by disabling LoRA adapters during teacher forward passes."
            )

        if self.use_ema_teacher and self.fixed_teacher:
            raise ValueError(
                "use_ema_teacher=True and fixed_teacher=True are mutually exclusive teacher strategies."
            )

        if self.use_ema_teacher:
            self.add_callback(EMAUpdateCallback(self))
            print(f"\n{'='*80}")
            print("EMA TEACHER MODE ENABLED")
            print(f"EMA decay: {self.ema_decay}")
            print("Teacher is an exponential moving average of the student weights.")
            print("EMA parameters are initialized on the first optimizer step.")
            print(f"{'='*80}\n")

        if self.fixed_teacher:
            print(f"\n{'='*80}")
            print("FIXED TEACHER MODE ENABLED")
            print("Teacher will use the initial policy (base model without LoRA adapters)")
            print("Student will update with LoRA adapters")
            print(f"{'='*80}\n")

        if self.reason_first:
            print(f"\n{'='*80}")
            print("REASON FIRST MODE ENABLED")
            print("Teacher will first reason about the privileged solution, then evaluate student's response")
            print(f"{'='*80}\n")

        if self.multi_view_mode != "single":
            print(f"\n{'='*80}")
            print("AVSD MODE ENABLED")
            print(f"Mode: {self.multi_view_mode}")
            print(f"Views: {self.pi_views}")
            if self.use_thinking_machines_loss:
                print("TINKER MULTI-VIEW LOSS ENABLED")
                print("View aggregation: uniform sampled-token consensus/arithmetic teachers")
                print("Teacher top-k: skipped")
                print(
                    "beta, jsd_token_clip, multi_view_teacher_topk, view_weight_mode, "
                    "and view_agreement_eta are ignored by this loss."
                )
            else:
                if self.multi_view_mode == "arithmetic":
                    print("View aggregation: uniform arithmetic teacher")
                    print("Weighting: uniform (view_weight_mode is ignored)")
                else:
                    print(f"Weighting: {self.view_weight_mode}")
                print(f"Teacher top-k: {self.multi_view_teacher_topk}")
            if self.multi_view_mode == "avsd":
                print(f"AVSD gate mode: {self.avsd_gate_mode}")
                if self.avsd_gate_mode == "consistency_exp":
                    print(
                        "consistency_exp variance term: "
                        f"{self.avsd_consistency_exp_variance_mode} (gamma={self.avsd_consistency_exp_var_coef})"
                    )
                elif self.avsd_gate_mode == "avsd":
                    print("AVSD uses exact uniform averaging and ignores legacy gate hyperparameters.")
            print(f"{'='*80}\n")

        # Track per-step loss statistics for on/off-policy batches (used in logging)
        self._on_policy_loss_total = 0.0
        self._off_policy_loss_total = 0.0
        self._on_policy_step_equiv = 0.0
        self._off_policy_step_equiv = 0.0

        self.use_transformers_paged = args.use_transformers_paged or False

        # Track generation outputs for saving
        self._generation_outputs_buffer = []
        self._generation_save_frequency = 5  # Save every 5 steps

        self.generation_config = GenerationConfig(
            max_new_tokens=args.max_completion_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Generation config for reasoning phase (when reason_first=True)
        max_reasoning_length = getattr(args, "max_reasoning_length", 4096)
        self.reasoning_generation_config = GenerationConfig(
            max_new_tokens=max_reasoning_length,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=True,
            top_k=args.top_k,
            pad_token_id=self.processing_class.pad_token_id,
            use_cache=True,
        )
        if (
            hasattr(self.model.generation_config, "eos_token_id")
            and self.model.generation_config.eos_token_id is not None
        ):
            self.reasoning_generation_config.eos_token_id = self.model.generation_config.eos_token_id

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.log_completion_steps = args.log_completions_steps
        self.num_completions_to_print = args.num_completions_to_print
        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = self.accelerator.num_processes * args.per_device_train_batch_size * args.steps_per_generation
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
            "advantages": deque(maxlen=maxlen),
        }

        self.use_vllm = args.use_vllm
        if self.use_vllm:
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and use_vllm is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )
            self.vllm_mode = args.vllm_mode
            self.vllm_tensor_parallel_size = args.vllm_tensor_parallel_size
            self.vllm_gpu_memory_utilization = args.vllm_gpu_memory_utilization
            self.vllm_enable_sleep_mode = args.vllm_enable_sleep_mode
            if self.vllm_mode == "server":
                if self.accelerator.is_main_process:
                    self.vllm_client = VLLMClient(
                        host=args.vllm_server_host,
                        server_port=args.vllm_server_port,
                        connection_timeout=args.vllm_server_timeout,
                    )
                    self.vllm_client.init_communicator()
            elif self.vllm_mode == "colocate":
                student_model_name_or_path = self.model_name_or_path

                # Make sure tensor_parallel_size divides world size evenly
                if not self.accelerator.num_processes % self.vllm_tensor_parallel_size == 0:
                    raise ValueError(
                        f"vllm_tensor_parallel_size ({self.vllm_tensor_parallel_size}) must divide world size "
                        f"({self.accelerator.num_processes}) evenly."
                    )

                if self.vllm_tensor_parallel_size > 1:
                    # Create subgroups of ranks for TP
                    self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(
                        [
                            list(
                                range(
                                    i * self.vllm_tensor_parallel_size,
                                    (i + 1) * self.vllm_tensor_parallel_size,
                                )
                            )
                            for i in range(self.accelerator.num_processes // self.vllm_tensor_parallel_size)
                        ]
                    )

                # vLLM requires the environment variables to be set for distributed training.
                os.environ["RANK"] = str(self.accelerator.process_index)
                os.environ["LOCAL_RANK"] = str(self.accelerator.local_process_index)
                os.environ["WORLD_SIZE"] = str(self.accelerator.num_processes)
                ensure_master_addr_port()

                self.vllm_engine = LLM(
                    model=student_model_name_or_path,
                    revision=self.model_revision,
                    tensor_parallel_size=self.vllm_tensor_parallel_size,
                    gpu_memory_utilization=self.vllm_gpu_memory_utilization,
                    max_num_seqs=self.args.per_device_train_batch_size
                    * self.args.gradient_accumulation_steps,
                    max_model_len=args.max_length,
                    distributed_executor_backend="external_launcher",
                    # Feed identical seed for tp groups to ensure sampling results are the same across workers
                    seed=self.accelerator.process_index // self.vllm_tensor_parallel_size,
                    enable_sleep_mode=self.vllm_enable_sleep_mode,
                )

                if self.vllm_enable_sleep_mode:
                    self.vllm_engine.sleep(level=2)

                # When using vLLM, the main process is responsible for loading the model weights. This can cause process
                # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
                # synchronize all processes after vLLM has been fully initialized.
                self.accelerator.wait_for_everyone()
            else:
                raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")
            self.vllm_guided_decoding_regex = args.vllm_guided_decoding_regex
            self.vllm_sync_frequency = args.vllm_sync_frequency
            self._last_vllm_sync_step = -1

            self.add_callback(GOLDVLLMSyncCallback(self))

    def _set_signature_columns_if_needed(self):
        super()._set_signature_columns_if_needed()
        required_columns = [
            "problem",
            "solution",
        ]
        if self._signature_columns is None:
            self._signature_columns = required_columns
        else:
            for column in required_columns:
                if column not in self._signature_columns:
                    self._signature_columns.append(column)

    @staticmethod
    def generalized_jsd_loss(
        student_logits,
        teacher_logits,
        labels=None,
        beta=0.5,
        temperature=1.0,
        reduction="batchmean",
        logits_are_probs=False,
        top_k=None,
        token_clip=None,
    ):
        """
        Compute the generalized Jensen-Shannon Divergence loss for knowledge distillation using F.kl_div. See Eq. (1)
        of https://huggingface.co/papers/2306.13649 for the definition.

        Args:
            student_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            teacher_logits:
                Tensor of shape (batch_size, sequence_length, vocab_size)
            labels:
                Tensor of shape (batch_size, sequence_length) with -100 for padding tokens to ignore when computing
                loss
            beta:
                Interpolation coefficient between 0 and 1 (default: 0.5)
            temperature:
                Softmax temperature (default: 1.0)
            reduction:
                Specifies the reduction to apply to the output (default: 'batchmean')
            top_k:
                If set, restricts the loss to only the top-k tokens of the teacher distribution. Both student and
                teacher distributions are renormalized over these k tokens before computing JSD. This reduces memory
                and focuses distillation on the teacher's most probable tokens. (default: None = full vocabulary)
            token_clip:
                if set, clips per-token divergence values to this maximum before reduction. Prevents style tokens from dominating the gradient signal over math tokens.

        Returns:
            loss: Scalar tensor with the generalized JSD loss
        """

        if logits_are_probs:
            student_log_probs = torch.log(student_logits.clamp_min(1e-8))
            teacher_log_probs = torch.log(teacher_logits.clamp_min(1e-8))
        else:
            # Apply temperature scaling to logits before computing probabilities
            student_logits = student_logits / temperature
            teacher_logits = teacher_logits / temperature

            if top_k is not None and top_k > 0:
                # Restrict to top-k tokens of the teacher distribution and renormalize.
                # Shape: [batch, seq_len, top_k]
                _, top_k_indices = torch.topk(teacher_logits, k=top_k, dim=-1)
                student_logits = torch.gather(student_logits, dim=-1, index=top_k_indices)
                teacher_logits = torch.gather(teacher_logits, dim=-1, index=top_k_indices)

            # Compute log probabilities for student and probabilities for teacher
            student_log_probs = F.log_softmax(student_logits, dim=-1)
            teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)

        if beta == 0:
            jsd = F.kl_div(student_log_probs, teacher_log_probs, reduction="none", log_target=True)
        elif beta == 1:
            jsd = F.kl_div(teacher_log_probs, student_log_probs, reduction="none", log_target=True)
        else:
            # Compute the log of the mixture distribution
            # log(a + b) = log(exp(log(a)) + exp(log(b))) -> for mixture
            beta = torch.tensor(beta, dtype=student_log_probs.dtype, device=student_log_probs.device)
            mixture_log_probs = torch.logsumexp(
                torch.stack([student_log_probs + torch.log1p(-beta), teacher_log_probs + torch.log(beta)]),
                dim=0,
            )

            # Compute KL divergences using F.kl_div
            # PyTorch differs from the standard mathematical definition, so the order of the probability distributions is swapped compared to that defined in the paper.
            kl_teacher = F.kl_div(mixture_log_probs, teacher_log_probs, reduction="none", log_target=True)
            kl_student = F.kl_div(mixture_log_probs, student_log_probs, reduction="none", log_target=True)

            # Compute the Generalized Jensen-Shannon Divergence
            jsd = beta * kl_teacher + (1 - beta) * kl_student

        # Per-token clipping: cap each token's divergence value
        if token_clip is not None:
            jsd = jsd.clamp(max=token_clip)

        # Masking
        if labels is not None:
            mask = labels != -100
            jsd = jsd[mask]

        # Apply reduction
        if reduction == "batchmean":
            return jsd.sum() / mask.sum() if labels is not None else jsd.sum() / jsd.size(0)
        elif reduction == "sum":
            return jsd.sum()
        elif reduction == "mean":
            return jsd.mean()
        else:
            return jsd

    def _update_ema(self):
        """Update EMA parameters after an optimizer step.

        On the very first call this lazily initializes the EMA state as an exact copy of the
        current (trainable) model parameters, then returns without applying a decay step.
        Subsequent calls apply: ema = decay * ema + (1 - decay) * student.

        Only trainable parameters are tracked (i.e. LoRA adapter weights for PEFT models,
        or all parameters for full fine-tuning).

        ZeRO-3 note: with ZeRO-3 each rank only holds a shard of every parameter.
        We use `deepspeed.zero.GatheredParameters` (read-only, modifier_rank=None) so that
        every rank sees the full parameter tensor when snapshotting / updating the EMA.
        The EMA tensors are therefore full-sized copies, which is also required by
        `_ema_teacher_context` when it swaps the gathered student weights with EMA values.
        """
        decay = self.ema_decay
        unwrapped = self.accelerator.unwrap_model(self.model)

        # Detect ZeRO-3 (same pattern used elsewhere in this file)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            trainable = [(name, param) for name, param in unwrapped.named_parameters() if param.requires_grad]
            params_list = [p for _, p in trainable]

            # modifier_rank=None → read-only gather; original partitions are restored on exit.
            with deepspeed.zero.GatheredParameters(params_list):
                if self._ema_params is None:
                    self._ema_params = {name: param.data.clone().detach() for name, param in trainable}
                    n_tensors = len(self._ema_params)
                    n_params = sum(p.numel() for p in self._ema_params.values())
                    print(
                        f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                        f"(decay={decay})"
                    )
                    return  # first call = initialization only, no decay update

                for name, param in trainable:
                    if name not in self._ema_params:
                        continue
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    ema.mul_(decay).add_(param.data, alpha=1.0 - decay)
        else:
            if self._ema_params is None:
                # Lazy init: snapshot the current weights as the initial EMA state.
                self._ema_params = {
                    name: param.data.clone().detach()
                    for name, param in unwrapped.named_parameters()
                    if param.requires_grad
                }
                n_tensors = len(self._ema_params)
                n_params = sum(p.numel() for p in self._ema_params.values())
                print(
                    f"\nEMA teacher initialized: {n_tensors} tensors, {n_params:,} parameters "
                    f"(decay={decay})"
                )
                return  # first call = initialization only, no decay update

            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                # Move EMA buffer to the same device as the live param (handles multi-GPU setups)
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                ema.mul_(decay).add_(param.data, alpha=1.0 - decay)

    @contextmanager
    def _ema_teacher_context(self, model):
        """Context manager that temporarily loads EMA weights for the teacher forward pass.

        Swaps `param.data` of every tracked (trainable) parameter with its EMA counterpart,
        runs the body (teacher forward), then restores the student weights unconditionally.
        Safe to use inside `torch.no_grad()`.  If EMA has not been initialized yet (step 0),
        this is a no-op and the current student weights are used instead.

        ZeRO-3 note: direct `param.data` assignment bypasses ZeRO-3's shard lifecycle and
        corrupts its internal state, causing size-mismatch errors during gradient-checkpoint
        recomputation.  When ZeRO-3 is active we therefore wrap the swap inside
        `deepspeed.zero.GatheredParameters` so the parameters are fully materialised on every
        rank before we touch them, and ZeRO-3 re-partitions cleanly when the context exits.
        """
        if self._ema_params is None:
            yield  # EMA not yet initialized; fall back to current weights
            return

        unwrapped = self.accelerator.unwrap_model(model)

        # Detect ZeRO-3 (same pattern used elsewhere in this file)
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3

        if zero_stage_3:
            import deepspeed

            name_to_param = {
                name: param
                for name, param in unwrapped.named_parameters()
                if param.requires_grad and name in self._ema_params
            }
            params_list = list(name_to_param.values())

            # modifier_rank=0 causes ZeRO-3 to re-partition from rank-0's param.data on exit,
            # which will be the restored student weights.
            with deepspeed.zero.GatheredParameters(params_list, modifier_rank=0):
                saved = {}
                for name, param in name_to_param.items():
                    ema = self._ema_params[name]
                    if ema.device != param.data.device:
                        ema = ema.to(param.data.device)
                        self._ema_params[name] = ema
                    saved[name] = param.data.clone()
                    param.data.copy_(ema)
                try:
                    yield
                finally:
                    for name, param in name_to_param.items():
                        if name in saved:
                            param.data.copy_(saved[name])
        else:
            saved = {}
            for name, param in unwrapped.named_parameters():
                if not param.requires_grad or name not in self._ema_params:
                    continue
                ema = self._ema_params[name]
                if ema.device != param.data.device:
                    ema = ema.to(param.data.device)
                    self._ema_params[name] = ema
                saved[name] = param.data
                param.data = ema
            try:
                yield
            finally:
                for name, param in unwrapped.named_parameters():
                    if name in saved:
                        param.data = saved[name]

    def _teacher_context(self, model):
        if self.use_ema_teacher:
            return self._ema_teacher_context(model)
        if self.fixed_teacher and is_peft_model(model):
            return self.accelerator.unwrap_model(model).disable_adapter()
        return nullcontext()

    def _append_metric(self, mode: str, key: str, value):
        if value is None:
            return
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return
            value = value.detach().float().mean().item()
        self._metrics[mode][key].append(value)

    def _masked_mean(self, values: torch.Tensor, mask: torch.Tensor):
        mask = mask.to(dtype=torch.bool)
        if not torch.any(mask):
            return None
        return values[mask].float().mean()

    def _teacher_view_topk_pass(self, model, inputs):
        topk_by_view = []
        for view in self.pi_views:
            with torch.no_grad(), self._teacher_context(model):
                outputs = model(
                    input_ids=inputs[f"teacher_{view}_input_ids"],
                    attention_mask=inputs[f"teacher_{view}_attention_mask"],
                )
                prompt_len = inputs[f"teacher_{view}_prompt_length"]
                teacher_logits = outputs.logits[:, prompt_len - 1 : -1, :]
                topk_k = min(self.multi_view_teacher_topk, teacher_logits.shape[-1])
                topk_indices = torch.topk(teacher_logits / self.temperature, k=topk_k, dim=-1).indices
                topk_by_view.append(topk_indices)
                del outputs, teacher_logits
                empty_cache()
        return topk_by_view

    def _teacher_view_candidate_pass(self, model, inputs, candidate_indices, candidate_mask):
        view_log_probs = []
        for view in self.pi_views:
            with torch.no_grad(), self._teacher_context(model):
                outputs = model(
                    input_ids=inputs[f"teacher_{view}_input_ids"],
                    attention_mask=inputs[f"teacher_{view}_attention_mask"],
                )
                prompt_len = inputs[f"teacher_{view}_prompt_length"]
                teacher_logits = outputs.logits[:, prompt_len - 1 : -1, :]
                candidate_logits = torch.gather(teacher_logits, dim=-1, index=candidate_indices)
                candidate_log_probs = masked_log_softmax(
                    candidate_logits / self.temperature,
                    candidate_mask,
                    dim=-1,
                )
                view_log_probs.append(candidate_log_probs)
                del outputs, teacher_logits, candidate_logits
                empty_cache()
        return torch.stack(view_log_probs, dim=1)

    def _teacher_view_sampled_log_prob_pass(self, model, inputs, sampled_token_ids):
        view_sample_log_probs = []
        sample_index = sampled_token_ids.unsqueeze(-1)
        for view in self.pi_views:
            with torch.no_grad(), self._teacher_context(model):
                outputs = model(
                    input_ids=inputs[f"teacher_{view}_input_ids"],
                    attention_mask=inputs[f"teacher_{view}_attention_mask"],
                )
                prompt_len = inputs[f"teacher_{view}_prompt_length"]
                teacher_logits = outputs.logits[:, prompt_len - 1 : -1, :]
                teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
                teacher_sample_log_probs = torch.gather(
                    teacher_log_probs,
                    dim=-1,
                    index=sample_index,
                ).squeeze(-1)
                view_sample_log_probs.append(teacher_sample_log_probs)
                del outputs, teacher_logits, teacher_log_probs
                empty_cache()
        return torch.stack(view_sample_log_probs, dim=1)

    def _log_multi_view_metrics(
        self,
        model,
        shifted_labels,
        candidate_mask,
        weights,
        weight_aux,
        target_log_probs,
        log_consensus_target=False,
        target_metric_name=None,
        log_centrality=None,
        avsd_aux=None,
        loss_epi=None,
    ):
        mode = "train" if model.training else "eval"
        token_mask = shifted_labels != -100
        candidate_token_mask = candidate_mask & token_mask.unsqueeze(-1)

        self._append_metric(
            mode,
            "mv_candidate_size_mean",
            self._masked_mean(candidate_mask.sum(dim=-1).float(), token_mask),
        )
        self._append_metric(mode, "mv_weight_entropy", self._masked_mean(weight_aux["entropy"], token_mask))

        for view_idx, view in enumerate(self.pi_views):
            self._append_metric(
                mode,
                f"mv_weight_{view}",
                self._masked_mean(weights[:, :, view_idx], token_mask),
            )

        if log_centrality is None:
            log_centrality = self.view_weight_mode == "agreement_centrality"
        if log_centrality:
            centrality = weight_aux["centrality"]
            for view_idx, view in enumerate(self.pi_views):
                self._append_metric(
                    mode,
                    f"mv_centrality_{view}",
                    self._masked_mean(centrality[:, :, view_idx], token_mask),
                )

        if log_consensus_target and target_metric_name is None:
            target_metric_name = "consensus"
        if target_metric_name is not None:
            finite_targets = torch.isfinite(target_log_probs) & candidate_token_mask
            self._append_metric(
                mode,
                f"mv_{target_metric_name}_logprob_mean",
                self._masked_mean(target_log_probs, finite_targets),
            )

        if avsd_aux is None:
            return

        self._append_metric(
            mode,
            "mv_jensen_gap_mean",
            self._masked_mean(avsd_aux["jensen_gap"], candidate_token_mask),
        )
        if self.avsd_gate_mode == "avsd":
            self._append_metric(
                mode,
                "mv_consensus_adv_abs_mean",
                self._masked_mean(avsd_aux["consensus_adv"].abs(), candidate_token_mask),
            )
            self._append_metric(
                mode,
                "mv_avsd_mean",
                self._masked_mean(avsd_aux["avsd"], candidate_token_mask),
            )
        else:
            self._append_metric(
                mode,
                "mv_var_adv_mean",
                self._masked_mean(avsd_aux["var_adv"], candidate_token_mask),
            )
            if self.avsd_gate_mode == "consistency_exp":
                self._append_metric(
                    mode,
                    "mv_consistency_mean",
                    self._masked_mean(avsd_aux["consistency"], candidate_token_mask),
                )
                self._append_metric(
                    mode,
                    "mv_sign_consistency_mean",
                    self._masked_mean(avsd_aux["consistency"], candidate_token_mask),
                )
            else:
                self._append_metric(
                    mode,
                    "mv_sign_consistency_mean",
                    self._masked_mean(avsd_aux["sign_consistency"], candidate_token_mask),
                )
        self._append_metric(
            mode,
            "mv_gate_mean",
            self._masked_mean(avsd_aux["gate"], candidate_token_mask),
        )
        if "variance_factor" in avsd_aux:
            self._append_metric(
                mode,
                "mv_gate_variance_factor_mean",
                self._masked_mean(avsd_aux["variance_factor"], candidate_token_mask),
            )

        if torch.any(candidate_token_mask):
            gate_open_frac = (avsd_aux["gate"][candidate_token_mask] > 0.5).float().mean()
            self._append_metric(mode, "mv_gate_open_frac", gate_open_frac)

        if loss_epi is not None:
            self._append_metric(mode, "mv_epistemic_loss", loss_epi)

    def _log_multi_view_tinker_metrics(
        self,
        model,
        shifted_labels,
        student_sample_log_probs,
        view_sample_log_probs,
        target_sample_log_probs,
        avsd_aux=None,
        loss_epi=None,
    ):
        mode = "train" if model.training else "eval"
        token_mask = shifted_labels != -100

        uniform_entropy = torch.log(student_sample_log_probs.new_tensor(float(len(self.pi_views))))
        uniform_entropy = torch.full_like(student_sample_log_probs, uniform_entropy.item())
        self._append_metric(
            mode,
            "mv_weight_entropy",
            self._masked_mean(uniform_entropy, token_mask),
        )
        for view_idx, view in enumerate(self.pi_views):
            self._append_metric(mode, f"mv_weight_{view}", 1.0 / len(self.pi_views))
            self._append_metric(
                mode,
                f"mv_tinker_view_logprob_{view}",
                self._masked_mean(view_sample_log_probs[:, view_idx, :], token_mask),
            )

        self._append_metric(
            mode,
            "mv_tinker_student_logprob_mean",
            self._masked_mean(student_sample_log_probs, token_mask),
        )
        self._append_metric(
            mode,
            "mv_tinker_target_logprob_mean",
            self._masked_mean(target_sample_log_probs, token_mask),
        )
        self._append_metric(
            mode,
            "mv_tinker_advantage_mean",
            self._masked_mean(target_sample_log_probs - student_sample_log_probs, token_mask),
        )

        if avsd_aux is None:
            return

        self._append_metric(
            mode,
            "mv_jensen_gap_mean",
            self._masked_mean(avsd_aux["jensen_gap"], token_mask),
        )
        if self.avsd_gate_mode == "avsd":
            self._append_metric(
                mode,
                "mv_consensus_adv_abs_mean",
                self._masked_mean(avsd_aux["consensus_adv"].abs(), token_mask),
            )
            self._append_metric(
                mode,
                "mv_avsd_mean",
                self._masked_mean(avsd_aux["avsd"], token_mask),
            )
        else:
            self._append_metric(
                mode,
                "mv_var_adv_mean",
                self._masked_mean(avsd_aux["var_adv"], token_mask),
            )
            if self.avsd_gate_mode == "consistency_exp":
                self._append_metric(
                    mode,
                    "mv_consistency_mean",
                    self._masked_mean(avsd_aux["consistency"], token_mask),
                )
                self._append_metric(
                    mode,
                    "mv_sign_consistency_mean",
                    self._masked_mean(avsd_aux["consistency"], token_mask),
                )
            else:
                self._append_metric(
                    mode,
                    "mv_sign_consistency_mean",
                    self._masked_mean(avsd_aux["sign_consistency"], token_mask),
                )
        self._append_metric(mode, "mv_gate_mean", self._masked_mean(avsd_aux["gate"], token_mask))
        if "variance_factor" in avsd_aux:
            self._append_metric(
                mode,
                "mv_gate_variance_factor_mean",
                self._masked_mean(avsd_aux["variance_factor"], token_mask),
            )
        if torch.any(token_mask):
            gate_open_frac = (avsd_aux["gate"][token_mask] > 0.5).float().mean()
            self._append_metric(mode, "mv_gate_open_frac", gate_open_frac)
        if loss_epi is not None:
            self._append_metric(mode, "mv_epistemic_loss", loss_epi)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.multi_view_mode == "single":
            return self._compute_loss_single_view(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        return self._compute_loss_multi_view(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    def _compute_loss_multi_view_tinker(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        student_prompt_len = inputs["student_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
        student_sample_log_probs = torch.gather(
            student_log_probs,
            dim=-1,
            index=sampled_token_ids.unsqueeze(-1),
        ).squeeze(-1)

        if return_outputs:
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        del outputs_student, student_logits, student_log_probs
        empty_cache()

        view_sample_log_probs = self._teacher_view_sampled_log_prob_pass(
            model,
            inputs,
            sampled_token_ids,
        )
        target_sample_log_probs, avsd_aux = build_uniform_sampled_tinker_target(
            student_sample_log_probs,
            view_sample_log_probs,
            self.multi_view_mode,
            gate_alpha=self.avsd_gate_alpha,
            gate_var_coef=self.avsd_gate_var_coef,
            gate_gap_coef=self.avsd_gate_gap_coef,
            sign_threshold=self.avsd_sign_threshold,
            gate_mode=self.avsd_gate_mode,
            consistency_exp_variance_mode=self.avsd_consistency_exp_variance_mode,
            consistency_exp_var_coef=self.avsd_consistency_exp_var_coef,
        )

        loss = sampled_token_tinker_loss_from_log_probs(
            student_sample_log_probs,
            target_sample_log_probs,
            shifted_labels,
        )
        loss_epi = None
        if self.use_epistemic_preservation and self.multi_view_mode == "avsd":
            loss_epi = sampled_epistemic_preservation_loss(
                student_sample_log_probs,
                avsd_aux["gate"],
                shifted_labels,
                tau=self.epistemic_tau,
            )
            loss = loss + loss_epi

        self._log_multi_view_tinker_metrics(
            model,
            shifted_labels,
            student_sample_log_probs,
            view_sample_log_probs,
            target_sample_log_probs,
            avsd_aux=avsd_aux if self.multi_view_mode == "avsd" else None,
            loss_epi=loss_epi,
        )

        del student_sample_log_probs, view_sample_log_probs, target_sample_log_probs
        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return loss, minimal_output
        return loss

    def _compute_loss_multi_view(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if self.use_thinking_machines_loss:
            return self._compute_loss_multi_view_tinker(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        del num_items_in_batch
        student_prompt_len = inputs["student_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        topk_by_view = self._teacher_view_topk_pass(model, inputs)
        candidate_indices, candidate_mask = build_candidate_union(topk_by_view, sampled_token_ids)
        del topk_by_view

        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]
        student_candidate_logits = torch.gather(student_logits, dim=-1, index=candidate_indices)
        student_log_probs = masked_log_softmax(
            student_candidate_logits / self.temperature,
            candidate_mask,
            dim=-1,
        )

        if return_outputs:
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        del outputs_student, student_logits, student_candidate_logits
        empty_cache()

        view_log_probs = self._teacher_view_candidate_pass(model, inputs, candidate_indices, candidate_mask)
        view_weight_mode = "uniform" if self.multi_view_mode == "arithmetic" else self.view_weight_mode
        weights, weight_aux = compute_view_weights(
            view_log_probs,
            candidate_mask,
            mode=view_weight_mode,
            agreement_eta=self.view_agreement_eta,
        )

        loss_epi = None
        if self.multi_view_mode == "consensus":
            target_log_probs = build_consensus_target(view_log_probs, weights, candidate_mask)
            loss = generalized_jsd_from_log_probs(
                student_log_probs,
                target_log_probs,
                shifted_labels,
                beta=self.beta,
                token_clip=self.jsd_token_clip,
            )
            self._log_multi_view_metrics(
                model,
                shifted_labels,
                candidate_mask,
                weights,
                weight_aux,
                target_log_probs,
                target_metric_name="consensus",
            )
        elif self.multi_view_mode == "arithmetic":
            target_log_probs = build_arithmetic_target(view_log_probs, candidate_mask)
            loss = generalized_jsd_from_log_probs(
                student_log_probs,
                target_log_probs,
                shifted_labels,
                beta=self.beta,
                token_clip=self.jsd_token_clip,
            )
            self._log_multi_view_metrics(
                model,
                shifted_labels,
                candidate_mask,
                weights,
                weight_aux,
                target_log_probs,
                target_metric_name="arithmetic",
                log_centrality=False,
            )
        else:
            target_log_probs, avsd_aux = build_avsd_target(
                student_log_probs,
                view_log_probs,
                weights,
                candidate_mask,
                gate_alpha=self.avsd_gate_alpha,
                gate_var_coef=self.avsd_gate_var_coef,
                gate_gap_coef=self.avsd_gate_gap_coef,
                sign_threshold=self.avsd_sign_threshold,
                gate_mode=self.avsd_gate_mode,
                consistency_exp_variance_mode=self.avsd_consistency_exp_variance_mode,
                consistency_exp_var_coef=self.avsd_consistency_exp_var_coef,
            )
            loss_sd = generalized_jsd_from_log_probs(
                student_log_probs,
                target_log_probs,
                shifted_labels,
                beta=self.beta,
                token_clip=self.jsd_token_clip,
            )
            loss = loss_sd
            if self.use_epistemic_preservation:
                loss_epi = epistemic_preservation_loss(
                    student_log_probs,
                    avsd_aux["gate"],
                    shifted_labels,
                    tau=self.epistemic_tau,
                )
                loss = loss + loss_epi

            self._log_multi_view_metrics(
                model,
                shifted_labels,
                candidate_mask,
                weights,
                weight_aux,
                target_log_probs,
                avsd_aux=avsd_aux,
                loss_epi=loss_epi,
            )

        del view_log_probs, weights, weight_aux, target_log_probs, student_log_probs
        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return loss, minimal_output
        return loss

    def _compute_loss_single_view(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute the self-distillation loss with memory-efficient log-prob extraction.

        Memory optimization: Extract only needed log-probs immediately and free large tensors.
        """
        # Get batch-level prompt lengths
        student_prompt_len = inputs["student_prompt_length"]
        teacher_prompt_len = inputs["teacher_prompt_length"]
        sampled_token_ids = inputs["student_input_ids"][:, student_prompt_len:]
        shifted_labels = inputs["labels"][:, student_prompt_len:]

        # === STUDENT FORWARD - Extract log-probs immediately ===
        outputs_student = model(
            input_ids=inputs["student_input_ids"],
            attention_mask=inputs["student_attention_mask"],
        )

        # Extract only what we need and convert to log-probs immediately
        student_logits = outputs_student.logits[:, student_prompt_len - 1 : -1, :]

        if self.use_thinking_machines_loss:
            # For reverse KL, we only need log-probs of sampled tokens
            student_log_probs = F.log_softmax(student_logits / self.temperature, dim=-1)
            student_log_probs_sampled = torch.gather(
                student_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
            ).squeeze(-1)
            del student_logits, student_log_probs  # Free immediately!
        else:
            # For JSD, keep logits (temperature will be applied in generalized_jsd_loss)
            student_logits_for_loss = student_logits
            del student_logits

        # Free the full outputs (but keep reference for return_outputs if needed)
        if return_outputs:
            # Create a minimal output object to return (just the loss, no logits)
            class MinimalOutput:
                def __init__(self):
                    self.loss = None

            minimal_output = MinimalOutput()

        del outputs_student
        empty_cache()

        # === TEACHER FORWARD - Extract log-probs immediately ===
        # Choose teacher context based on mode:
        #   use_ema_teacher  → swap in EMA weights temporarily
        #   fixed_teacher    → disable LoRA adapters (base model = initial policy)
        #   default (dynamic)→ no-op, use current student weights
        with torch.no_grad(), self._teacher_context(model):
            outputs_teacher = model(
                input_ids=inputs["teacher_input_ids"],
                attention_mask=inputs["teacher_attention_mask"],
            )

            teacher_logits = outputs_teacher.logits[:, teacher_prompt_len - 1 : -1, :]

            if self.use_thinking_machines_loss:
                teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
                teacher_log_probs_sampled = torch.gather(
                    teacher_log_probs, dim=-1, index=sampled_token_ids.unsqueeze(-1)
                ).squeeze(-1)
                del teacher_logits, teacher_log_probs  # Free immediately!
            else:
                teacher_logits_for_loss = teacher_logits
                del teacher_logits

            del outputs_teacher
            empty_cache()

        # === COMPUTE LOSS with only small tensors ===
        if self.use_thinking_machines_loss:
            # Thinking Machines uses RL-style policy gradient:
            # Advantage = log π_teacher(x) - log π_student(x)
            # Loss = -E[Advantage * log π_student(x)]
            #
            # CRITICAL: advantage must be detached to prevent gradients flowing through it.
            # We want: ∇θ L = -E[A(x) * ∇θ log π_student(x)]
            # NOT: ∇θ L = -E[(T(x) - S(x)) * ∇θ S(x)] where both terms differentiate

            advantage = (teacher_log_probs_sampled - student_log_probs_sampled).detach()

            # Apply masking before computing loss
            if shifted_labels is not None:
                mask = shifted_labels != -100
                advantage = advantage[mask]
                student_log_probs_sampled_masked = student_log_probs_sampled[mask]
            else:
                student_log_probs_sampled_masked = student_log_probs_sampled

            # Policy gradient loss: -advantage * log π_student
            # Negative because we minimize loss (gradient descent), but want to maximize reward
            loss = -(advantage * student_log_probs_sampled_masked).mean()

            del (
                student_log_probs_sampled,
                teacher_log_probs_sampled,
                advantage,
                student_log_probs_sampled_masked,
            )
        else:
            # Temperature is applied inside generalized_jsd_loss
            loss = self.generalized_jsd_loss(
                student_logits=student_logits_for_loss,
                teacher_logits=teacher_logits_for_loss,
                labels=shifted_labels,
                beta=self.beta,
                temperature=self.temperature,  # Let the function handle temperature
                top_k=self.top_k_loss,
                token_clip=self.jsd_token_clip,
            )
            del student_logits_for_loss, teacher_logits_for_loss

        empty_cache()

        if return_outputs:
            minimal_output.loss = loss
            return (loss, minimal_output)
        else:
            return loss

    def generate_teacher_reasoning(
        self, model, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher's reasoning about the solution."""
        if self.use_vllm:
            # Use vLLM for fast reasoning generation
            return self._generate_teacher_reasoning_vllm(teacher_reasoning_prompts)
        else:
            # Use transformers generation (slower)
            with torch.no_grad():
                # Temporarily enable KV cache
                original_use_cache = model.config.use_cache
                original_gen_use_cache = self.reasoning_generation_config.use_cache

                model.config.use_cache = True
                self.reasoning_generation_config.use_cache = True

                # If fixed_teacher=True, disable LoRA adapters
                adapter_context = (
                    self.accelerator.unwrap_model(model).disable_adapter()
                    if self.fixed_teacher and is_peft_model(model)
                    else nullcontext()
                )

                try:
                    with adapter_context:
                        reasoning_outputs = model.generate(
                            input_ids=teacher_reasoning_prompts,
                            attention_mask=teacher_reasoning_attention_mask,
                            generation_config=self.reasoning_generation_config,
                            return_dict_in_generate=True,
                            use_cache=True,
                        )
                        reasoning_ids = reasoning_outputs.sequences
                finally:
                    model.config.use_cache = original_use_cache
                    self.reasoning_generation_config.use_cache = original_gen_use_cache

                return reasoning_ids

    def generate_on_policy_outputs(self, model, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs from student prompts only."""
        import time

        start_time = time.time()

        # Temporarily enable KV cache for generation if it was disabled for training
        original_use_cache = model.config.use_cache
        original_gen_use_cache = generation_config.use_cache

        model.config.use_cache = True
        generation_config.use_cache = True

        print(f"\n{'='*80}")
        print(f"GENERATION DEBUG INFO:")
        print(f"  Model dtype: {model.dtype}")
        print(f"  Model config use_cache: {model.config.use_cache}")
        print(f"  Attention implementation: {getattr(model.config, '_attn_implementation', 'unknown')}")
        print(f"  Generation config use_cache: {generation_config.use_cache}")
        print(f"  Batch size: {inputs['student_prompts'].shape[0]}")
        print(f"  Prompt length: {inputs['student_prompts'].shape[1]}")
        print(f"  Max new tokens: {generation_config.max_new_tokens}")
        print(f"{'='*80}\n")

        # Generate output with respect to the student prompt only
        try:
            generated_outputs = model.generate(
                input_ids=inputs["student_prompts"],
                attention_mask=inputs.get("student_prompt_attention_mask", None),
                generation_config=generation_config,
                return_dict_in_generate=True,
                use_cache=True,
            )
            # Get the generated token IDs
            generated_tokens = generated_outputs.sequences
        finally:
            # Restore original settings
            model.config.use_cache = original_use_cache
            generation_config.use_cache = original_gen_use_cache

        elapsed_time = time.time() - start_time
        num_prompts = generated_tokens.shape[0]
        total_completion_tokens = generated_tokens.shape[1] - inputs["student_prompts"].shape[1]
        num_tokens = total_completion_tokens * num_prompts
        avg_completion_length = total_completion_tokens
        tokens_per_sec = num_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {num_tokens}, avg length: {avg_completion_length}, speed: {tokens_per_sec:.1f} tok/s"
        )

        new_attention_mask = torch.ones_like(generated_tokens)
        new_labels = generated_tokens.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[generated_tokens == pad_token_id] = 0

        return generated_tokens, new_attention_mask, new_labels

    @profiling_decorator
    def _generate_on_policy_outputs_vllm(self, inputs, generation_config, pad_token_id=None):
        """Generate on-policy outputs from student prompts using vLLM."""
        import time

        device = self.accelerator.device

        prompts_text_for_vllm = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )
        # Remove padding token text if it appears, as vLLM expects clean prompts
        if self.processing_class.pad_token:
            prompts_text_for_vllm = [
                p.replace(self.processing_class.pad_token, "") for p in prompts_text_for_vllm
            ]

        # Also decode prompts WITH special tokens for logging
        prompts_text_with_special = self.processing_class.batch_decode(
            inputs["student_prompts"],
            skip_special_tokens=False,
        )

        # system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
        # target_system_prompt = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
        # prompts_text = [p.replace(target_system_prompt, system_prompt) for p in prompts_text]
        # Add system prompt to prompts

        max_completion_length = generation_config.max_new_tokens
        temperature = generation_config.temperature
        # vLLM uses top_k=-1 for no top_k, transformers uses 0 or None.
        top_k = generation_config.top_k if generation_config.top_k and generation_config.top_k > 0 else -1
        # top_p, repetition_penalty, min_p, presence_penalty are not directly in generation_config, get from trainer args
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0
        repetition_penalty = self.args.repetition_penalty if hasattr(self.args, "repetition_penalty") else 1.0
        min_p = self.args.min_p if hasattr(self.args, "min_p") else 0.0
        presence_penalty = self.args.presence_penalty if hasattr(self.args, "presence_penalty") else 0.0

        # Start timing for vLLM generation
        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text_for_vllm)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,  # In GKD, we generate 1 completion per prompt from student
                    repetition_penalty=repetition_penalty,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    min_p=min_p,
                    max_tokens=max_completion_length,
                    presence_penalty=presence_penalty,
                    guided_decoding_regex=self.vllm_guided_decoding_regex,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text_for_vllm),
                (self.accelerator.process_index + 1) * len(prompts_text_for_vllm),
            )
            completion_ids = completion_ids[process_slice]
        elif self.vllm_mode == "colocate":
            if self.vllm_guided_decoding_regex:
                guided_decoding = GuidedDecodingParams(
                    backend="outlines", regex=self.vllm_guided_decoding_regex
                )
            else:
                guided_decoding = None
            sampling_params = SamplingParams(
                n=1,
                repetition_penalty=repetition_penalty,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                max_tokens=max_completion_length,
                presence_penalty=presence_penalty,
                guided_decoding=guided_decoding,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Gather prompts from all ranks in the TP group and flatten.
                # Each rank starts with its own prompts; after gathering, all ranks see the full group set.
                orig_size = len(prompts_text_for_vllm)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(
                    gathered_prompts, prompts_text_for_vllm, group=self.vllm_tp_group
                )
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text_for_vllm

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                # Slice completions for this rank within its TP group.
                # Each rank generates all outputs — we keep only our share.
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)
        else:
            raise ValueError(f"Unknown vllm_mode: {self.vllm_mode}")

        # Calculate and print vLLM generation statistics
        elapsed_time = time.time() - start_time
        total_completion_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        avg_completion_length = total_completion_tokens / num_prompts if num_prompts > 0 else 0
        tokens_per_sec = total_completion_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            f"vLLM generation done - elapsed time: {elapsed_time:.2f}s, prompts: {num_prompts}, total tokens: {total_completion_tokens}, avg length: {avg_completion_length:.1f}, speed: {tokens_per_sec:.1f} tok/s"
        )

        # We need to combine prompt and completion for new_input_ids
        # Tokenize prompts again to get prompt_ids on the correct device and format
        # Use prompts_text_for_vllm (without special tokens) for tokenization since vLLM expects clean text
        # Ensure add_special_tokens=False as vLLM typically handles prompts as raw text
        # Calculate max_length for prompts, ensuring it's positive
        prompt_max_length = (
            max(1, self.args.max_length - max_completion_length) if self.args.max_length else None
        )
        prompt_tokenized = self.processing_class(
            prompts_text_for_vllm,
            return_tensors="pt",
            padding="longest",
            truncation=True if prompt_max_length else False,
            max_length=prompt_max_length,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        # Manually pad/truncate completions to max_completion_length length before using pad function
        padded_completion_ids_list = []
        for completion_tensor in completion_ids_tensors:
            if len(completion_tensor) > max_completion_length:
                # Truncate if longer than max_completion_length
                padded_completion_ids_list.append(completion_tensor[:max_completion_length])
            elif len(completion_tensor) < max_completion_length:
                # Pad if shorter than max_completion_length
                padding_needed = max_completion_length - len(completion_tensor)
                padded_tensor = torch.cat(
                    [
                        completion_tensor,
                        torch.full(
                            (padding_needed,), pad_token_id, device=device, dtype=completion_tensor.dtype
                        ),
                    ]
                )
                padded_completion_ids_list.append(padded_tensor)
            else:
                # Already the right length
                padded_completion_ids_list.append(completion_tensor)

        # Now all tensors are the same length, so we can stack them
        padded_completion_ids = torch.stack(padded_completion_ids_list)

        # Ensure prompt_ids and padded_completion_ids are 2D
        if prompt_ids.ndim == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        if padded_completion_ids.ndim == 1:
            padded_completion_ids = padded_completion_ids.unsqueeze(0)

        new_input_ids = torch.cat([prompt_ids, padded_completion_ids], dim=1)

        new_attention_mask = torch.ones_like(new_input_ids, device=device)
        new_labels = new_input_ids.clone()

        if pad_token_id is not None:
            new_labels[new_labels == pad_token_id] = -100
            new_attention_mask[new_input_ids == pad_token_id] = 0

        # Extract completion texts from the generated completion IDs
        completion_texts = []
        for comp_ids in completion_ids:
            completion_text = self.processing_class.decode(comp_ids, skip_special_tokens=False)
            completion_texts.append(completion_text)

        return new_input_ids, new_attention_mask, new_labels, prompts_text_with_special, completion_texts

    def _generate_teacher_reasoning_vllm(
        self, teacher_reasoning_prompts, teacher_reasoning_attention_mask=None
    ):
        """Generate teacher's reasoning using vLLM."""
        import time

        device = self.accelerator.device

        # Decode prompts for vLLM
        prompts_text = self.processing_class.batch_decode(
            teacher_reasoning_prompts,
            skip_special_tokens=True,
        )
        if self.processing_class.pad_token:
            prompts_text = [p.replace(self.processing_class.pad_token, "") for p in prompts_text]

        max_reasoning_length = self.reasoning_generation_config.max_new_tokens
        temperature = self.reasoning_generation_config.temperature
        top_k = (
            self.reasoning_generation_config.top_k
            if self.reasoning_generation_config.top_k and self.reasoning_generation_config.top_k > 0
            else -1
        )
        top_p = self.args.top_p if hasattr(self.args, "top_p") else 1.0

        start_time = time.time()

        if self.vllm_mode == "server":
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                completion_ids = self.vllm_client.generate(
                    prompts=all_prompts_text,
                    n=1,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    max_tokens=max_reasoning_length,
                )
            else:
                completion_ids = [None] * len(all_prompts_text)
            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            process_slice = slice(
                self.accelerator.process_index * len(prompts_text),
                (self.accelerator.process_index + 1) * len(prompts_text),
            )
            completion_ids = completion_ids[process_slice]

        elif self.vllm_mode == "colocate":
            sampling_params = SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_tokens=max_reasoning_length,
            )

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                orig_size = len(prompts_text)
                gathered_prompts = [None for _ in range(self.vllm_tensor_parallel_size)]
                torch.distributed.all_gather_object(gathered_prompts, prompts_text, group=self.vllm_tp_group)
                all_prompts_text = [p for sublist in gathered_prompts for p in sublist]
            else:
                all_prompts_text = prompts_text

            all_outputs = self.vllm_engine.generate(
                all_prompts_text, sampling_params=sampling_params, use_tqdm=False
            )
            completion_ids = [output.token_ids for outputs in all_outputs for output in outputs.outputs]

            if hasattr(self, "vllm_tp_group") and self.vllm_tensor_parallel_size > 1:
                local_rank_in_group = torch.distributed.get_rank(group=self.vllm_tp_group)
                tp_slice = slice(local_rank_in_group * orig_size, (local_rank_in_group + 1) * orig_size)
                completion_ids = completion_ids[tp_slice]

            if self.vllm_enable_sleep_mode:
                self.vllm_engine.sleep(level=2)

        elapsed_time = time.time() - start_time
        total_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        print(
            f"vLLM teacher reasoning generation done - elapsed: {elapsed_time:.2f}s, prompts: {num_prompts}, tokens: {total_tokens}, speed: {total_tokens/elapsed_time:.1f} tok/s"
        )

        # Combine prompt + completion
        prompt_tokenized = self.processing_class(
            prompts_text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            add_special_tokens=False,
        ).to(device)
        prompt_ids = prompt_tokenized.input_ids

        completion_ids_tensors = [torch.tensor(ids, device=device) for ids in completion_ids]
        padded_completions = pad(
            completion_ids_tensors, padding_value=self.processing_class.pad_token_id, padding_side="right"
        )

        reasoning_ids = torch.cat([prompt_ids, padded_completions], dim=1)

        return reasoning_ids

    def _sync_fsdp_params_to_vllm(self, module: nn.Module, prefix: str = "", visited=None):
        """Memory-efficient post-order traversal of FSDP modules to extract full parameters and sync with student vLLM."""
        if visited is None:
            visited = set()

        for child_name, child_module in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            # recurse into the child
            self._sync_fsdp_params_to_vllm(child_module, prefix=child_prefix, visited=visited)

        if isinstance(module, FSDP):
            with FSDP.summon_full_params(module, recurse=False, writeback=False):
                for param_name, param in module.named_parameters():
                    full_name = f"{prefix}.{param_name}" if prefix else param_name
                    for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                        full_name = full_name.replace(extra, "")

                    if full_name in visited:
                        continue  # skip FSDP subtrees already traversed
                    visited.add(full_name)

                    if self.vllm_mode == "server" and self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(full_name, param.data)
                    elif self.vllm_mode == "colocate":
                        llm_model = (
                            self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                        )
                        llm_model.load_weights([(full_name, param.data)])

    def _move_model_to_vllm(self):
        """Synchronize student model weights to vLLM engine."""
        # For DeepSpeed ZeRO-3 and FSDP, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        if zero_stage_3:
            import deepspeed

            gather_if_zero3 = deepspeed.zero.GatheredParameters
        else:
            gather_if_zero3 = nullcontext

        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["weights"])

        if is_peft_model(self.model):
            # With PEFT and FSDP/DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as
            # merging adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                if self.is_fsdp_enabled:  # note if using FSDP, gather_if_zero3 is nullcontext
                    # Update vLLM weights while parameters are gathered
                    # For PEFT with FSDP we need to use the memory efficient post-order traversal
                    self._sync_fsdp_params_to_vllm(self.model)
                else:
                    # DeepSpeed ZeRO-3 with PEFT
                    for name, param in self.model.named_parameters():
                        # When using PEFT, we need to recover the original parameter name and discard some parameters
                        name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                        if self.model.prefix in name:
                            continue
                        # When module to save, remove its prefix and discard the original module
                        if "original_module" in name:
                            continue
                        name = name.replace("modules_to_save.default.", "")

                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])
                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather (if needed) and update each parameter individually.
            if self.is_fsdp_enabled:
                # use memory-efficient post-order traversal for FSDP
                self._sync_fsdp_params_to_vllm(self.model)
            else:
                # For DeepSpeed ZeRO-3, gather each parameter individually like GRPO trainer
                for name, param in self.model.named_parameters():
                    with gather_if_zero3([param]):
                        if self.vllm_mode == "server" and self.accelerator.is_main_process:
                            self.vllm_client.update_named_param(name, param.data)
                        elif self.vllm_mode == "colocate":
                            llm_model = (
                                self.vllm_engine.llm_engine.model_executor.driver_worker.model_runner.model
                            )
                            llm_model.load_weights([(name, param.data)])

        # Reset cache on vLLM
        if self.vllm_mode == "server" and self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
        elif self.vllm_mode == "colocate":
            self.vllm_engine.reset_prefix_cache()

    def _wake_vllm_if_needed(self):
        if self.vllm_mode == "colocate" and self.vllm_enable_sleep_mode:
            empty_cache()
            self.vllm_engine.wake_up(tags=["kv_cache"])

    def _save_generation_outputs(self, step: int):
        """Save generation outputs to disk."""
        if not self.accelerator.is_main_process:
            return

        if len(self._generation_outputs_buffer) == 0:
            return

        import json
        from pathlib import Path

        # Create generations directory in output_dir
        generations_dir = Path(self.args.output_dir) / "generations"
        generations_dir.mkdir(parents=True, exist_ok=True)

        # Save to JSON file
        output_file = generations_dir / f"generations_step_{step}.json"

        output_data = {
            "step": step,
            "num_samples": len(self._generation_outputs_buffer),
            "generations": self._generation_outputs_buffer,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f"\n{'='*80}")
        print(f"Saved {len(self._generation_outputs_buffer)} generation outputs to:")
        print(f"  {output_file}")
        print(f"{'='*80}\n")

        # Clear buffer after saving
        self._generation_outputs_buffer.clear()

    @profiling_decorator
    def training_step(
        self, model: nn.Module, inputs: dict[str, torch.Tensor | Any], num_items_in_batch: int | None = None
    ) -> torch.Tensor:
        """
        Perform a training step with self-distillation.

        If reason_first=True:
        1. Generate teacher's reasoning about the solution
        2. Append reasoning to teacher prompt
        3. Generate completions from student prompts
        4. Compute JSD loss

        Otherwise:
        1. Generate completions from student prompts
        2. Construct full sequences for both student and teacher with the generation
        3. Compute JSD loss on the generation tokens
        """
        on_policy = True

        # === REASONING PHASE (if enabled) ===
        if self.reason_first:
            print(f"\n{'='*80}")
            print("REASONING PHASE: Teacher analyzing solution...")
            print(f"{'='*80}\n")

            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                # Generate teacher's reasoning
                teacher_reasoning_ids = self.generate_teacher_reasoning(
                    unwrapped_model,
                    inputs["teacher_reasoning_prompts"],
                    inputs.get("teacher_reasoning_attention_mask"),
                )

                # Decode reasoning
                reasoning_prompt_len = inputs["teacher_reasoning_prompt_length"]
                reasoning_completions = teacher_reasoning_ids[:, reasoning_prompt_len:]
                reasoning_texts = self.processing_class.batch_decode(
                    reasoning_completions, skip_special_tokens=True
                )

                # Occasionally print reasoning
                if random.random() < 0.01:
                    print(f"\n{'='*80}")
                    print(f"TEACHER REASONING SAMPLE (Step {self.state.global_step}):")
                    print(f"{'='*80}")
                    sample_idx = random.randint(0, len(reasoning_texts) - 1)
                    print(f"\n{'='*80}")
                    # Decode the prompt from token IDs to text
                    sample_prompt = self.processing_class.decode(
                        inputs["teacher_reasoning_prompts"][sample_idx], skip_special_tokens=False
                    )
                    print(f"PROMPT:\n{sample_prompt}")
                    print(f"\nReasoning:\n{reasoning_texts[sample_idx]}")
                    print(f"{'='*80}\n")

                # Update teacher prompts with reasoning
                # Construct: [teacher_reasoning_prompt][reasoning][transition_to_teaching]
                teacher_prompts_with_reasoning = torch.cat(
                    [
                        inputs["teacher_reasoning_prompts"],
                        reasoning_completions,
                        inputs["teacher_transition_tokens"],
                    ],
                    dim=1,
                )

                # Update inputs with new teacher prompts
                inputs["teacher_prompts"] = teacher_prompts_with_reasoning
                teacher_attention_mask = torch.ones_like(teacher_prompts_with_reasoning)
                if self.processing_class.pad_token_id is not None:
                    teacher_attention_mask[
                        teacher_prompts_with_reasoning == self.processing_class.pad_token_id
                    ] = 0
                inputs["teacher_prompt_attention_mask"] = teacher_attention_mask
                inputs["teacher_prompt_length"] = teacher_prompts_with_reasoning.shape[1]

        # === GENERATION PHASE ===
        if self.use_vllm:
            self._wake_vllm_if_needed()
            result = self._generate_on_policy_outputs_vllm(
                inputs, self.generation_config, self.processing_class.pad_token_id
            )
            generated_ids, generated_attention_mask, _, prompt_texts, completion_texts = result
        else:
            with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
                result = self.generate_on_policy_outputs(
                    unwrapped_model, inputs, self.generation_config, self.processing_class.pad_token_id
                )
                generated_ids, generated_attention_mask, _ = result
                # Decode for logging
                prompt_texts = self.processing_class.batch_decode(
                    inputs["student_prompts"], skip_special_tokens=False
                )
                student_prompt_len = inputs["student_prompt_length"]
                completion_ids = generated_ids[:, student_prompt_len:]
                completion_texts = self.processing_class.batch_decode(
                    completion_ids, skip_special_tokens=False
                )

        # Get batch-level student prompt length
        student_prompt_len = inputs["student_prompt_length"]

        # Extract generation part (same slice for all examples since prompts are padded)
        generation_ids = generated_ids[:, student_prompt_len:]

        # Construct student full sequence: [student_prompt][generation]
        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask

        if self.multi_view_mode == "single":
            # Construct teacher full sequence: [teacher_prompt][generation]
            teacher_prompts = inputs["teacher_prompts"]
            teacher_full_ids = torch.cat([teacher_prompts, generation_ids], dim=1)

            # Create attention mask for teacher
            teacher_attention_mask = torch.ones_like(teacher_full_ids)
            if self.processing_class.pad_token_id is not None:
                teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0

            inputs["teacher_input_ids"] = teacher_full_ids
            inputs["teacher_attention_mask"] = teacher_attention_mask
        else:
            for view in self.pi_views:
                prompt_ids = inputs[f"teacher_{view}_prompts"]
                teacher_full_ids = torch.cat([prompt_ids, generation_ids], dim=1)
                teacher_attention_mask = torch.ones_like(teacher_full_ids)
                if self.processing_class.pad_token_id is not None:
                    teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0
                inputs[f"teacher_{view}_input_ids"] = teacher_full_ids
                inputs[f"teacher_{view}_attention_mask"] = teacher_attention_mask

        # Create labels for generation tokens
        # Mask prompt tokens (use per-example lengths for accurate masking)
        labels = generated_ids.clone()
        for i in range(labels.shape[0]):
            actual_prompt_len = inputs["student_prompt_lengths_per_example"][i].item()
            labels[i, :actual_prompt_len] = -100  # Mask actual prompt

        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100

        inputs["labels"] = labels

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompt_texts))
        self._textual_logs["completion"].extend(gather_object(completion_texts))

        # Collect generation outputs for saving
        for prompt, completion in zip(prompt_texts, completion_texts):
            self._generation_outputs_buffer.append(
                {"step": self.state.global_step, "prompt": prompt, "completion": completion}
            )

        # Occasionally print student's generation with 1% probability
        if random.random() < 0.01:
            print(f"\n{'='*80}")
            print(f"STUDENT GENERATION SAMPLE (Step {self.state.global_step}):")
            print(f"{'='*80}")
            sample_idx = random.randint(0, len(prompt_texts) - 1)
            print(f"\nPrompt:\n{prompt_texts[sample_idx]}")
            print(f"\nCompletion:\n{completion_texts[sample_idx]}")
            print(f"{'='*80}\n")

        loss = super().training_step(model, inputs, num_items_in_batch)

        # Save generation outputs every N steps
        if (
            self.state.global_step > 0
            and self.state.global_step % self._generation_save_frequency == 0
            and self.accelerator.sync_gradients
        ):
            self._save_generation_outputs(self.state.global_step)

        loss_scalar = float(loss.detach())
        ga = max(1, int(self.args.gradient_accumulation_steps))
        step_equiv = 1.0 / ga

        if on_policy:
            self._on_policy_loss_total += loss_scalar
            self._on_policy_step_equiv += step_equiv
        else:
            self._off_policy_loss_total += loss_scalar
            self._off_policy_step_equiv += step_equiv
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model.training else "eval"
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        if mode == "train":
            device = self.accelerator.device if hasattr(self.accelerator, "device") else torch.device("cpu")
            # Track on/off-policy loss statistics
            vec = torch.tensor(
                [
                    self._on_policy_loss_total,
                    self._off_policy_loss_total,
                    self._on_policy_step_equiv,
                    self._off_policy_step_equiv,
                ],
                dtype=torch.float64,
                device=device,
            )

            # Sum across processes so we mirror Trainer's distributed reduction
            if (
                getattr(self.accelerator, "distributed_type", DistributedType.NO) != DistributedType.NO
                and dist.is_available()
                and dist.is_initialized()
            ):
                dist.all_reduce(vec, op=dist.ReduceOp.SUM)

            (
                on_sum,
                off_sum,
                on_eq,
                off_eq,
            ) = vec.tolist()

            # Compute category averages over the *same window* as Trainer's logs
            # (avoid div-by-zero if, e.g., no on-policy steps in the window)
            if on_eq > 0:
                logs["on_policy_loss"] = round(on_sum / on_eq, 4)
            if off_eq > 0:
                logs["off_policy_loss"] = round(off_sum / off_eq, 4)

            # Reset window accumulators after logging (just like Trainer resets its window)
            self._on_policy_loss_total = self._off_policy_loss_total = 0.0
            self._on_policy_step_equiv = self._off_policy_step_equiv = 0.0

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs, start_time)
        self._metrics[mode].clear()
