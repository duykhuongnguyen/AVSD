from collections import Counter, defaultdict
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.utils import is_peft_model
from transformers import TrainerCallback, TrainerControl, TrainerState

try:
    from trl.extras.profiling import profiling_decorator
except ImportError:
    def profiling_decorator(func):
        return func

from trl.trainer.sft_trainer import SFTTrainer
try:
    from trl.trainer.utils import disable_dropout_in_model, empty_cache
except ImportError:
    from trl.trainer.utils import disable_dropout_in_model

    def empty_cache():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

from avsd.common.multiview_distill import (
    build_uniform_sampled_tinker_target,
    sampled_token_tinker_loss_from_log_probs,
)

from avsd.code.data.code_dataset import normalize_code_example
from avsd.code.execution.extract_code import extract_python_code
from avsd.code.execution.feedback import build_feedback
from avsd.code.execution.run_tests import run_code_on_tests
from avsd.code.execution.sandbox import check_sandbox_available
from avsd.code.prompts.view_prompts import build_teacher_prompt, normalize_code_views
from avsd.code.training.code_collator import CodeAVSDDataCollator
from avsd.code.training.vllm_utils import CodeVLLMGenerator, CodeVLLMSamplingConfig


def force_skip_sft_dataset_preparation(args) -> None:
    args.dataset_kwargs = dict(args.dataset_kwargs or {})
    args.dataset_kwargs["skip_prepare_dataset"] = True


class CodeVLLMSyncCallback(TrainerCallback):
    """Same sync timing as math OPSD: update colocated vLLM only after optimizer steps."""

    def __init__(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        del args, control, kwargs
        if (
            self.trainer.use_vllm
            and state.global_step != self.trainer._last_vllm_sync_step
            and state.global_step % self.trainer.vllm_sync_frequency == 0
        ):
            if hasattr(self.trainer.accelerator, "sync_gradients") and self.trainer.accelerator.sync_gradients:
                self.trainer._move_model_to_vllm()
                self.trainer._last_vllm_sync_step = state.global_step


class CodeAVSDTrainer(SFTTrainer):
    """Code-domain trainer with math OPSD mechanics and code-only prompts/data."""

    _tag_names = ["trl", "code-avsd"]
    _name = "CodeAVSD"

    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        peft_config=None,
        *,
        tokenizer=None,
        views: str | tuple[str, ...] = ("reference", "hint", "feedback"),
        multi_view_mode: str = "avsd",
        single_view: str | None = None,
        max_prompt_length: int = 8192,
        max_response_tokens: int | None = None,
        generation_min_p: float = 0.0,
        generation_presence_penalty: float = 0.0,
        generation_repetition_penalty: float = 1.0,
        student_enable_thinking: bool = False,
        teacher_enable_thinking: bool = True,
        fixed_teacher: bool = False,
        execution_timeout_s: float = 2.0,
        unsafe_subprocess_sandbox: bool = False,
        avsd_gate_mode: str = "avsd",
        avsd_gate_alpha: float = 6.0,
        avsd_gate_var_coef: float = 1.0,
        avsd_gate_gap_coef: float = 1.0,
        avsd_sign_threshold: float = 0.8,
        avsd_consistency_exp_variance_mode: str = "on",
        avsd_consistency_exp_var_coef: float = 1.0,
        model_name_or_path: str | None = None,
        **_ignored_compat_kwargs,
    ) -> None:
        if args is None:
            raise ValueError("CodeAVSDTrainer requires GOLDConfig/TrainingArguments.")
        processing_class = processing_class or tokenizer
        if processing_class is None:
            raise ValueError("CodeAVSDTrainer requires processing_class/tokenizer.")
        if not getattr(args, "use_vllm", True):
            raise ValueError("CodeAVSDTrainer requires use_vllm=True; transformers.generate is not supported.")

        self.model_name_or_path = model_name_or_path or (model if isinstance(model, str) else None)
        if self.model_name_or_path is None and getattr(model, "config", None) is not None:
            self.model_name_or_path = getattr(model.config, "_name_or_path", None)
        if not self.model_name_or_path:
            raise ValueError("model_name_or_path or string model is required for code vLLM rollouts.")

        self.code_tokenizer = processing_class
        self.views = normalize_code_views(views)
        self.multi_view_mode = _normalize_multi_view_mode(multi_view_mode)
        self.single_view = single_view or self.views[0]
        if self.single_view not in self.views:
            raise ValueError("single_view must be one of the configured views.")
        if self.multi_view_mode not in {"single", "consensus", "arithmetic", "avsd"}:
            raise ValueError("multi_view_mode must be one of: single, consensus, arithmetic, avsd")
        self.active_views = (self.single_view,) if self.multi_view_mode == "single" else tuple(self.views)
        self.max_prompt_length = max_prompt_length
        self.max_response_tokens = max_response_tokens or getattr(args, "max_completion_length", 4096)
        self.generation_min_p = generation_min_p
        self.generation_presence_penalty = generation_presence_penalty
        self.generation_repetition_penalty = generation_repetition_penalty
        self.student_enable_thinking = student_enable_thinking
        self.teacher_enable_thinking = teacher_enable_thinking
        self.fixed_teacher = fixed_teacher
        self.execution_timeout_s = execution_timeout_s
        self.unsafe_subprocess_sandbox = unsafe_subprocess_sandbox
        self.avsd_gate_mode = _normalize_avsd_gate_mode(avsd_gate_mode)
        self.avsd_gate_alpha = avsd_gate_alpha
        self.avsd_gate_var_coef = avsd_gate_var_coef
        self.avsd_gate_gap_coef = avsd_gate_gap_coef
        self.avsd_sign_threshold = avsd_sign_threshold
        self.avsd_consistency_exp_variance_mode = avsd_consistency_exp_variance_mode
        self.avsd_consistency_exp_var_coef = avsd_consistency_exp_var_coef
        self._code_metric_buffer: dict[str, list[float]] = defaultdict(list)

        if "feedback" in self.active_views and not unsafe_subprocess_sandbox:
            check_sandbox_available(raise_on_error=True)

        force_skip_sft_dataset_preparation(args)
        if data_collator is None:
            data_collator = CodeAVSDDataCollator(
                processing_class,
                max_length=max_prompt_length,
                views=self.views,
                student_enable_thinking=student_enable_thinking,
                teacher_enable_thinking=teacher_enable_thinking,
            )
        args.remove_unused_columns = False

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

        if getattr(args, "disable_dropout", False):
            disable_dropout_in_model(self.model)
        if self.fixed_teacher and peft_config is None and not is_peft_model(self.model):
            raise ValueError("fixed_teacher=True requires use_peft=True / a PEFT config.")

        self.temperature = getattr(args, "temperature", 1.0)
        self.use_vllm = getattr(args, "use_vllm", True)
        self.vllm_mode = getattr(args, "vllm_mode", "colocate")
        self.vllm_tensor_parallel_size = getattr(args, "vllm_tensor_parallel_size", 1)
        self.vllm_gpu_memory_utilization = getattr(args, "vllm_gpu_memory_utilization", 0.4)
        self.vllm_enable_sleep_mode = getattr(args, "vllm_enable_sleep_mode", False)
        self.vllm_guided_decoding_regex = getattr(args, "vllm_guided_decoding_regex", None)
        self.vllm_sync_frequency = max(1, int(getattr(args, "vllm_sync_frequency", 1)))
        self._last_vllm_sync_step = -1

        model_init_kwargs = getattr(args, "model_init_kwargs", None) or {}
        self.vllm_generator = CodeVLLMGenerator(
            model_name_or_path=self.model_name_or_path,
            tokenizer=self.processing_class,
            accelerator=self.accelerator,
            mode=self.vllm_mode,
            gpu_memory_utilization=self.vllm_gpu_memory_utilization,
            tensor_parallel_size=self.vllm_tensor_parallel_size,
            max_model_len=getattr(args, "max_length", None),
            max_num_seqs=int(getattr(args, "per_device_train_batch_size", 1))
            * int(getattr(args, "gradient_accumulation_steps", 1)),
            trust_remote_code=bool(model_init_kwargs.get("trust_remote_code", False)),
            enable_sleep_mode=self.vllm_enable_sleep_mode,
            seed=int(getattr(self.accelerator, "process_index", 0) or 0) // max(1, self.vllm_tensor_parallel_size),
        )
        self.vllm_sampling = CodeVLLMSamplingConfig(
            max_tokens=int(getattr(args, "max_completion_length", self.max_response_tokens)),
            temperature=float(getattr(args, "temperature", 0.7)),
            top_p=float(getattr(args, "top_p", 0.95)),
            top_k=int(getattr(args, "top_k", 0) or 0),
            min_p=float(getattr(args, "min_p", self.generation_min_p) or self.generation_min_p),
            presence_penalty=float(
                getattr(args, "presence_penalty", self.generation_presence_penalty)
                or self.generation_presence_penalty
            ),
            repetition_penalty=float(
                getattr(args, "repetition_penalty", self.generation_repetition_penalty)
                or self.generation_repetition_penalty
            ),
            n=1,
            guided_decoding_regex=self.vllm_guided_decoding_regex,
        )
        self.add_callback(CodeVLLMSyncCallback(self))

    def _teacher_context(self, model):
        if self.fixed_teacher and is_peft_model(model):
            return self.accelerator.unwrap_model(model).disable_adapter()
        return nullcontext()

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        del num_items_in_batch
        student_prompt_len = _as_int(inputs["student_prompt_length"])
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
        del outputs_student, student_logits, student_log_probs
        empty_cache()

        view_sample_log_probs = self._teacher_view_sampled_log_prob_pass(model, inputs, sampled_token_ids)
        if self.multi_view_mode == "single":
            target_sample_log_probs = view_sample_log_probs[:, 0, :]
            avsd_aux = {}
        else:
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
        self._record_code_loss_metrics(
            inputs,
            shifted_labels,
            student_sample_log_probs,
            view_sample_log_probs,
            target_sample_log_probs,
            avsd_aux,
        )

        del student_sample_log_probs, view_sample_log_probs, target_sample_log_probs
        empty_cache()

        if return_outputs:
            return loss, SimpleNamespace(loss=loss)
        return loss

    def _teacher_view_sampled_log_prob_pass(self, model, inputs, sampled_token_ids):
        view_samples = []
        for view in self.active_views:
            with torch.no_grad(), self._teacher_context(model):
                outputs = model(
                    input_ids=inputs[f"teacher_{view}_input_ids"],
                    attention_mask=inputs[f"teacher_{view}_attention_mask"],
                )
                teacher_prompt_len = _as_int(inputs[f"teacher_{view}_prompt_length"])
                teacher_logits = outputs.logits[:, teacher_prompt_len - 1 : -1, :]
                teacher_log_probs = F.log_softmax(teacher_logits / self.temperature, dim=-1)
                teacher_sample = torch.gather(
                    teacher_log_probs,
                    dim=-1,
                    index=sampled_token_ids.unsqueeze(-1),
                ).squeeze(-1)
                view_samples.append(teacher_sample)
                del outputs, teacher_logits, teacher_log_probs
                empty_cache()
        return torch.stack(view_samples, dim=1)

    @profiling_decorator
    def training_step(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:
        if not self.use_vllm:
            raise ValueError("CodeAVSDTrainer requires vLLM generation during training.")

        generated_ids, generated_attention_mask, prompt_texts, completion_texts, completion_lengths = (
            self._generate_on_policy_outputs_vllm(inputs)
        )
        student_prompt_len = _as_int(inputs["student_prompt_length"])
        generation_ids = generated_ids[:, student_prompt_len:]

        inputs["student_input_ids"] = generated_ids
        inputs["student_attention_mask"] = generated_attention_mask

        examples = [normalize_code_example(row) for row in inputs["raw_examples"]]
        feedback_texts, exec_results = self._build_feedbacks(examples, completion_texts)
        for view in self.active_views:
            prompt_ids, prompt_len = self._teacher_prompt_tensors(inputs, examples, view, feedback_texts)
            teacher_full_ids = torch.cat([prompt_ids, generation_ids], dim=1)
            teacher_attention_mask = torch.ones_like(teacher_full_ids)
            if self.processing_class.pad_token_id is not None:
                teacher_attention_mask[teacher_full_ids == self.processing_class.pad_token_id] = 0
            inputs[f"teacher_{view}_input_ids"] = teacher_full_ids
            inputs[f"teacher_{view}_attention_mask"] = teacher_attention_mask
            inputs[f"teacher_{view}_prompt_length"] = prompt_len

        labels = generated_ids.clone()
        for idx, actual_prompt_len in enumerate(inputs["student_prompt_lengths_per_example"]):
            labels[idx, : int(actual_prompt_len.item())] = -100
        if self.processing_class.pad_token_id is not None:
            labels[labels == self.processing_class.pad_token_id] = -100
        inputs["labels"] = labels
        inputs["_code_exec_results"] = exec_results
        inputs["_code_completion_lengths"] = completion_lengths
        inputs["_code_prompt_texts"] = prompt_texts
        inputs["_code_completion_texts"] = completion_texts

        return super().training_step(model, inputs, num_items_in_batch)

    def _generate_on_policy_outputs_vllm(self, inputs):
        import time

        device = self.accelerator.device
        start_time = time.time()
        completion_ids = self.vllm_generator.generate_from_prompt_ids(
            inputs["student_prompts"],
            self.vllm_sampling,
        )
        elapsed_time = time.time() - start_time
        total_completion_tokens = sum(len(ids) for ids in completion_ids)
        num_prompts = len(completion_ids)
        avg_completion_length = total_completion_tokens / num_prompts if num_prompts > 0 else 0
        tokens_per_sec = total_completion_tokens / elapsed_time if elapsed_time > 0 else 0
        print(
            "vLLM generation done - "
            f"elapsed time: {elapsed_time:.2f}s, "
            f"prompts: {num_prompts}, "
            f"total tokens: {total_completion_tokens}, "
            f"avg length: {avg_completion_length:.1f}, "
            f"speed: {tokens_per_sec:.1f} tok/s"
        )
        max_completion_length = int(getattr(self.args, "max_completion_length", self.max_response_tokens))
        pad_token_id = self.processing_class.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.processing_class.eos_token_id

        padded_completions = []
        completion_attention = []
        completion_lengths = []
        for ids in completion_ids:
            ids = list(ids[:max_completion_length])
            completion_lengths.append(len(ids))
            tensor = torch.tensor(ids, device=device, dtype=torch.long)
            if len(ids) < max_completion_length:
                pad = torch.full(
                    (max_completion_length - len(ids),),
                    pad_token_id,
                    device=device,
                    dtype=torch.long,
                )
                tensor = torch.cat([tensor, pad], dim=0)
            padded_completions.append(tensor)
            attn = torch.zeros(max_completion_length, device=device, dtype=torch.long)
            attn[: len(ids)] = 1
            completion_attention.append(attn)

        prompt_ids = inputs["student_prompts"].to(device)
        prompt_attention = inputs["student_prompt_attention_mask"].to(device)
        completion_tensor = torch.stack(padded_completions)
        completion_attention_tensor = torch.stack(completion_attention)
        generated_ids = torch.cat([prompt_ids, completion_tensor], dim=1)
        generated_attention_mask = torch.cat([prompt_attention, completion_attention_tensor], dim=1)
        prompt_texts = self.processing_class.batch_decode(inputs["student_prompts"], skip_special_tokens=False)
        completion_texts = [
            self.processing_class.decode(ids, skip_special_tokens=False)
            for ids in completion_ids
        ]
        return generated_ids, generated_attention_mask, prompt_texts, completion_texts, completion_lengths

    def _teacher_prompt_tensors(self, inputs, examples, view: str, feedbacks: list[str]):
        if view != "feedback" and f"teacher_{view}_prompts" in inputs:
            return inputs[f"teacher_{view}_prompts"].to(self.accelerator.device), _as_int(
                inputs[f"teacher_{view}_prompt_length"]
            )
        prompts = [
            build_teacher_prompt(
                self.processing_class,
                example,
                view,
                feedback=feedback,
                enable_thinking=self.teacher_enable_thinking,
            )
            for example, feedback in zip(examples, feedbacks)
        ]
        encoded = self.processing_class(
            prompts,
            padding=False,
            truncation=True,
            max_length=self.max_prompt_length,
        )
        prompt_len = max(len(ids) for ids in encoded["input_ids"])
        padded = self.processing_class(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=prompt_len,
            return_tensors="pt",
        )
        return padded["input_ids"].to(self.accelerator.device), prompt_len

    def _build_feedbacks(self, examples, completion_texts: list[str]) -> tuple[list[str], list[Any]]:
        feedbacks = []
        results = []
        for example, completion in zip(examples, completion_texts):
            code = extract_python_code(completion)
            result = run_code_on_tests(
                code,
                list(example.public_tests),
                timeout_s=self.execution_timeout_s,
                unsafe_subprocess_sandbox=self.unsafe_subprocess_sandbox,
            )
            feedbacks.append(build_feedback(result))
            results.append(result)
        return feedbacks, results

    def _record_code_loss_metrics(
        self,
        inputs,
        shifted_labels,
        student_sample_log_probs,
        view_sample_log_probs,
        target_sample_log_probs,
        avsd_aux,
    ) -> None:
        exec_results = inputs.get("_code_exec_results") or []
        completion_lengths = inputs.get("_code_completion_lengths") or []
        if exec_results:
            statuses = Counter(result.status for result in exec_results)
            total = max(1, len(exec_results))
            self._code_metric_buffer["code_public_pass_rate"].append(statuses.get("passed", 0) / total)
            for status in ("wrong_answer", "runtime_error", "syntax_error", "timeout", "extraction_failure"):
                self._code_metric_buffer[f"code_{status}_rate"].append(statuses.get(status, 0) / total)
        if completion_lengths:
            self._code_metric_buffer["code_avg_completion_tokens"].append(
                sum(completion_lengths) / max(1, len(completion_lengths))
            )
        mask = shifted_labels != -100
        if torch.any(mask):
            advantage = (target_sample_log_probs - student_sample_log_probs).detach()
            self._code_metric_buffer["code_advantage_mean"].append(float(advantage[mask].float().mean()))
            self._code_metric_buffer["code_teacher_logp_mean"].append(
                float(view_sample_log_probs.mean(dim=1)[mask].detach().float().mean())
            )
        gate = avsd_aux.get("gate") if isinstance(avsd_aux, dict) else None
        if gate is not None and torch.any(mask):
            self._code_metric_buffer["avsd.code_gate_mean"].append(float(gate[mask].detach().float().mean()))

    def _move_model_to_vllm(self) -> None:
        self.vllm_generator.sync_model_weights(
            self.model,
            accelerator=self.accelerator,
            is_fsdp_enabled=getattr(self, "is_fsdp_enabled", False),
        )

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        for key, values in list(self._code_metric_buffer.items()):
            if values:
                logs[key] = sum(values) / len(values)
        self._code_metric_buffer.clear()
        super().log(logs, start_time)


def _as_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


def _normalize_multi_view_mode(mode: str) -> str:
    return str(mode).strip().lower()


def _normalize_avsd_gate_mode(mode: str) -> str:
    return str(mode).strip().lower()
