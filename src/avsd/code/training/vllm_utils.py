import importlib
import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


def require_vllm():
    try:
        return importlib.import_module("vllm")
    except ImportError as exc:
        raise ImportError(
            "vLLM is required for code-domain generation. Install vLLM or run in the opsd environment; "
            "there is intentionally no transformers.generate fallback."
        ) from exc


def require_vllm_lora_request():
    try:
        module = importlib.import_module("vllm.lora.request")
        return module.LoRARequest
    except ImportError as exc:
        raise ImportError("vLLM LoRA evaluation requires vllm.lora.request.LoRARequest.") from exc


@dataclass(frozen=True)
class CodeVLLMSamplingConfig:
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 0
    min_p: float = 0.0
    presence_penalty: float = 0.0
    repetition_penalty: float = 1.0
    n: int = 1
    guided_decoding_regex: str | None = None


def make_sampling_params(config: CodeVLLMSamplingConfig):
    vllm = require_vllm()
    top_k = int(config.top_k) if config.top_k and config.top_k > 0 else -1
    guided_decoding = None
    if config.guided_decoding_regex:
        guided_module = importlib.import_module("vllm.sampling_params")
        guided_decoding = guided_module.GuidedDecodingParams(
            backend="outlines",
            regex=config.guided_decoding_regex,
        )
    return vllm.SamplingParams(
        n=config.n,
        temperature=config.temperature,
        top_p=config.top_p,
        top_k=top_k,
        min_p=config.min_p,
        max_tokens=config.max_tokens,
        presence_penalty=config.presence_penalty,
        repetition_penalty=config.repetition_penalty,
        guided_decoding=guided_decoding,
    )


def decode_prompt_batch(tokenizer, prompt_ids: torch.Tensor) -> list[str]:
    prompts = tokenizer.batch_decode(prompt_ids, skip_special_tokens=False)
    pad_token = getattr(tokenizer, "pad_token", None)
    if pad_token:
        prompts = [prompt.replace(pad_token, "") for prompt in prompts]
    return prompts


def load_vllm_model_for_eval(
    *,
    model_name_or_path: str,
    tokenizer_name_or_path: str | None = None,
    checkpoint_dir: str | None = None,
    trust_remote_code: bool = False,
    gpu_memory_utilization: float = 0.9,
    tensor_parallel_size: int = 1,
    max_model_len: int | None = None,
    enable_lora: bool = False,
    max_lora_rank: int = 64,
    distributed_executor_backend: str = "mp",
    enforce_eager: bool = True,
) -> tuple[Any, Any, Any | None]:
    vllm = require_vllm()
    from transformers import AutoTokenizer

    llm_config: dict[str, Any] = {
        "model": model_name_or_path,
        "gpu_memory_utilization": gpu_memory_utilization,
        "tensor_parallel_size": tensor_parallel_size,
        "trust_remote_code": trust_remote_code,
        "distributed_executor_backend": distributed_executor_backend,
        "enforce_eager": enforce_eager,
    }
    if max_model_len is not None:
        llm_config["max_model_len"] = max_model_len

    lora_request = None
    if checkpoint_dir is not None:
        has_adapter = (Path(checkpoint_dir) / "adapter_model.safetensors").exists() or (
            Path(checkpoint_dir) / "adapter_model.bin"
        ).exists()
        if has_adapter:
            llm_config.update(
                {
                    "enable_lora": True,
                    "max_lora_rank": max_lora_rank,
                    "max_loras": 1,
                    "max_cpu_loras": 1,
                }
            )
            LoRARequest = require_vllm_lora_request()
            lora_request = LoRARequest("code_avsd_lora", 1, checkpoint_dir)
        elif enable_lora:
            raise FileNotFoundError(f"No LoRA adapter weights found in checkpoint_dir={checkpoint_dir}")
    elif enable_lora:
        llm_config.update(
            {
                "enable_lora": True,
                "max_lora_rank": max_lora_rank,
                "max_loras": 1,
                "max_cpu_loras": 1,
            }
        )

    llm = vllm.LLM(**llm_config)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name_or_path or model_name_or_path,
        trust_remote_code=trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return llm, tokenizer, lora_request


class CodeVLLMGenerator:
    def __init__(
        self,
        *,
        model_name_or_path: str,
        tokenizer,
        accelerator=None,
        mode: str = "colocate",
        gpu_memory_utilization: float = 0.4,
        tensor_parallel_size: int = 1,
        max_model_len: int | None = None,
        max_num_seqs: int | None = None,
        trust_remote_code: bool = False,
        enable_sleep_mode: bool = False,
        seed: int = 0,
    ) -> None:
        if mode != "colocate":
            raise ValueError("CodeAVSDTrainer supports vllm_mode='colocate' only.")
        self.vllm = require_vllm()
        self.tokenizer = tokenizer
        self.accelerator = accelerator
        self.mode = mode
        self.tensor_parallel_size = int(tensor_parallel_size)
        self.enable_sleep_mode = enable_sleep_mode
        self.vllm_tp_group = None

        llm_config: dict[str, Any] = {
            "model": model_name_or_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization,
            "trust_remote_code": trust_remote_code,
            "seed": seed,
            "enable_sleep_mode": enable_sleep_mode,
        }
        if max_model_len is not None:
            llm_config["max_model_len"] = max_model_len
        if max_num_seqs is not None:
            llm_config["max_num_seqs"] = max_num_seqs

        if accelerator is not None:
            if accelerator.num_processes % self.tensor_parallel_size != 0:
                raise ValueError(
                    f"vllm_tensor_parallel_size ({self.tensor_parallel_size}) must divide world size "
                    f"({accelerator.num_processes}) evenly."
                )
            self._prepare_external_launcher(accelerator)
            llm_config["distributed_executor_backend"] = "external_launcher"
            llm_config["seed"] = accelerator.process_index // max(1, self.tensor_parallel_size)
        else:
            llm_config["distributed_executor_backend"] = "mp"

        self.engine = self.vllm.LLM(**llm_config)
        if self.enable_sleep_mode:
            self.engine.sleep(level=2)
        if accelerator is not None:
            accelerator.wait_for_everyone()

    def generate_from_prompt_ids(
        self,
        prompt_ids: torch.Tensor,
        sampling: CodeVLLMSamplingConfig,
    ) -> list[list[int]]:
        prompts = decode_prompt_batch(self.tokenizer, prompt_ids)
        return self.generate_from_text(prompts, sampling)

    def generate_from_text(
        self,
        prompts: list[str],
        sampling: CodeVLLMSamplingConfig,
        *,
        lora_request=None,
        use_tqdm: bool = False,
    ) -> list[list[int]]:
        if self.enable_sleep_mode:
            self.engine.wake_up(tags=["kv_cache"])

        orig_size = len(prompts)
        prompts_for_generation = prompts
        if self.vllm_tp_group is not None:
            gathered_prompts = [None for _ in range(self.tensor_parallel_size)]
            torch.distributed.all_gather_object(gathered_prompts, prompts, group=self.vllm_tp_group)
            prompts_for_generation = [prompt for group_prompts in gathered_prompts for prompt in group_prompts]

        generate_kwargs = {
            "sampling_params": make_sampling_params(sampling),
            "use_tqdm": use_tqdm,
        }
        if lora_request is not None:
            generate_kwargs["lora_request"] = lora_request
        outputs = self.engine.generate(prompts_for_generation, **generate_kwargs)
        completion_ids = [list(output.token_ids) for request_output in outputs for output in request_output.outputs]

        if self.vllm_tp_group is not None:
            local_rank = torch.distributed.get_rank(group=self.vllm_tp_group)
            completion_ids = completion_ids[local_rank * orig_size : (local_rank + 1) * orig_size]

        if self.enable_sleep_mode:
            self.engine.sleep(level=2)
        return completion_ids

    def sync_model_weights(self, model, *, accelerator=None, is_fsdp_enabled: bool = False) -> None:
        accelerator = accelerator or self.accelerator
        if self.enable_sleep_mode:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.engine.wake_up(tags=["weights"])
        _sync_model_weights_to_vllm(
            self.engine,
            model,
            accelerator=accelerator,
            is_fsdp_enabled=is_fsdp_enabled,
        )
        self.engine.reset_prefix_cache()

    def _prepare_external_launcher(self, accelerator) -> None:
        os.environ["RANK"] = str(accelerator.process_index)
        os.environ["LOCAL_RANK"] = str(accelerator.local_process_index)
        os.environ["WORLD_SIZE"] = str(accelerator.num_processes)
        try:
            from trl.trainer.utils import ensure_master_addr_port

            ensure_master_addr_port()
        except Exception:
            pass
        if self.tensor_parallel_size > 1:
            groups = [
                list(range(i * self.tensor_parallel_size, (i + 1) * self.tensor_parallel_size))
                for i in range(accelerator.num_processes // self.tensor_parallel_size)
            ]
            self.vllm_tp_group, _ = torch.distributed.new_subgroups_by_enumeration(groups)


def _sync_model_weights_to_vllm(engine, model, *, accelerator=None, is_fsdp_enabled: bool = False) -> None:
    from accelerate.utils import is_peft_model

    llm_model = engine.llm_engine.model_executor.driver_worker.model_runner.model
    zero_stage_3 = False
    if accelerator is not None:
        deepspeed_plugin = getattr(accelerator.state, "deepspeed_plugin", None)
        zero_stage_3 = deepspeed_plugin is not None and getattr(deepspeed_plugin, "zero_stage", None) == 3
    gather_if_zero3 = nullcontext
    if zero_stage_3:
        import deepspeed

        gather_if_zero3 = deepspeed.zero.GatheredParameters

    if is_peft_model(model):
        with gather_if_zero3(list(model.parameters())):
            model.merge_adapter()
            try:
                for name, param in model.named_parameters():
                    clean_name = name.removeprefix("base_model.model.").replace(".base_layer", "")
                    prefix = getattr(model, "prefix", None)
                    if prefix and prefix in clean_name:
                        continue
                    if "original_module" in clean_name:
                        continue
                    clean_name = clean_name.replace("modules_to_save.default.", "")
                    llm_model.load_weights([(clean_name, param.data)])
            finally:
                model.unmerge_adapter()
        return

    if is_fsdp_enabled:
        _sync_fsdp_params_to_vllm(llm_model, model)
        return

    for name, param in model.named_parameters():
        with gather_if_zero3([param]):
            llm_model.load_weights([(name, param.data)])


def _sync_fsdp_params_to_vllm(llm_model, module, prefix: str = "", visited=None) -> None:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    if visited is None:
        visited = set()
    for child_name, child_module in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        _sync_fsdp_params_to_vllm(llm_model, child_module, child_prefix, visited)
    if isinstance(module, FSDP):
        with FSDP.summon_full_params(module, recurse=False, writeback=False):
            for param_name, param in module.named_parameters():
                full_name = f"{prefix}.{param_name}" if prefix else param_name
                for extra in ("_fsdp_wrapped_module.", "_checkpoint_wrapped_module."):
                    full_name = full_name.replace(extra, "")
                if full_name in visited:
                    continue
                visited.add(full_name)
                llm_model.load_weights([(full_name, param.data)])
