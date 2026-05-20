#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

RUN_CONFIG="${RUN_CONFIG:-deepseek_r1_distill_qwen_7b_avsd}"
EXTRA_ARGS=("$@")

cmd=(
    accelerate launch
    --config_file configs/accelerate.yaml
    --num_processes 4
    --gradient_accumulation_steps 2
    --main_process_port 12949
    -m avsd.math.train
    --model_name_or_path deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
    --training_dataset openthought
    --learning_rate 5e-6
    --max_grad_norm 0.1
    --per_device_train_batch_size 4
    --gradient_checkpointing
    --gradient_accumulation_steps 2
    --output_dir outputs/avsd/deepseek_r1_distill_qwen_7b
    --run_config "${RUN_CONFIG}"
    --max_steps 500
    --max_completion_length 4096
    --save_steps 10
    --logging_steps 2
    --attn_implementation flash_attention_2
    --torch_dtype bfloat16
    --max_length 16384
    --use_vllm
    --vllm_mode colocate
    --vllm_gpu_memory_utilization 0.4
    --vllm_tensor_parallel_size 1
    --use_peft
    --lora_r 64
    --lora_alpha 128
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --temperature 0.6
    --top_p 0.95
    --top_k 0
    --fixed_teacher
    --use_tinker_loss
    --multi_view_mode avsd
    --pi_views full_solution,partial_solution,answer_only
    --partial_solution_ratio 0.5
    --avsd_gate_mode avsd
)

if ((${#EXTRA_ARGS[@]} > 0)); then
    cmd+=("${EXTRA_ARGS[@]}")
fi

echo "=== Running ${RUN_CONFIG} ==="
(
    cd "${REPO_ROOT}"
    "${cmd[@]}"
)
