#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

TRAIN_FILE="${TRAIN_FILE:-data/codeforces_cots_py_train_views.jsonl}"
RUN_CONFIG="${RUN_CONFIG:-qwen3_8b_code_avsd}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
MAX_COMPLETION_LENGTH="${MAX_COMPLETION_LENGTH:-4096}"
EXTRA_ARGS=("$@")

cmd=(
    accelerate launch
    --config_file configs/accelerate.yaml
    --num_processes "${NUM_PROCESSES:-4}"
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --main_process_port "${MAIN_PROCESS_PORT:-12949}"
    -m avsd.code.train
    --train_file "${TRAIN_FILE}"
    --model_name_or_path "${MODEL_NAME_OR_PATH:-Qwen/Qwen3-8B}"
    --learning_rate "${LEARNING_RATE:-5e-6}"
    --max_grad_norm "${MAX_GRAD_NORM:-0.1}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
    --gradient_checkpointing
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --output_dir "${OUTPUT_DIR:-outputs/code_avsd/qwen3_8b}"
    --run_config "${RUN_CONFIG}"
    --max_steps "${MAX_STEPS:-500}"
    --max_completion_length "${MAX_COMPLETION_LENGTH}"
    --save_steps "${SAVE_STEPS:-10}"
    --logging_steps "${LOGGING_STEPS:-2}"
    --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}"
    --torch_dtype "${TORCH_DTYPE:-bfloat16}"
    --max_length "${MAX_LENGTH:-16384}"
    --views reference,hint,feedback
    --multi_view_mode avsd
    --avsd_gate_mode avsd
    --use_vllm
    --vllm_mode colocate
    --vllm_gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
    --vllm_tensor_parallel_size "${VLLM_TENSOR_PARALLEL_SIZE:-1}"
    --vllm_sync_frequency "${VLLM_SYNC_FREQUENCY:-1}"
    --use_peft
    --lora_r "${LORA_R:-64}"
    --lora_alpha "${LORA_ALPHA:-128}"
    --lora_target_modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
    --temperature "${TEMPERATURE:-0.7}"
    --top_p "${TOP_P:-0.95}"
    --top_k "${TOP_K:-20}"
    --fixed_teacher
    --use_tinker_loss
)

if [[ "${STUDENT_ENABLE_THINKING:-0}" == "1" ]]; then
    cmd+=(--student_enable_thinking)
fi

if [[ "${TEACHER_ENABLE_THINKING:-1}" == "0" ]]; then
    cmd+=(--no_teacher_enable_thinking)
fi

cmd+=("${EXTRA_ARGS[@]}")
(
    cd "${REPO_ROOT}"
    "${cmd[@]}"
)

