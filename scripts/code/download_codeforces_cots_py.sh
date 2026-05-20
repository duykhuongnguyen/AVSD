#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

cmd=(
    python -m avsd.code.download_codeforces_cots_py
    --train-output "${TRAIN_OUTPUT:-data/codeforces_cots_py_train.jsonl}"
    --eval-output "${EVAL_OUTPUT:-data/codeforces_cots_py_eval.jsonl}"
    --train-size "${TRAIN_SIZE:-1000}"
    --eval-size "${EVAL_SIZE:-100}"
    --seed "${SEED:-1337}"
    --verify-reference "${VERIFY_REFERENCE:-public}"
    --timeout-s "${TIMEOUT_S:-2.0}"
)

if [[ -n "${HF_CACHE_DIR:-}" ]]; then
    cmd+=(--cache-dir "$HF_CACHE_DIR")
fi

if [[ "${UNSAFE_SUBPROCESS_SANDBOX:-0}" == "1" ]]; then
    cmd+=(--unsafe-subprocess-sandbox)
fi

if [[ "${DISABLE_PROGRESS:-0}" == "1" ]]; then
    cmd+=(--disable-progress)
fi

cmd+=("$@")
(
    cd "${REPO_ROOT}"
    "${cmd[@]}"
)

