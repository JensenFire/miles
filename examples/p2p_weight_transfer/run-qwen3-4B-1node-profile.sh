#!/bin/bash

# Single-node profiling script for Qwen3-4B with p2p weight transfer.
# Disaggregated on 1 node: 4 GPUs train + 4 GPUs rollout.
#
# Usage:
#   bash run-qwen3-4B-1node-profile.sh <MODE>
#
#   MODE : broadcast | p2p

set -ex

export PYTHONBUFFERED=16

# ---------------------------------------------------------------------------
# Positional arguments
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <MODE>"
    echo "  MODE : broadcast | p2p"
    exit 1
fi

MODE="$1"

# ---------------------------------------------------------------------------
# Cleanup stale processes
# ---------------------------------------------------------------------------
pkill -9 sglang || true
sleep 3
ray stop --force || true
pkill -9 ray || true
pkill -9 python || true
sleep 3
pkill -9 ray || true
pkill -9 python || true
pkill -9 redis || true

# ---------------------------------------------------------------------------
# Fixed config
# ---------------------------------------------------------------------------
GPUS_PER_NODE=8
NUM_TRAIN_GPUS=4
NUM_ROLLOUT_GPUS=4
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"
BUCKET_SIZE_GB="${BUCKET_SIZE_GB:-1.0}"

NUM_TRAIN_NODES=1

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME="Qwen3-4B"
MODEL_TYPE="qwen3-4B"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
MILES_ROOT="/root/miles"
source "${MILES_ROOT}/scripts/models/${MODEL_TYPE}.sh"

# ---------------------------------------------------------------------------
# Execute
# ---------------------------------------------------------------------------
run_mode() {
    local mode="$1"

    # --- Checkpoint ---
    CKPT_ARGS=(
        --hf-checkpoint "/root/models/${MODEL_NAME}/"
        --ref-load "/root/multinode/${MODEL_NAME}_torch_dist/"
    )

    # --- Rollout ---
    ROLLOUT_ARGS=(
        --prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl
        --input-key prompt
        --label-key label
        --apply-chat-template
        --rollout-shuffle
        --rm-type deepscaler
        --num-rollout 13
        --rollout-batch-size 8
        --n-samples-per-prompt 8
        --rollout-max-response-len 100
        --rollout-temperature 0.8
        --global-batch-size 32
        --balance-data
    )

    # --- Training parallelism ---
    PERF_ARGS=(
        --tensor-model-parallel-size 2
        --sequence-parallel
        --pipeline-model-parallel-size 1
        --context-parallel-size 2
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
        --use-dynamic-batch-size
        --max-tokens-per-gpu 2048
    )

    # --- GRPO ---
    GRPO_ARGS=(
        --advantage-estimator grpo
        --kl-loss-coef 0.00
        --kl-loss-type low_var_kl
        --entropy-coef 0.00
        --eps-clip 0.2
        --eps-clip-high 0.28
    )

    # --- Optimizer ---
    OPTIMIZER_ARGS=(
        --optimizer adam
        --lr 1e-6
        --lr-decay-style constant
        --weight-decay 0.1
        --adam-beta1 0.9
        --adam-beta2 0.98
    )

    # --- SGLang: 2 engines x 2 GPUs ---
    SGLANG_ARGS=(
        --rollout-num-gpus-per-engine 2
        --rollout-num-gpus ${NUM_ROLLOUT_GPUS}
        --sglang-mem-fraction-static 0.8
    )
    if [ "$mode" = "p2p" ]; then
        SGLANG_ARGS+=(--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine)
    fi
    if [ "$SKIP_VALIDATION" -eq 1 ]; then
        SGLANG_ARGS+=(--sglang-load-format dummy)
    fi

    # --- Misc ---
    BUFFER_SIZE=$((1 * 1024 * 1024 * 1024))

    MISC_ARGS=(
        --attention-dropout 0.0
        --hidden-dropout 0.0
        --accumulate-allreduce-grads-in-fp32
        --attention-softmax-in-fp32
        --attention-backend flash
        --actor-num-nodes ${NUM_TRAIN_NODES}
        --actor-num-gpus-per-node ${NUM_TRAIN_GPUS}
        --update-weight-buffer-size ${BUFFER_SIZE}
        --update-weight-transfer-mode ${mode}
    )
    if [ "$SKIP_VALIDATION" -eq 0 ]; then
        MISC_ARGS+=(--check-weight-update-equal)
    fi

    # --- Launch Ray (single node) ---
    ray start --head --num-gpus ${GPUS_PER_NODE} \
        --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265

    # --- Build runtime env JSON ---
    RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"RAY_DEBUG\": \"1\",
    \"PYTHONPATH\": \"/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"0\",
    \"MILES_LOG_DIR\": \"${MILES_LOG_DIR:-}\"
  }
}"

    # --- Submit Ray job ---
    ray job submit --address="http://127.0.0.1:8265" \
        --runtime-env-json="${RUNTIME_ENV_JSON}" \
        -- python3 train.py \
        ${MODEL_ARGS[@]} \
        ${CKPT_ARGS[@]} \
        ${ROLLOUT_ARGS[@]} \
        ${OPTIMIZER_ARGS[@]} \
        ${GRPO_ARGS[@]} \
        ${PERF_ARGS[@]} \
        ${SGLANG_ARGS[@]} \
        ${MISC_ARGS[@]}
}

echo ""
echo "============================================================"
echo "  Running: Qwen3-4B / ${MODE}"
echo "============================================================"
echo ""

run_mode "$MODE"

echo "Done."
