#!/bin/bash

# Multi-node (2-node) profiling script for Moonlight-16B-A3B with p2p weight transfer.
# Node 0 = training (8 GPUs), Node 1 = rollout (8 GPUs).
#
# Usage:
#   bash run-moonlight-16B-2node-profile.sh <MODE> <NODE_RANK> <HEAD_NODE_IP>
#
#   MODE          : broadcast | p2p
#   NODE_RANK     : 0 (head node) | 1 (worker node)
#   HEAD_NODE_IP  : IP address of the head node

set -ex

export PYTHONBUFFERED=16

# ---------------------------------------------------------------------------
# Positional arguments
# ---------------------------------------------------------------------------
if [ $# -lt 3 ]; then
    echo "Usage: $0 <MODE> <NODE_RANK> <HEAD_NODE_IP>"
    echo "  MODE         : broadcast | p2p"
    echo "  NODE_RANK    : 0 (head) | 1 (worker)"
    echo "  HEAD_NODE_IP : IP of the head node"
    exit 1
fi

MODE="$1"
NODE_RANK="$2"
HEAD_NODE_IP="$3"

# ---------------------------------------------------------------------------
# Cleanup stale processes (ALL nodes in container env to avoid conflicts)
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
NNODES=2
GPUS_PER_NODE=8
NUM_TRAIN_GPUS=8      # 1 node
NUM_ROLLOUT_GPUS=8    # 1 node
SKIP_VALIDATION="${SKIP_VALIDATION:-0}"
BUCKET_SIZE_GB="${BUCKET_SIZE_GB:-1.0}"

NUM_TRAIN_NODES=$((NUM_TRAIN_GPUS / GPUS_PER_NODE))

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME="Moonlight-16B-A3B-Instruct"
MODEL_TYPE="moonlight"

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
        --rollout-batch-size 4
        --n-samples-per-prompt 4
        --rollout-max-response-len 100
        --rollout-temperature 1.0
        --global-batch-size 16
        --balance-data
    )

    # --- Training parallelism (TP=2, CP=1, EP=8, 8 GPUs) ---
    PERF_ARGS=(
        --tensor-model-parallel-size 2
        --sequence-parallel
        --pipeline-model-parallel-size 1
        --context-parallel-size 1
        --expert-model-parallel-size 8
        --expert-tensor-parallel-size 1
        --recompute-granularity full
        --recompute-method uniform
        --recompute-num-layers 1
        --use-dynamic-batch-size
        --max-tokens-per-gpu 2048
    )

    # --- GRPO ---
    GRPO_ARGS=(
        --advantage-estimator grpo
        --use-kl-loss
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

    # --- SGLang: 1 engine x 8 GPUs (WS=8, EP=8, DP attention) ---
    SGLANG_ARGS=(
        --rollout-num-gpus-per-engine 8
        --rollout-num-gpus ${NUM_ROLLOUT_GPUS}
        --sglang-mem-fraction-static 0.7
        --sglang-ep-size 8
        --sglang-cuda-graph-bs 1 2 4 8 16
        --sglang-enable-dp-attention
        --sglang-enable-dp-lm-head
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
        --actor-num-gpus-per-node ${GPUS_PER_NODE}
        --update-weight-buffer-size ${BUFFER_SIZE}
        --update-weight-transfer-mode ${mode}
    )
    if [ "$SKIP_VALIDATION" -eq 0 ]; then
        MISC_ARGS+=(--check-weight-update-equal)
    fi

    # --- Worker nodes sleep to let head node start first ---
    if [ "$NODE_RANK" -gt 0 ]; then
        sleep 20
    fi

    # --- Launch Ray ---
    if [ "$NODE_RANK" -eq 0 ]; then
        RAY_memory_monitor_refresh_ms=0 \
        ray start --head --node-ip-address "${HEAD_NODE_IP}" --num-gpus ${GPUS_PER_NODE} \
            --disable-usage-stats --dashboard-host=0.0.0.0 --dashboard-port=8265
    else
        RAY_memory_monitor_refresh_ms=0 \
        ray start --address="${HEAD_NODE_IP}:6379" --num-gpus ${GPUS_PER_NODE} --disable-usage-stats
    fi

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

    # --- Wait for all nodes to join Ray cluster (head node only) ---
    EXPECTED_GPUS=$((NNODES * GPUS_PER_NODE))
    if [ "$NODE_RANK" -eq 0 ]; then
        echo "Waiting for ${EXPECTED_GPUS} GPUs in Ray cluster..."
        while true; do
            AVAILABLE_GPUS=$(python3 -c "import ray; ray.init(address='auto', ignore_reinit_error=True); print(int(ray.cluster_resources().get('GPU', 0))); ray.shutdown()" 2>/dev/null || echo 0)
            echo "  ... detected ${AVAILABLE_GPUS}/${EXPECTED_GPUS} GPUs"
            if [ "$AVAILABLE_GPUS" -ge "$EXPECTED_GPUS" ]; then
                break
            fi
            sleep 5
        done
        echo "All ${EXPECTED_GPUS} GPUs available. Submitting job."
    fi

    # --- Signal file for worker synchronization (container env) ---
    # In container environments, worker nodes must stay alive while the
    # head node runs the Ray job. We use a signal file on shared storage.
    SIGNAL_DIR="${MILES_LOG_DIR:-/data/ray/signals}"
    mkdir -p "${SIGNAL_DIR}"
    DONE_FILE="${SIGNAL_DIR}/job_done_${mode}"

    # Clean up any stale signal file
    rm -f "${DONE_FILE}"

    # --- Submit Ray job (head node only) ---
    if [ "$NODE_RANK" -eq 0 ]; then
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
        JOB_EXIT=$?
        # Signal workers that the job is done
        echo "${JOB_EXIT}" > "${DONE_FILE}"
    else
        # Worker nodes: block until head node signals completion.
        # In container environments (pyxis/enroot), exiting kills the container
        # and the Ray worker with it, so we must stay alive.
        echo "Worker node ${NODE_RANK}: Ray joined, waiting for head to finish..."
        while [ ! -f "${DONE_FILE}" ]; do
            sleep 10
        done
        echo "Worker node ${NODE_RANK}: head finished (exit=$(cat "${DONE_FILE}")), exiting."
    fi
}

echo ""
echo "============================================================"
echo "  Running: Moonlight-16B-A3B / ${MODE}"
echo "============================================================"
echo ""

run_mode "$MODE"

echo "Done."
