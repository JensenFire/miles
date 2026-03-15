#!/usr/bin/env python3
"""
Test rfork (seed) loading for rollout engine initialization.

Simulates a multi-node rollout setup on a single machine using different GPU subsets.
Validates that rollout engines correctly load weights via RFork (seed) loading, with 
one engine loading from disk and the other loading from the seed engine via transfer.

Uses --debug-rollout-only to only start sglang rollout engines (skip training).

Requirements: 2 GPUs on a single machine

Usage:
    python test_rfork_loading.py                 # prepare + run
    python test_rfork_loading.py --skip-prepare  # skip model download
"""

import os

import miles.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3 "
        "--rollout-batch-size 16 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1 "
        "--over-sampling-batch-size 32 "
        "--global-batch-size 64 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
    )

    sglang_args = (
        "--rollout-num-gpus-per-engine 1 "
        "--num-gpus-per-node 1 "
        "--sglang-remote-instance-weight-loader-start-seed-via-transfer-engine "
        "--sglang-mem-fraction-static 0.8 "
    )

    misc_args = (
        "--rollout-num-gpus 2 "
        "--debug-rollout-only "
        "--train-backend fsdp "
    )

    ci_args = "--ci-test "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{sglang_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=2,
        megatron_model_type=None,
        extra_env_vars={"MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1"},
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-prepare", action="store_true", help="Skip model/dataset download")
    args = parser.parse_args()

    if not args.skip_prepare:
        prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
