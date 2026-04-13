#!/bin/bash

# Prepare script for Moonlight-16B-A3B-Instruct: download model/datasets and convert checkpoint.
#
# Usage:
#   bash prepare-moonlight-16B.sh

set -ex

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME="Moonlight-16B-A3B-Instruct"
MODEL_TYPE="moonlight"
HF_REPO="moonshotai/Moonlight-16B-A3B-Instruct"
GPUS_PER_NODE=8

# ---------------------------------------------------------------------------
# Download model and datasets
# ---------------------------------------------------------------------------
mkdir -p /root/models /root/datasets
hf download "$HF_REPO" --local-dir "/root/models/${MODEL_NAME}"

python3 -c "
from miles.utils.external_utils.command_utils import hf_download_dataset
hf_download_dataset('zhuzilin/dapo-math-17k')
"

# ---------------------------------------------------------------------------
# Convert checkpoint
# ---------------------------------------------------------------------------
mkdir -p /root/multinode
python3 -c "
from miles.utils.external_utils.command_utils import convert_checkpoint
convert_checkpoint(
    model_name='${MODEL_NAME}',
    megatron_model_type='${MODEL_TYPE}',
    num_gpus_per_node=${GPUS_PER_NODE},
    dir_dst='/root/multinode',
)
"

echo "Prepare done."
