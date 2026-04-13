#!/bin/bash

# Prepare script for GLM-Z1-9B-0414: download model/datasets and convert checkpoint.
#
# Usage:
#   bash prepare-glm-z1-9B.sh

set -ex

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
MODEL_NAME="GLM-Z1-9B-0414"
MODEL_TYPE="glm4-9B"
HF_REPO="zai-org/GLM-Z1-9B-0414"
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
