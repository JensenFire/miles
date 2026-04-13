#!/bin/bash

# Unified prepare script for all GLM-5 variants:
#   GLM-5_4layer   - 4-layer pruned (3 dense + 1 MoE)
#   GLM-5_20layer  - 20-layer pruned (3 dense + 17 MoE)
#   GLM-5          - full 744B model (3 dense + 75 MoE)
#
# Downloads HF model, patches config.json for DeepseekV32, and converts to
# megatron format.
#
# Usage:
#   bash prepare-glm5.sh <MODEL_NAME> [--download-only]
#
# MODEL_NAME:
#   GLM-5_4layer   (Pinaster/GLM-5_4layer)
#   GLM-5_20layer  (Pinaster/GLM-5_20layer)
#   GLM-5          (zai-org/GLM-5)
#
# Options:
#   --download-only  Skip checkpoint conversion (for rollout/worker nodes)

set -ex

# ---------------------------------------------------------------------------
# Parse arguments
# ---------------------------------------------------------------------------
if [ $# -lt 1 ]; then
    echo "Usage: $0 <MODEL_NAME> [--download-only]"
    echo "  MODEL_NAME : GLM-5_4layer | GLM-5_20layer | GLM-5"
    exit 1
fi

MODEL_NAME="$1"
shift

DOWNLOAD_ONLY=0
for arg in "$@"; do
    if [ "$arg" = "--download-only" ]; then
        DOWNLOAD_ONLY=1
    fi
done

# ---------------------------------------------------------------------------
# Resolve model config from MODEL_NAME
# ---------------------------------------------------------------------------
case "${MODEL_NAME}" in
    GLM-5_4layer)
        HF_REPO="Pinaster/GLM-5_4layer"
        MODEL_TYPE="glm5-744B-A40B_4layer"
        CONVERT_GPUS=4
        CONVERT_EXTRA_ARGS="--pipeline-model-parallel-size 1 --expert-model-parallel-size 1 --tensor-model-parallel-size 1 --expert-tensor-parallel-size 1"
        CONVERT_MULTINODE="false"
        CONVERT_NUM_NODES=""
        ;;
    GLM-5_20layer)
        HF_REPO="Pinaster/GLM-5_20layer"
        MODEL_TYPE="glm5-744B-A40B_20layer"
        CONVERT_GPUS=8
        CONVERT_EXTRA_ARGS="--tensor-model-parallel-size 1 --expert-tensor-parallel-size 1 --expert-model-parallel-size 4"
        CONVERT_MULTINODE="true"
        CONVERT_NUM_NODES="2"
        ;;
    GLM-5)
        HF_REPO="zai-org/GLM-5"
        MODEL_TYPE="glm5-744B-A40B"
        CONVERT_GPUS=8
        CONVERT_EXTRA_ARGS="--pipeline-model-parallel-size 4 --expert-model-parallel-size 32 --tensor-model-parallel-size 1 --expert-tensor-parallel-size 1 --decoder-last-pipeline-num-layers 18"
        CONVERT_MULTINODE="true"
        CONVERT_NUM_NODES=""
        ;;
    *)
        echo "ERROR: Unknown model '${MODEL_NAME}'. Use GLM-5_4layer, GLM-5_20layer, or GLM-5."
        exit 1
        ;;
esac

# Checkpoint save directory (override with CKPT_SAVE_DIR for shared storage)
CKPT_SAVE_DIR="${CKPT_SAVE_DIR:-/root}"

echo "=== Preparing ${MODEL_NAME} ==="
echo "HF repo    : ${HF_REPO}"
echo "Model type : ${MODEL_TYPE}"
echo "Ckpt dir   : ${CKPT_SAVE_DIR}"
echo "Download-only: ${DOWNLOAD_ONLY}"

# ---------------------------------------------------------------------------
# Download model and datasets
# ---------------------------------------------------------------------------
mkdir -p /root/models /root/datasets
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('${HF_REPO}', local_dir='/root/models/${MODEL_NAME}')
"

python3 -c "
from miles.utils.external_utils.command_utils import hf_download_dataset
hf_download_dataset('zhuzilin/dapo-math-17k')
"

# ---------------------------------------------------------------------------
# Patch checkpoint config.json to use deepseek_v32 architecture
# ---------------------------------------------------------------------------
python3 -c "
import json
from pathlib import Path

model_dir = Path('/root/models/${MODEL_NAME}')
config_path = model_dir / 'config.json'
if not config_path.exists():
    raise FileNotFoundError(f'{config_path} not found')

with open(config_path) as f:
    config = json.load(f)

# Patch config only if not already using deepseek_v32 (i.e. zai-org/GLM-5).
# Pinaster variants already ship with model_type=deepseek_v32 + stubs.
if config.get('model_type') == 'deepseek_v32':
    print('Checkpoint already patched, skipping config')
else:
    config['architectures'] = ['DeepseekV32ForCausalLM']
    config['auto_map'] = {
        'AutoConfig': 'configuration_deepseek_v32.DeepseekV32Config',
        'AutoModelForCausalLM': 'modeling_deepseek_v32.DeepseekV32ForCausalLM',
    }
    config['model_type'] = 'deepseek_v32'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f'Patched {config_path}')

# Write stub files only if absent (zai-org/GLM-5 doesn't ship them;
# Pinaster variants already include them).
config_py = model_dir / 'configuration_deepseek_v32.py'
if not config_py.exists():
    config_py.write_text('''from transformers.models.deepseek_v3.configuration_deepseek_v3 import DeepseekV3Config


class DeepseekV32Config(DeepseekV3Config):
    model_type = \"deepseek_v32\"

    def __init__(self, index_topk=2048, **kwargs):
        super().__init__(**kwargs)
        self.index_topk = index_topk
''')
    print(f'Wrote {config_py}')
else:
    print(f'Stub {config_py} already exists, skipping')

modeling_py = model_dir / 'modeling_deepseek_v32.py'
if not modeling_py.exists():
    modeling_py.write_text('''from transformers import PreTrainedModel
from .configuration_deepseek_v32 import DeepseekV32Config


class DeepseekV32ForCausalLM(PreTrainedModel):
    config_class = DeepseekV32Config
''')
    print(f'Wrote {modeling_py}')
else:
    print(f'Stub {modeling_py} already exists, skipping')
"

# ---------------------------------------------------------------------------
# Convert checkpoint to megatron format
# Skip if --download-only (e.g. worker/rollout node doesn't need megatron format)
# ---------------------------------------------------------------------------
if [ "${DOWNLOAD_ONLY}" -eq 0 ]; then
    # Build Python conversion call
    PY_MULTINODE="True"
    if [ "${CONVERT_MULTINODE}" = "false" ]; then
        PY_MULTINODE="False"
    fi

    PY_NUM_NODES="None"
    if [ -n "${CONVERT_NUM_NODES}" ]; then
        PY_NUM_NODES="${CONVERT_NUM_NODES}"
    fi

    python3 -c "
from miles.utils.external_utils.command_utils import convert_checkpoint
kwargs = dict(
    model_name='${MODEL_NAME}',
    megatron_model_type='${MODEL_TYPE}',
    num_gpus_per_node=${CONVERT_GPUS},
    multinode=${PY_MULTINODE},
    extra_args='${CONVERT_EXTRA_ARGS}',
    dir_dst='${CKPT_SAVE_DIR}',
)
num_nodes = ${PY_NUM_NODES}
if num_nodes is not None:
    kwargs['num_nodes'] = num_nodes
convert_checkpoint(**kwargs)
"
    echo "Prepare done (full: download + patch + convert)."
else
    echo "Prepare done (download-only: skipped checkpoint conversion)."
fi
