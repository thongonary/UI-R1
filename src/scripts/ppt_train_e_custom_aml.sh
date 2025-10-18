#!/usr/bin/env bash
set -euo pipefail

# Install required packages
pip install Levenshtein

# AML-aware variant of ppt_train_e_custom.sh
# Key differences:
#  - Removes manual torch.distributed.run launcher; AML PyTorch distribution handles process spawning.
#  - Respects externally provided SAVE_PATH (e.g., AML output binding) by only assigning a default if unset.
#  - Avoids hard-coded nnodes/master_addr/master_port which are injected by AML runtime.
#  - Removes CUDA_VISIBLE_DEVICES manual masking (AML sets visibility per process when using distribution spec).
#  - Adds lightweight environment diagnostics.

export DEBUG_MODE="true"

export DATA_PATH=../../data/ppt-font-grounding
# Base model checkpoint (HF Hub or local path)
export CKPT_PATH=${CKPT_PATH:-Qwen/Qwen2.5-VL-3B-Instruct}

# Only set SAVE_PATH default if not already provided by environment (AML passes it in env vars)
: "${SAVE_PATH:=/home/data/ckpt/Qwen2.5-VL-PPT-font-ground-dast-16ep-clipped-higher}" 

export LOG_PATH="debug_log.txt"
export Train_PATH="train.log"
mkdir -p "${SAVE_PATH}" || true

echo "[INFO] STARTING TRAINING"
echo "[INFO] DATE: $(date -u)"
echo "[INFO] HOSTNAME: $(hostname)"
echo "[INFO] PWD: $(pwd)"
echo "[INFO] SAVE_PATH: ${SAVE_PATH}" 
# World/distributed env vars if present (AML sets these)
for v in RANK LOCAL_RANK NODE_RANK WORLD_SIZE MASTER_ADDR MASTER_PORT AZ_BATCHAI_JOB_MASTER_NODE_IP; do
  if [[ -n "${!v-}" ]]; then echo "[DIST] $v=${!v}"; fi
done

# NOTE: No manual torchrun/torch.distributed invocation here; AML launches multiple processes.
python ../ui_r1/src/open_r1/grpo_json_action_coord-dast.py \
    --output_dir "${SAVE_PATH}" \
    --model_name_or_path "${CKPT_PATH}" \
    --data_file_paths ../../data/ppt-font-grounding/train_ground_click_only.json \
    --image_folders ../../data/ppt-font-grounding/train_imgs \
    --dataset_name "${DATA_PATH}" \
    --deepspeed ../ui_r1/local_scripts/zero3.json \
    --max_prompt_length 1024 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to tensorboard \
    --max_completion_length 512 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 12845056 \
    --num_train_epochs 1 \
    --run_name GRPO_example \
    --save_strategy epoch \
    --save_only_model true \
    --num_generations 16

EXIT_CODE=$?

echo "[INFO] Training script exited with code ${EXIT_CODE}"
exit ${EXIT_CODE}
