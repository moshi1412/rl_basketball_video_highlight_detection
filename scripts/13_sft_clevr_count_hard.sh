#!/usr/bin/env bash
# 任务 B：CLEVR 高计数难题 + 结构化输出（<think>…</think><answer>{"count": N}</answer>）的 LoRA SFT。
# 数据由 scripts/prepare_clevr_count_hard.py 生成（离线读 CLEVR 缓存，只留 count>=8 的图）。
#
# 用法: TRAIN_GPUS=4,5,6,7 MAX_STEPS=150 bash scripts/13_sft_clevr_count_hard.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-4,5,6,7}
export NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}
DATA=${DATA_DIR:-$SWIFT_RL_ROOT/data/clevr_count_hard}
MAX_STEPS=${MAX_STEPS:-150}

echo "[sft-count] GPUs=$CUDA_VISIBLE_DEVICES steps=$MAX_STEPS dataset=$DATA"

exec swift sft \
  --model "$SWIFT_RL_MODEL" \
  --tuner_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
  --freeze_vit true --freeze_aligner true \
  --dataset "$DATA/train.jsonl" \
  --val_dataset "$DATA/val.jsonl" \
  --max_pixels 262144 \
  --max_length 2048 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 2 \
  --num_train_epochs 2 \
  --max_steps "$MAX_STEPS" \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --eval_strategy steps --eval_steps "$MAX_STEPS" \
  --save_strategy steps --save_steps "$MAX_STEPS" --save_total_limit 2 \
  --logging_steps 5 \
  --dataloader_num_workers 2 \
  --dataset_num_proc 4 \
  --report_to swanlab --swanlab_mode local --swanlab_project clevr-count-hard-sft \
  --output_dir "$SWIFT_RL_ROOT/output/sft_clevr_count_hard"
