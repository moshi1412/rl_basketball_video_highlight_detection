#!/usr/bin/env bash
# 任务 B 的 GRPO：在 CLEVR 高计数(<think>+JSON)任务上做强化学习。
#   START=base    → 直接从 base 模型 RL（看"纯 RL 能提多少"）
#   START=sft     → 从 SFT 的 LoRA 权重继续 RL（ADAPTER=...）
#
# 用法：
#   VLLM_GPUS=0,1 bash scripts/01_serve_vllm.sh
#   TRAIN_GPUS=2,3,4,5,6,7,8 MAX_STEPS=100 bash scripts/17_grpo_clevr_count.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-2,3,4,5,6,7,8}
export NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}
DATA=${DATA_DIR:-$SWIFT_RL_ROOT/data/clevr_count_hard}
MAX_STEPS=${MAX_STEPS:-100}
NUM_GEN=${NUM_GEN:-4}
START=${START:-base}

ADAPTER_ARGS=""
if [ "$START" = "sft" ]; then
  ADAPTER_ARGS="--adapters ${ADAPTER:?START=sft 需要指定 ADAPTER=...}"
fi

echo "[grpo-count] start=$START GPUs=$CUDA_VISIBLE_DEVICES num_gen=$NUM_GEN steps=$MAX_STEPS"

exec swift rlhf --rlhf_type grpo \
  --model "$SWIFT_RL_MODEL" \
  $ADAPTER_ARGS \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --dataset "$DATA/train.jsonl" \
  --val_dataset "$DATA/val.jsonl" \
  --external_plugins "$SWIFT_RL_ROOT/scripts/clevr_count_reward_plugin.py" \
  --reward_funcs count_accuracy count_format count_brevity \
  --reward_weights 1.0 0.2 0.2 \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port "$VLLM_PORT" \
  --max_pixels 262144 \
  --max_length 2048 \
  --max_completion_length 256 \
  --num_train_epochs 1 \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$NUM_GEN" \
  --per_device_eval_batch_size "$NUM_GEN" \
  --gradient_accumulation_steps 1 \
  --num_generations "$NUM_GEN" \
  --temperature 1.0 \
  --learning_rate 1e-6 \
  --beta 0.01 \
  --max_grad_norm 0.5 \
  --warmup_ratio 0.03 \
  --deepspeed zero3 \
  --logging_steps 1 \
  --save_strategy steps --save_steps 50 --save_total_limit 4 \
  --eval_strategy steps --eval_steps 50 \
  --dataset_num_proc 4 --dataloader_num_workers 2 \
  --log_completions true \
  --report_to swanlab --swanlab_mode local --swanlab_project clevr-count-hard-grpo \
  --output_dir "$SWIFT_RL_ROOT/output/grpo_clevr_count"
