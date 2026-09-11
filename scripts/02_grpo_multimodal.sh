#!/usr/bin/env bash
# 多模态 GRPO 训练（ClevrCount 任务 / Qwen2.5-VL-3B-Instruct）
# 对应官方文档: BestPractices/GRPO-Multi-Modal-Training
set -euo pipefail
source "$(dirname "$0")/../env.sh"

export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
export NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$TRAIN_GPUS")
export MASTER_PORT=${MASTER_PORT:-29500}
MAX_STEPS=${MAX_STEPS:-20}

echo "[grpo] train GPUs=$CUDA_VISIBLE_DEVICES (nproc=$NPROC_PER_NODE) vllm server=$VLLM_PORT max_steps=$MAX_STEPS"

exec swift rlhf --rlhf_type grpo \
  --model "$SWIFT_RL_MODEL" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --dataset 'AI-ModelScope/clevr_cogen_a_train#2000' \
  --load_from_cache_file true \
  --split_dataset_ratio 0.01 \
  --reward_funcs accuracy format \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port "$VLLM_PORT" \
  --max_pixels "$MAX_PIXELS" \
  --max_length 4096 \
  --max_completion_length 512 \
  --num_train_epochs 1 \
  --per_device_train_batch_size 2 \
  --gradient_accumulation_steps 1 \
  --num_generations 2 \
  --temperature 1.0 \
  --learning_rate 1e-6 \
  --beta 0.001 \
  --max_grad_norm 0.5 \
  --deepspeed zero3 \
  --logging_steps 1 \
  --save_steps "$MAX_STEPS" \
  --save_total_limit 3 \
  --max_steps "$MAX_STEPS" \
  --dataset_num_proc 4 \
  --dataloader_num_workers 2 \
  --log_completions true \
  --report_to swanlab \
  --swanlab_mode local \
  --swanlab_project "$SWIFT_RL_SWANLAB_PROJECT" \
  --output_dir "$SWIFT_RL_ROOT/output/grpo_clevr_3b"
