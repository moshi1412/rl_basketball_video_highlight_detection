#!/usr/bin/env bash
# Megatron-SWIFT 的 GRPO（Mcore-Bridge 方式：直接用 HF 权重，无需格式转换）
# 对应官方文档: Megatron-SWIFT/GRPO + Megatron-SWIFT/快速开始(Mcore-Bridge)
set -euo pipefail
source "$(dirname "$0")/../env.sh"

# TransformerEngine / NCCL / cuDNN 的 .so 在 venv 的 nvidia/*/lib 里，需要加进 LD_LIBRARY_PATH
NVLIB=$(find "$SWIFT_RL_VENV/lib/python3.12/site-packages/nvidia" -maxdepth 2 -type d -name lib | tr '\n' ':')
export LD_LIBRARY_PATH=${NVLIB}${LD_LIBRARY_PATH:-}

export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
export NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$TRAIN_GPUS")
TRAIN_ITERS=${TRAIN_ITERS:-10}

echo "[megatron-grpo] GPUs=$CUDA_VISIBLE_DEVICES nproc=$NPROC_PER_NODE vllm_server=$VLLM_PORT train_iters=$TRAIN_ITERS"

exec megatron rlhf --rlhf_type grpo \
  --model "$SWIFT_RL_MODEL" \
  --dataset 'AI-ModelScope/clevr_cogen_a_train#2000' \
  --load_from_cache_file true \
  --split_dataset_ratio 0.01 \
  --reward_funcs accuracy format \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port "$VLLM_PORT" \
  --tensor_model_parallel_size 1 \
  --pipeline_model_parallel_size 1 \
  --micro_batch_size 1 \
  --global_batch_size 8 \
  --num_generations 2 \
  --max_length 4096 \
  --max_completion_length 512 \
  --max_pixels "$MAX_PIXELS" \
  --train_iters "$TRAIN_ITERS" \
  --lr 1e-6 --min_lr 1e-7 --lr_warmup_fraction 0.05 --lr_decay_style constant \
  --attention_backend auto \
  --recompute_granularity full \
  --recompute_method uniform \
  --recompute_num_layers 1 \
  --no_gradient_accumulation_fusion \
  --main_grads_dtype bf16 \
  --use_precision_aware_optimizer true \
  --optimizer_cpu_offload true \
  --optimizer_offload_fraction 1.0 \
  --finetune true \
  --freeze_vit true \
  --freeze_aligner true \
  --save_steps "$TRAIN_ITERS" \
  --no_save_optim \
  --no_save_rng \
  --log_completions true \
  --report_to swanlab \
  --swanlab_project "$SWIFT_RL_SWANLAB_PROJECT" \
  --output_dir "$SWIFT_RL_ROOT/output/megatron_grpo_clevr_3b" \
  --dataloader_num_workers 2 \
  --dataset_num_proc 4 \
  --beta 0.001 \
  --temperature 1.0
