#!/usr/bin/env bash
# 从 GEOQA GRPO 的 checkpoint 继续训练（上一次跑到 step ~124 因 rank0 显存 OOM 中断）。
#
# 相比第一次的关键改动（都是"降显存 + 控漂移"）：
#   * num_generations 8 → 4（completion 数减半，激活显存大幅下降）
#   * per_device_train/eval_batch_size = num_generations（swift 要求整除）
#   * beta 0.01 → 0.03（第一次 KL 涨到 0.5，漂移明显）
#   * 从 checkpoint-100 续训，output_dir 保持原 run 目录，曲线续在同一条上
#
# 用法（需先起 rollout 服务）：
#   VLLM_GPUS=0,1 MAX_PIXELS=401408 bash scripts/01_serve_vllm.sh
#   TRAIN_GPUS=2,3,4,5,6,7,8 MAX_STEPS=400 bash scripts/12_resume_geoqa_grpo.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-2,3,4,5,6,7,8}
export NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}
export MAX_PIXELS=${GEOQA_MAX_PIXELS:-401408}

CKPT=${CKPT:-$(ls -d "$SWIFT_RL_ROOT"/output/grpo_geoqa_3b/v1-*/checkpoint-100 2>/dev/null | head -1)}
if [ -z "${CKPT:-}" ]; then echo "找不到 checkpoint-100，请用 CKPT=... 指定"; exit 1; fi
RUN_DIR=$(dirname "$CKPT")
NUM_GEN=${NUM_GEN:-4}
MAX_STEPS=${MAX_STEPS:-400}
LR=${LR:-1e-6}
BETA=${BETA:-0.03}

echo "[geoqa-resume] from $CKPT"
echo "[geoqa-resume] GPUs=$CUDA_VISIBLE_DEVICES num_gen=$NUM_GEN max_steps=$MAX_STEPS beta=$BETA lr=$LR"

exec swift rlhf --rlhf_type grpo \
  --model "$SWIFT_RL_MODEL" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --dataset 'AI-ModelScope/GEOQA_R1V_Train_8K' \
  --load_from_cache_file true \
  --split_dataset_ratio 0.01 \
  --resume_from_checkpoint "$CKPT" \
  --reward_funcs accuracy format \
  --system "$SWIFT_RL_ROOT/prompts/geoqa_system.txt" \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port "$VLLM_PORT" \
  --max_pixels "$MAX_PIXELS" \
  --max_length 4096 \
  --max_completion_length 1024 \
  --num_train_epochs 1 \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$NUM_GEN" \
  --per_device_eval_batch_size "$NUM_GEN" \
  --gradient_accumulation_steps 1 \
  --num_generations "$NUM_GEN" \
  --num_iterations 1 \
  --temperature 1.0 \
  --repetition_penalty 1.1 \
  --learning_rate "$LR" \
  --beta "$BETA" \
  --max_grad_norm 0.5 \
  --warmup_ratio 0.02 \
  --deepspeed zero3 \
  --logging_steps 1 \
  --save_strategy steps --save_steps 50 --save_total_limit 8 \
  --eval_strategy steps --eval_steps 50 \
  --dataset_num_proc 8 --dataloader_num_workers 4 \
  --log_completions true \
  --report_to swanlab --swanlab_mode local --swanlab_project geoqa-grpo-vlm \
  --output_dir "$RUN_DIR"
