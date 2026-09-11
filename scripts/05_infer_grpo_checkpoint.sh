#!/usr/bin/env bash
# 用 GRPO 训练出来的 checkpoint 做推理，验证“rollout → 训练 → 推理”闭环
# 用法: CKPT=output/grpo_clevr_3b/v1-xxx/checkpoint-20 bash scripts/05_infer_grpo_checkpoint.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"

CKPT=${CKPT:-$(ls -d "$SWIFT_RL_ROOT"/output/grpo_clevr_3b/v*/checkpoint-* | head -1)}
export CUDA_VISIBLE_DEVICES=${INFER_GPU:-1}

echo "[infer] gpu=$CUDA_VISIBLE_DEVICES ckpt=$CKPT"
exec swift infer \
  --model "$CKPT" \
  --val_dataset 'AI-ModelScope/clevr_cogen_a_train#4' \
  --max_pixels "$MAX_PIXELS" \
  --max_new_tokens 256 \
  --temperature 0.01 \
  --attn_impl sdpa
