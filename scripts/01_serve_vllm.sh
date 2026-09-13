#!/usr/bin/env bash
# 启动 vLLM 推理服务（GRPO rollout 用，server 模式）
set -euo pipefail
source "$(dirname "$0")/../env.sh"

export CUDA_VISIBLE_DEVICES=$VLLM_GPUS
export NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$VLLM_GPUS")

echo "[rollout] GPUs=$CUDA_VISIBLE_DEVICES port=$VLLM_PORT model=$SWIFT_RL_MODEL"
# 多帧片段（把 4 帧当多图送进去）需要放开每条 prompt 的图片数限制，否则 vLLM 会返回 500
export VLLM_LIMIT_MM=${VLLM_LIMIT_MM:-'{"image":1}'}
echo "[rollout] vllm_limit_mm_per_prompt=$VLLM_LIMIT_MM"
# LoRA 训练时让 vLLM 直接加载/热更新 LoRA，权重同步只传 adapter（省显存/带宽）
export VLLM_ENABLE_LORA=${VLLM_ENABLE_LORA:-false}
export VLLM_MAX_LORA_RANK=${VLLM_MAX_LORA_RANK:-16}
# 注意：GRPO 的 vLLM server 要用 `swift rollout`（它注册了 /init_communicator、
# /update_named_param 等权重同步接口），`swift deploy` 只用于普通推理部署。
exec swift rollout \
  --model "$SWIFT_RL_MODEL" \
  --infer_backend vllm \
  --vllm_tensor_parallel_size "$NPROC_PER_NODE" \
  --vllm_gpu_memory_utilization 0.75 \
  --vllm_max_model_len 8192 \
  --vllm_enable_lora "$VLLM_ENABLE_LORA" \
  --vllm_max_lora_rank "$VLLM_MAX_LORA_RANK" \
  --vllm_limit_mm_per_prompt "$VLLM_LIMIT_MM" \
  --max_pixels "$MAX_PIXELS" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT"
