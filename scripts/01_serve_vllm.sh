#!/usr/bin/env bash
# 启动 vLLM 推理服务（GRPO rollout 用，server 模式）
set -euo pipefail
source "$(dirname "$0")/../env.sh"

export CUDA_VISIBLE_DEVICES=$VLLM_GPUS
export NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$VLLM_GPUS")

echo "[rollout] GPUs=$CUDA_VISIBLE_DEVICES port=$VLLM_PORT model=$SWIFT_RL_MODEL"
# 注意：GRPO 的 vLLM server 要用 `swift rollout`（它注册了 /init_communicator、
# /update_named_param 等权重同步接口），`swift deploy` 只用于普通推理部署。
exec swift rollout \
  --model "$SWIFT_RL_MODEL" \
  --infer_backend vllm \
  --vllm_tensor_parallel_size "$NPROC_PER_NODE" \
  --vllm_gpu_memory_utilization 0.75 \
  --vllm_max_model_len 8192 \
  --vllm_limit_mm_per_prompt '{"image":1}' \
  --max_pixels "$MAX_PIXELS" \
  --host 127.0.0.1 \
  --port "$VLLM_PORT"
