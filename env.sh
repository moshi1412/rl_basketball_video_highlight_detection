#!/usr/bin/env bash
# 公共环境变量：任何脚本/终端使用前先 `source <本工程>/env.sh`
# 环境由 uv 创建：uv venv --python 3.12 .venv  (详见 scripts/00_setup_env.sh)
# 根目录由脚本位置推导，工程整体挪动后无需改路径

export SWIFT_RL_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export SWIFT_RL_VENV=$SWIFT_RL_ROOT/.venv
export PATH=$SWIFT_RL_VENV/bin:$PATH

# 模型/数据集缓存（放在工程目录里，避免污染 ~/.cache）
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-$SWIFT_RL_ROOT/.cache/modelscope}
export HF_HOME=${HF_HOME:-$SWIFT_RL_ROOT/.cache/huggingface}

# 小模型：Qwen2.5-VL-3B-Instruct，ms-swift(swift rlhf) 与 Megatron-SWIFT 都支持
export SWIFT_RL_MODEL=${SWIFT_RL_MODEL:-$MODELSCOPE_CACHE/models/Qwen--Qwen2.5-VL-3B-Instruct/snapshots/master}

# 视觉 token 上限：3090 24G 显存友好（官方 GRPO 实践里也用 262144 防 OOM）
export MAX_PIXELS=${MAX_PIXELS:-262144}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# SwanLab：local 模式，不需要账号/API key，日志落在 $SWANLAB_LOG_DIR
# 查看面板：bash scripts/03_watch_swanlab.sh  然后浏览器打开 http://127.0.0.1:5092
# 注意：不要设置 SWANLAB_PROJECT / SWANLAB_RUN 等环境变量（swanlab 0.10 里它们是嵌套
# 配置字段，env 值必须是 JSON，否则 pydantic 解析报错）；project 名用命令行参数传。
export SWANLAB_MODE=${SWANLAB_MODE:-local}
export SWANLAB_LOG_DIR=${SWANLAB_LOG_DIR:-$SWIFT_RL_ROOT/swanlog}
export SWIFT_RL_SWANLAB_PROJECT=${SWIFT_RL_SWANLAB_PROJECT:-swift-grpo-vlm}

# GPU 分配（0、1 号卡已被其他任务占用）
export VLLM_GPUS=${VLLM_GPUS:-2,3}      # 推理(rollout)用卡
export TRAIN_GPUS=${TRAIN_GPUS:-4,5,6,7} # 训练用卡
export VLLM_PORT=${VLLM_PORT:-8000}
