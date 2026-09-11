#!/usr/bin/env bash
# 记录环境是如何建立的（可重复执行；已装好则会直接跳过）
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

# 1) uv + python3.12 虚拟环境
# uv 安装: curl -LsSf https://astral.sh/uv/install.sh | sh   （本机已有 uv 0.9.26）
uv venv --python 3.12 .venv

# 2) 安装 ms-swift（含 megatron / swanlab 依赖）+ vllm + deepspeed
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install --python "$ROOT/.venv/bin/python" \
  "ms-swift[swanlab,megatron]==4.5.3" \
  vllm deepspeed math-verify

# 3) SwanLab 本地看板（swanlab watch 需要）
uv pip install --python "$ROOT/.venv/bin/python" --prerelease=allow "swanlab[dashboard]"

# 4) 可选：flash-attn（训练加速，非必需）
#    本机 nvcc 在 .venv/lib/python3.12/site-packages/nvidia/cu13/bin/nvcc（CUDA 13）
#    MAX_JOBS=16 CUDA_HOME=... .venv/bin/pip install "flash-attn==2.8.3" --no-build-isolation

echo "done. use: source $ROOT/env.sh"
