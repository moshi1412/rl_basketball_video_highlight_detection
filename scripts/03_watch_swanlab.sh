#!/usr/bin/env bash
# 打开 SwanLab 本地看板（读取 $SWANLAB_SAVE_DIR 下的离线日志）
set -euo pipefail
source "$(dirname "$0")/../env.sh"

echo "SwanLab log dir: $SWANLAB_LOG_DIR"
exec swanlab watch "$SWANLAB_LOG_DIR" --host 127.0.0.1 --port ${SWANLAB_PORT:-5092}
