#!/usr/bin/env bash
# 篮球高光「候选片段二分类」的 SFT（LoRA）——教模型看懂片段 + 输出 <think>/<answer> 格式。
# 数据来自 scripts/prepare_highlight_dataset.py --mode cut --task classify --with-answer
#
# 划分原则：**按“节”划分**（1/2 节训练、3 节验证），绝不能按机位划分，
# 因为同一事件的 4 个机位画面不同、时间轴相同，拆开就是泄漏。
#
# 用法: TRAIN_GPUS=4,5,6,7 bash scripts/10_sft_highlight_cls.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-4,5,6,7}
export NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}
DATA=${DATA_DIR:-$SWIFT_RL_ROOT/data/highlight_cls_v3}
MAX_STEPS=${MAX_STEPS:-30}
# 默认用「答案 token 加权」的 loss（见 scripts/highlight_sft_loss_plugin.py）；
# LOSS_TYPE=cross_entropy 可退回普通 CE 做对照
LOSS_TYPE=${LOSS_TYPE:-answer_weighted}

TRAIN=()
VAL=()
for f in "$DATA"/*.jsonl; do
  case "$(basename "$f")" in
    3-3v3_*) VAL+=("$f") ;;      # 第 3 节做验证（按节划分，避免同事件跨集合）
    *)       TRAIN+=("$f") ;;
  esac
done
echo "[sft] train=${#TRAIN[@]} 个文件  val=${#VAL[@]} 个文件  GPUs=$CUDA_VISIBLE_DEVICES"

exec swift sft \
  --model "$SWIFT_RL_MODEL" \
  --external_plugins "$SWIFT_RL_ROOT/scripts/highlight_sft_loss_plugin.py" \
  --loss_type "$LOSS_TYPE" \
  --tuner_type lora --lora_rank 8 --lora_alpha 32 --target_modules all-linear \
  --freeze_vit "${FREEZE_VIT:-true}" --freeze_aligner "${FREEZE_ALIGNER:-true}" \
  --dataset "${TRAIN[@]}" \
  --val_dataset "${VAL[@]}" \
  --max_pixels 100352 \
  --max_length 3072 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --num_train_epochs 1 \
  --max_steps "$MAX_STEPS" \
  --learning_rate 1e-4 \
  --warmup_ratio 0.05 \
  --eval_strategy steps --eval_steps "$MAX_STEPS" \
  --save_strategy steps --save_steps "$MAX_STEPS" --save_total_limit 2 \
  --logging_steps 1 \
  --dataloader_num_workers 2 \
  --dataset_num_proc 4 \
  --report_to swanlab --swanlab_mode local --swanlab_project highlight-cls-sft \
  --output_dir "$SWIFT_RL_ROOT/output/sft_highlight_cls"
