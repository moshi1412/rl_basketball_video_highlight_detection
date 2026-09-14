#!/usr/bin/env bash
# 篮球高光「候选片段二分类」的 GRPO —— 用 scripts/highlight_reward_plugin.py 里那套
# 「格式 + 判对 + 理由/事件词」reward，把模型的对高光的口味对齐到标注上。
#
# 需要先起 rollout 服务（2 卡）：
#   VLLM_GPUS=0,1 MAX_PIXELS=100352 bash scripts/01_serve_vllm.sh
# 然后：
#   TRAIN_GPUS=2,3,4,5,6,7,8 MAX_STEPS=30 bash scripts/11_grpo_highlight_cls.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"

export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS:-2,3,4,5,6,7,8}
export NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES")
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}
DATA=${DATA_DIR:-$SWIFT_RL_ROOT/data/highlight_cls_v2}
MAX_STEPS=${MAX_STEPS:-30}
NUM_GEN=${NUM_GEN:-4}
PER_DEVICE=$NUM_GEN          # swift 要求 per_device_train_batch_size 能被 num_generations 整除
TEMPERATURE=${TEMPERATURE:-1.0}
EVAL_STRATEGY=${EVAL_STRATEGY:-steps}   # 设成 no 就跳过内置验证（自己用 infer 测更快）

TRAIN=()
VAL=()
for f in "$DATA"/*.jsonl; do
  case "$(basename "$f")" in
    3-3v3_*) VAL+=("$f") ;;
    *)       TRAIN+=("$f") ;;
  esac
done
echo "[grpo-highlight] train=${#TRAIN[@]} val=${#VAL[@]} GPUs=$CUDA_VISIBLE_DEVICES num_gen=$NUM_GEN steps=$MAX_STEPS"

# 想从 SFT 的 LoRA 权重继续 RL（playbook 推荐的"SFT 打底再 GRPO"），传 ADAPTER=<sft checkpoint>
ADAPTER_ARGS=""
if [ -n "${ADAPTER:-}" ]; then
  ADAPTER_ARGS="--adapters $ADAPTER"
  echo "[grpo-highlight] 从 SFT adapter 继续: $ADAPTER"
fi

# tuner：默认 LoRA；若 TUNER_TYPE=full（配合 MODEL=合并后的模型）则走全参微调
TUNER=${TUNER_TYPE:-lora}
if [ "$TUNER" = "lora" ]; then
  LORA_ARGS="--lora_rank ${LORA_RANK:-8} --lora_alpha 32 --target_modules all-linear --vllm_enable_lora true --vllm_max_lora_rank ${LORA_RANK:-8}"
else
  LORA_ARGS=""
fi

# 注意：reward 名字必须作为**多个参数**传给 swift，写成 "${VAR}" 会被当成一个字符串
read -r -a REWARD_FUNCS_ARR <<< "${REWARD_FUNCS:-highlight_format highlight_label_balanced highlight_evidence}"
read -r -a REWARD_WEIGHTS_ARR <<< "${REWARD_WEIGHTS:-0.2 1.0 0.3}"
echo "[grpo-highlight] rewards: ${REWARD_FUNCS_ARR[*]} | weights: ${REWARD_WEIGHTS_ARR[*]}"
echo "[grpo-highlight] tuner=$TUNER model=${MODEL:-$SWIFT_RL_MODEL}"

exec swift rlhf --rlhf_type grpo \
  --model "${MODEL:-$SWIFT_RL_MODEL}" \
  $ADAPTER_ARGS \
  --tuner_type "$TUNER" \
  $LORA_ARGS \
  --freeze_vit true --freeze_aligner true \
  --torch_dtype bfloat16 \
  --dataset "${TRAIN[@]}" \
  --val_dataset "${VAL[@]}" \
  --load_from_cache_file true \
  --external_plugins "$SWIFT_RL_ROOT/scripts/highlight_reward_plugin.py" \
  --reward_funcs "${REWARD_FUNCS_ARR[@]}" \
  --reward_weights "${REWARD_WEIGHTS_ARR[@]}" \
  --use_vllm true \
  --vllm_mode server \
  --vllm_server_host 127.0.0.1 \
  --vllm_server_port "$VLLM_PORT" \
  --max_pixels 100352 \
  --max_length 3072 \
  --max_completion_length 256 \
  --num_train_epochs 1 \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$PER_DEVICE" \
  --per_device_eval_batch_size "$PER_DEVICE" \
  --gradient_accumulation_steps 1 \
  --num_generations "$NUM_GEN" \
  --temperature "$TEMPERATURE" \
  --learning_rate 1e-6 \
  --beta 0.01 \
  --max_grad_norm 0.5 \
  --warmup_ratio 0.05 \
  --deepspeed zero3 \
  --logging_steps 1 \
  --save_strategy steps --save_steps "$MAX_STEPS" --save_total_limit 2 \
  --eval_strategy "$EVAL_STRATEGY" --eval_steps "$MAX_STEPS" \
  --dataset_num_proc 4 --dataloader_num_workers 2 \
  --log_completions true \
  --report_to swanlab --swanlab_mode local --swanlab_project highlight-cls-grpo \
  --output_dir "$SWIFT_RL_ROOT/output/grpo_highlight_cls"
