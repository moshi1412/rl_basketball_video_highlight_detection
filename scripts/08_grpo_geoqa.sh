#!/usr/bin/env bash
# GEOQA 几何问答 GRPO：比 ClevrCount 难得多（base 模型有明显提升空间），
# 超参按「组大小 / KL 约束 / 批大小」三条经验修正过（见 docs/grpo_geoqa_report.md）。
#
# 用法:
#   VLLM_GPUS=0,1 bash scripts/01_serve_vllm.sh &          # rollout 服务
#   TRAIN_GPUS=2,3,4,5,6,7,8 MAX_STEPS=200 bash scripts/08_grpo_geoqa.sh
set -euo pipefail
source "$(dirname "$0")/../env.sh"

export CUDA_VISIBLE_DEVICES=$TRAIN_GPUS
export NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$TRAIN_GPUS")
export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/mplconfig}
MAX_STEPS=${MAX_STEPS:-200}
GRAD_ACCUM=${GRAD_ACCUM:-1}     # 实测 grad_accum=2 时单步 ~70s（训练占 ~50s），=1 约 35s/步
export MAX_PIXELS=${GEOQA_MAX_PIXELS:-401408}   # 官方 GEOQA 实践用的像素上限
RESUME_ARGS=""
if [ "${RESUME:-0}" = "1" ]; then
  RESUME_ARGS="--resume_from_checkpoint true"
fi

echo "[geoqa-grpo] train GPUs=$CUDA_VISIBLE_DEVICES (nproc=$NPROC_PER_NODE) rollout=$VLLM_PORT max_steps=$MAX_STEPS"

exec swift rlhf --rlhf_type grpo \
  --model "$SWIFT_RL_MODEL" \
  --tuner_type full \
  --torch_dtype bfloat16 \
  --dataset 'AI-ModelScope/GEOQA_R1V_Train_8K' \
  --load_from_cache_file true \
  --split_dataset_ratio 0.01 \
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
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --num_generations 8 \
  --num_iterations 1 \
  --temperature 1.0 \
  --repetition_penalty 1.1 \
  --learning_rate 1e-6 \
  --beta 0.01 \
  --max_grad_norm 0.5 \
  --warmup_ratio 0.05 \
  --deepspeed zero3 \
  --logging_steps 1 \
  --save_strategy steps \
  --save_steps 50 \
  --save_total_limit 4 \
  --eval_strategy steps \
  --eval_steps 50 \
  --per_device_eval_batch_size 8 \
  --dataset_num_proc 8 \
  --dataloader_num_workers 4 \
  --log_completions true \
  --report_to swanlab \
  --swanlab_mode local \
  --swanlab_project geoqa-grpo-vlm \
  $RESUME_ARGS \
  --output_dir "$SWIFT_RL_ROOT/output/grpo_geoqa_3b"
