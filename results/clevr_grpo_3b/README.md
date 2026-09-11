# ClevrCount 多模态 GRPO 结果包（环境/框架验证）

这个文件夹是「用 uv 装 ms-swift + 跑通 VLM 强化学习」这次任务的完整结果，
包括训练曲线、逐 step 指标、抽样数据、三个模型的回答、以及一个对比 demo 视频。

## 任务

* 数据集：`AI-ModelScope/clevr_cogen_a_train`（CLEVR 数物体，输出 `<think>…</think><answer>N</answer>`）
* 模型：**Qwen2.5-VL-3B-Instruct**（多模态、体积小，ms-swift 与 Megatron-SWIFT 都支持）
* 算法：GRPO（`swift rlhf --rlhf_type grpo`），reward = `accuracy`（math_verify）+ `format`
* 硬件：4×RTX3090 训练 + 2×RTX3090 跑 vLLM rollout（server 模式），deepspeed zero3
* 另有一路 **Megatron-SWIFT GRPO**（mcore-bridge，5 iter）验证，日志见 `logs_megatron_grpo.txt`

## 训练结果（100 步那一次）

| 指标 | 数值 |
| --- | --- |
| step | 1 → 100 |
| reward | 1.125 → **1.625** |
| accuracy reward | 0.25 → **0.625** |
| format reward | 0.875 → **1.0** |
| KL | 0 → 0.0495 |
| completion 长度 | 154 → 90 token |
| 速度 / 显存 | 12.3 s/iter，23.0 GiB/卡 |
| 验证集（20 条） | reward 1.6 / accuracy 0.6 / format 1.0 |

曲线见 `figures/training_curve_grpo100.png`（reward/accuracy/format、KL、长度、grad_norm 四联图），
逐 step 原始数据见 `metrics/logging_grpo100.jsonl`。

## Demo（抽样数据 + 模型回答 + 视频）

`demo/clevr_grpo_demo.mp4`：20 条**训练用同一分布的留出验证样本**，左边是图片，右边是
GT 与三个模型的回答（base / GRPO-20 步 / GRPO-100 步），带 ✅❌ 和累计准确率。

准确率（n=20，同一批样本、同一 prompt、temperature=0.01）：

| 模型 | accuracy |
| --- | --- |
| base Qwen2.5-VL-3B-Instruct | **80%** |
| GRPO 20 步 | 75% |
| GRPO 100 步 | 70% |

另外在训练分布之外的 12 张随机图片上：base 66.7% / GRPO-100 步 50%（见 `samples_random/`）。

**结论要说清楚**：这次的目标是验证「环境 + 框架 + 训练链路 + 监控」是否可用——这一点已经跑通
（reward 稳定上升、format 收敛到 1.0、SwanLab 曲线正常、checkpoint 可推理）。
但**这个 100 步的 GRPO 并没有超过 base 模型的数数准确率**：训练只用了 2000 条样本、
reward 只有 accuracy+format、LR 1e-6、batch 很小，属于「跑得通」而不是「训得好」。
base 模型本身在 CLEVR 数数上就有 ~80%，RL 在没有增量信号（base 已经会做）时容易把模型带偏。
要真正提升，应该换更有难度的任务（如 `GEOQA_R1V_Train_8K`）、加大数据量与步数、
调 reward 权重/LR，并把 RFT/拒绝采样纳入流程。

## 文件说明

```
README.md                      本文件
configs/args_grpo100.json      100 步 run 的完整训练参数
metrics/logging_grpo100.jsonl  逐 step 指标（reward/kl/grad_norm/长度…）
metrics/logging_grpo20.jsonl   20 步那次
metrics/summary.json           汇总指标（训练/验证/demo 准确率）
figures/training_curve_grpo100.png   四联曲线图
figures/frames_example/        demo 视频的两帧截图
samples/images/*.png           20 条验证样本图片（GT 在 infer_input.jsonl 里）
samples/infer_input.jsonl      swift infer 的输入（messages+images+solution）
samples/vlm_answers.jsonl      三个模型的逐样本回答 + 对错
samples/vlm_answers_readable.md  同上，人类可读版
samples_random/                另一次「训练分布之外」的 12 张随机抽样结果
vlm_outputs/*.jsonl            三个模型的原始 infer 输出（含完整 reasoning）
demo/clevr_grpo_demo.mp4       对比 demo 视频（1280×720，20 段，每段 3s）
logs_grpo100.txt               ms-swift 训练完整 stdout
logs_megatron_grpo.txt         Megatron-SWIFT GRPO 完整 stdout
```

## 复现命令

```bash
cd /data/ljy23/basketball_project/cuttingclip/rl_basketball_video_highlight_detection
source env.sh

# 1) 训练（占 4-7 卡）
VLLM_GPUS=2,3 bash scripts/01_serve_vllm.sh > logs/rollout_server.log 2>&1 &
MAX_STEPS=100 bash scripts/02_grpo_multimodal.sh

# 2) 抽样 + 生成 demo
python3 scripts/07_make_clevr_demo.py prepare --num 20 --index-range 1980:2000 --out results/clevr_grpo_3b
CUDA_VISIBLE_DEVICES=1 swift infer --model "$SWIFT_RL_MODEL" \
    --val_dataset results/clevr_grpo_3b/samples/infer_input.jsonl \
    --max_pixels 262144 --max_new_tokens 256 --temperature 0.01 --attn_impl sdpa
CUDA_VISIBLE_DEVICES=2 swift infer --model output/grpo_clevr_3b/v2-*/checkpoint-100 \
    --val_dataset results/clevr_grpo_3b/samples/infer_input.jsonl \
    --max_pixels 262144 --max_new_tokens 256 --temperature 0.01 --attn_impl sdpa
python3 scripts/07_make_clevr_demo.py render --out results/clevr_grpo_3b \
    --models base=<base_infer.jsonl> GRPO-100step=<trained_infer.jsonl>

# 3) 看训练曲线
bash scripts/03_watch_swanlab.sh     # SwanLab 本地看板 :5092
python3 scripts/06_plot_curves.py output/grpo_clevr_3b/v2-*/logging.jsonl -o figures/curve.png
```
