# 训练总结：CLEVR 高计数任务（base / SFT / GRPO）

## 一、环境与框架

| 组件 | 版本 / 说明 |
| --- | --- |
| 环境管理 | **uv**（`uv venv --python 3.12 .venv`，见 `scripts/00_setup_env.sh`） |
| 训练框架 | **ms-swift 4.5.3**（`swift sft` / `swift rlhf --rlhf_type grpo`） |
| 推理加速 | **vLLM 0.29**（`swift rollout` 起服务，server 模式供 GRPO 采样） |
| 分布式 | deepspeed **zero3**（`NPROC_PER_NODE` 多卡）+ **LoRA/PEFT** |
| 监控 | **SwanLab 0.10**（local 模式，`scripts/03_watch_swanlab.sh` 看图） |
| 底座 | PyTorch 2.13+cu130、transformers 5.16、TRL 0.29、Python 3.12 |
| 硬件 | 9×RTX3090 24GB（本次用 4 卡 SFT / 7 卡+2 卡 RL） |
| 模型 | Qwen2.5-VL-3B-Instruct（多模态 3B） |

## 二、任务设计（为什么选它）

* 数据：本地缓存的 **CLEVR** 图，只挑 **物体数 ≥8 的难题**（base 在这种图上最差）
* 规模：3000 训练 / 300 验证（离线生成，无需联网）
* 输出格式：`<think>…</think><answer>{"count": N}</answer>`
* 判定：accuracy = JSON 中 N 与 GT 完全一致；format = 严格 `<think>…</think><answer>…</answer>`

选它的理由：base 明显会错、判定客观、对错一眼可见 —— 适合做演示。

## 三、训练流程

| 步骤 | 脚本 | 配置 | 实测耗时 |
| --- | --- | --- | --- |
| 1. 造数据 | `scripts/prepare_clevr_count_hard.py` | 离线读 CLEVR arrow 缓存，筛 count≥8，落图+jsonl | ~3 分钟 |
| 2. SFT（LoRA） | `scripts/13_sft_clevr_count_hard.sh` | LoRA r8/alpha32/all-linear，冻结 ViT+aligner，lr 1e-4，2 epoch 上限 150 步，4 卡，max_pixels 262144 | **8 分 40 秒** |
| 3. RL（GRPO） | `scripts/17_grpo_clevr_count.sh` | 从 base 直接 RL；`num_generations=4`、`per_device_batch=4`、max_completion_length 256、lr 1e-6、beta 0.01、100 步、7 卡训练 + 2 卡 rollout（`scripts/01_serve_vllm.sh`） | **~45 分钟** |
| 4. 评测 + demo | `scripts/16_make_count_demo.py` | 三方自由生成评测（temperature 0.01）+ 渲染对比视频 | ~5 分钟 |

GRPO 的 reward（自定义插件 `scripts/clevr_count_reward_plugin.py`，`--external_plugins` 挂载）：

| reward | 含义 | 权重 |
| --- | --- | --- |
| `count_accuracy` | 数对=1；差 1=0.5；否则 0 | 1.0 |
| `count_format` | 严格 `<think>…</think><answer>…</answer>` 且 JSON 可解析 =1；只有 answer=0.5 | 0.2 |
| `count_brevity` | 200 字符内满分，800 字符以上 0 分 | 0.2 |

## 四、结果指标

300 条验证样本、自由生成、判定口径一致：

| 模型 | 计数准确率 | 格式合规率 | 平均输出 | 训练成本 |
| --- | --- | --- | --- | --- |
| base Qwen2.5-VL-3B-Instruct | **13.3%** | 0% | 29 字符 | — |
| **RL（GRPO 100 步，从 base 训）** | **98.3%** | 0% | 29 字符 | 7 卡 × 45 分钟 |
| **SFT（LoRA r8，150 步）** | **98.7%** | **100%** | 143 字符 | 4 卡 × 8.7 分钟 |

训练过程指标：

* SFT：loss 2.7 → ~0.01 量级，验证 token_acc **0.9997**（含格式模板）
* GRPO：`count_accuracy` reward 0.60 → **1.00**（验证集 0.99），KL 稳定在 0.02~0.04，
  显存 17.8 GB/卡，27 s/iter；`count_format` 停在 0.5（模型选择只报答案，不写推理链），
  输出长度 34 → 11 token

## 五、结论与经验

1. **两条路线都能把 13.3% 提到 98%+**：RL 从零训也能收敛（奖励稠密、判定客观时 RL 很有效）。
2. **reward 权重决定学习形态**：accuracy 权重 1.0、format 仅 0.2 时，RL 学会"只报答案"；
   SFT 因为直接模仿标注，天然带上 `<think>` 推理链。
3. **性价比**：SFT 8.7 分钟 vs RL 45 分钟，同样达到 ~99%；实际项目建议 **先 SFT 打底，再用 GRPO 精调**。
4. 监控要点：`frac_reward_zero_std`（组内无梯度比例）、completion 长度（是否越训越长）、KL
   —— GEOQA 那次就是长度从 258 涨到 440 字符、越训越差，说明需要早停选最佳 checkpoint。
5. 工程注意：GRPO 的服务必须用 `swift rollout`（不是 `swift deploy`）；多帧输入要放开
   `VLLM_LIMIT_MM`；`per_device_eval_batch_size` 也要能被 `num_generations` 整除；
   24G 卡把 `num_generations` 降到 4 可把显存从 23.5GB 压到 15.7GB。

## 六、复现

```bash
cd /data/ljy23/basketball_project/cuttingclip/rl_basketball_video_highlight_detection
source env.sh
python3 scripts/prepare_clevr_count_hard.py --min-count 8 --train 3000 --val 300
python3 scripts/14_make_count_val_infer.py
TRAIN_GPUS=4,5,6,7 MAX_STEPS=150 bash scripts/13_sft_clevr_count_hard.sh
VLLM_GPUS=0,1 bash scripts/01_serve_vllm.sh
TRAIN_GPUS=2,3,4,5,6,7,8 MAX_STEPS=100 START=base bash scripts/17_grpo_clevr_count.sh
python3 scripts/16_make_count_demo.py --out results/clevr_count_hard --models base=... RL=... SFT=...
```

结果包：`results/clevr_count_hard/`（指标、逐 step 日志、三方原始输出、demo 视频）。
