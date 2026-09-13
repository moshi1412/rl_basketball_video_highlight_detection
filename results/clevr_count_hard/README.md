# 任务 B：CLEVR 高计数难题（base / RL / SFT 三方对比）

目的：做一个**演示直观**的任务——base 明显会错、判定客观、错法一眼可见，用来展示"训练后模型变好"。

## 任务设计

| 项目 | 设置 |
| --- | --- |
| 数据 | 本地缓存的 CLEVR 图（离线读取，无需联网），**只保留物体数 ≥8 的难题** |
| 样本量 | 3000 训练 / 300 验证（8、9、10 个物体各约 1/3） |
| 输出格式 | `<think>…</think><answer>{"count": N}</answer>` |
| 判定 | accuracy = JSON 里的 N 与 GT 完全一致；format = 严格 `<think>…</think><answer>…</answer>` |

## 结果（300 条验证样本，自由生成、temperature 0.01）

| 模型 | 计数准确率 | 格式合规率 | 平均输出 | 训练成本 |
| --- | --- | --- | --- | --- |
| base Qwen2.5-VL-3B-Instruct | **13.3%** | 0% | 29 字符（裸答案） | — |
| **RL（GRPO 100 步，从 base 直接训）** | **98.3%** | 0% | 29 字符（裸答案） | 7 卡 × ~45 分钟 |
| **SFT（LoRA r8，150 步）** | **98.7%** | **100%** | 143 字符（带推理链） | 4 卡 × 8.7 分钟 |

三方都跑在同一批 300 条验证样本上，判定脚本一致（`scripts/16_make_count_demo.py`）。

## 结论（两种训练路线都有效，但形态不同）

1. **RL 单独就能把 13.3% 拉到 98.3%**：GRPO 的 `CountAccuracy` reward 从 0.60 → 1.00
   （验证集 0.99），说明这个任务对 RL 很友好（奖励稠密、判定客观、有 headroom）。
2. **但它学成了"只报答案"**：`CountFormat` reward 停在 0.5（只满足 `<answer>`，没有 `<think>`），
   输出长度从 34 token 塌到 11 token。原因是 reward 权重里 accuracy 占 1.0、格式只占 0.2，
   模型发现"多写推理链"的边际收益不划算 → 这是典型的 **reward 权重决定学习形态**。
3. **SFT 更省（8.7 分钟 vs 45 分钟）且能带出推理链**（格式 100%），因为它直接模仿标注写法。
4. 对演示来说两种都好看：RL 版能讲"强化学习真的把 13% 干到 98%"，SFT 版能讲"还学会了写推理过程"。

## 文件

```
README.md                 本文件
demo/clevr_count_demo.mp4 三方对比 demo（20 题、60 秒、1280×720）：
                          左边 CLEVR 图，右边 base / RL(GRPO-100) / SFT 的答案与 ✅❌，
                          底部显示累计准确率与格式合规率
metrics/eval.json         三方准确率/格式/长度
metrics/logging_*.jsonl   SFT 与 GRPO 的逐 step 指标
configs/args_*.json       两次训练的完整参数
vlm_outputs/*.jsonl       三方在 300 条上的原始输出
samples/                  推理输入、逐样本判定（jsonl + 可读 md）
figures/frames_example/   demo 截图
```

## 复现

```bash
cd /data/ljy23/basketball_project/cuttingclip/rl_basketball_video_highlight_detection
source env.sh

# 0) 数据（离线读 CLEVR 缓存，只留 count>=8）
python3 scripts/prepare_clevr_count_hard.py --min-count 8 --train 3000 --val 300
python3 scripts/14_make_count_val_infer.py

# 1) SFT（LoRA，4 卡，约 9 分钟）
TRAIN_GPUS=4,5,6,7 MAX_STEPS=150 bash scripts/13_sft_clevr_count_hard.sh

# 2) RL（GRPO，7 卡 + 2 卡 rollout，约 45 分钟）
VLLM_GPUS=0,1 MAX_PIXELS=262144 bash scripts/01_serve_vllm.sh
TRAIN_GPUS=2,3,4,5,6,7,8 MAX_STEPS=100 START=base bash scripts/17_grpo_clevr_count.sh

# 3) 评测 + demo
CUDA_VISIBLE_DEVICES=0 swift infer --model "$SWIFT_RL_MODEL" \
    --val_dataset results/clevr_count_hard/samples/infer_input.jsonl \
    --max_pixels 262144 --max_new_tokens 256 --temperature 0.01 --attn_impl sdpa
python3 scripts/16_make_count_demo.py --out results/clevr_count_hard \
    --models base=<base.jsonl> "RL(GRPO-100)"=<grpo.jsonl> SFT=<sft.jsonl>
```
