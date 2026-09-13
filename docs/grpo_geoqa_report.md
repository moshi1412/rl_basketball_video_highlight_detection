# GEOQA 多模态 GRPO 实验报告（换难任务 + 调超参 + 多卡）

> 目的：ClevrCount 那次 GRPO 没有超过 base（base 已经 80%，任务没有提升空间），
> 所以换到 base 明显不会的 **GEOQA 几何问答**，并按 GRPO 的经验修正超参：加大组大小、
> 加强 KL 约束、用 7 卡训练 + 2 卡 rollout。

## 配置对比

| 项目 | ClevrCount（上一版） | GEOQA（本次） |
| --- | --- | --- |
| 数据 | `clevr_cogen_a_train#2000`（数物体，base 已 80%） | `GEOQA_R1V_Train_8K`（几何推理，base 只有 ~20%） |
| 训练卡 | 4×RTX3090 | **7×RTX3090**（2–8） |
| rollout | 2 卡（TP=2，server 模式） | 2 卡（TP=2，server 模式） |
| `num_generations` | 2 | **8** |
| completion 数/步 | 8 | 56 |
| `beta`（KL 系数） | 0.001 | **0.01** |
| `max_completion_length` | 512 | **1024** |
| `repetition_penalty` | 1.0 | **1.1** |
| 步数 | 100 | 200 |
| 其它 | lr 1e-6 / temperature 1.0 / zero3 | 同（+ warmup_ratio 0.05、`--system` 约束输出格式） |

## 结果

### 在 100 条独立抽样上的评测（判定用训练时同一套 accuracy reward / math_verify）

| 模型 | accuracy | 输出格式合规率 | 平均输出长度 |
| --- | --- | --- | --- |
| base Qwen2.5-VL-3B-Instruct | 24.0% | 0% | 20 字符 |
| GRPO 50 步 | 29.0% | 99% | 339 字符 |
| **GRPO 100 步** | **32.0%** | **100%** | 125 字符 |

### 训练过程中每 50 步的验证集

| step | eval accuracy | eval format | eval reward | eval KL |
| --- | --- | --- | --- | --- |
| 50 | 0.177 | 0.956 | 1.133 | 0.215 |
| 100 | 0.242 | 0.992 | 1.234 | 0.513 |
| 150 | 0.289 | 0.994 | 1.282 | 0.228 |
| 200 | 0.276 | 0.987 | 1.263 | 0.091 |
| 300 | 0.269 | 0.994 | 1.263 | 0.097 |
| 400 | 0.256 | 0.997 | 1.253 | 0.089 |

## 续训（checkpoint-100 → 400 步）与"训过头"现象

第一次跑到 ~124 步因 rank0 显存 OOM 中断。第二次用修正配置从 checkpoint-100 续训到 400 步：
`num_generations` 8→4（显存 23.5GB → **15.7GB/卡**，不再 OOM）、`per_device_train/eval_batch_size`=4、
`beta` 0.01→0.03（KL 从 0.5 回落到 0.09）、29.5 s/it。

在 100 条独立抽样上的最终评测：

| 模型 | accuracy | format | 平均输出 |
| --- | --- | --- | --- |
| base | 24.0% | 0% | 20 字符 |
| GRPO-100 步 | 32.0% | 100% | 125 字符 |
| **GRPO-150 步（最佳）** | **39.0%** | 100% | 258 字符 |
| GRPO-400 步（最终） | 34.0% | 100% | 440 字符 |

**结论：150 步是甜点（24%→39%，相对 +62%），继续训到 400 步反而掉到 34%，输出从 258 涨到 440 字符
（开始兜圈子）**。教训：GRPO 不是越久越好，要用验证集选最佳 checkpoint 并早停。

### 训练曲线特征

* **format reward**：0.06 起步 → 30 步左右到 0.6-0.7 → 60 步后稳定 1.0（彻底学会 `<think></think><answer></answer>`）；
* **accuracy reward**：base 约 0.17→0.19，训练后期在 0.2~0.55 之间抖动（单步 batch 只有 7 个 prompt，噪声大），
  验证集口径是 0.177 → 0.242；
* **completion 长度**：300 token → 30~80 token（学会简短推理），eval 长度 89 → 44；
* **KL**：从 0 涨到 0.5，说明 beta=0.01 仍偏小、策略漂移明显；
* 没有出现 ClevrCount 那种复读collapse（`repetition_penalty=1.1` 起作用了），但输出里有轻微中英混杂。

## 中断与教训

* 训练在 **step ~124/200** 中断：`rank0` 显存 OOM（`expandable_segments: memory mapping failed
  ... free: 1.3MB`），训练卡长期贴在 23.5/24 GB；rank0 挂掉后其余 rank 卡在 NCCL 里空转，
  rollout 服务也随之退出，日志被 NCCL heartbeat 刷到 298MB。
* 已保存的 **checkpoint-50 / checkpoint-100 都完好**，评测与 demo 用的就是它们。
* 复训建议：
  1. 降显存：`num_generations=4`（`per_device_train_batch_size` 同步降到 4）或改 `tuner_type lora`，
     或 `--offload_optimizer true`；把占用从 23.5GB 压到 20GB 以下；
  2. 拉长步数：官方 GEOQA 实践约 1300 步，200 步只是起步；
  3. 压制漂移：beta 提到 0.02~0.05，并盯 KL/长度曲线；
  4. 每次 run 的日志单独一份文件（本次并发了两次尝试，日志混在一起难以排查）。
