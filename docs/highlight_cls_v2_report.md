# 高光分类 v2：按 playbook 推荐配置的训练结果（失败案例复盘）

按 [`highlight_clip_training_playbook.md`](highlight_clip_training_playbook.md) 第 5 节的推荐配置，
在企业数据（11.19 四个机位 × 三节）上完整跑了一遍 SFT → GRPO。**结论：这一版没有跑出可用的分类器**，
三个模型都停在随机水平，但失败原因定位得很清楚，下面把过程和证据都留下。

## 一、按推荐配置执行的内容

| playbook 建议 | 本次执行 |
| --- | --- |
| 8~16 帧 / 448px | 8 帧 / 448px ✅ |
| 全部正例 + 2~3 倍时间相邻难负例 | 全部正例（196 事件 × 4 视角 = 784）+ 2 倍 adjacent 难负例 ✅ |
| 按节切分 | 1、2 节训练（1428 条），第 3 节验证（924 条）✅ |
| 先 SFT（LoRA r8，1~3 epoch） | LoRA r8，200 步（≈2.1 epoch），4 卡 22.8 分钟，eval_token_acc 0.955 ✅ |
| 再 GRPO（num_generations 4~8、beta 0.01~0.03、lr 1e-6、max_completion 256、LoRA 或 offload） | 全参（LoRA+server 权重同步在本组合下报 `MergedColumnParallelLinear has no attribute base_layer`，改为合并 LoRA 后全参）、num_generations 4、beta 0.01、lr 1e-6、100 步、6 卡训练 + 2 卡 rollout ✅ |
| reward 换组内平衡 + 整组答案相同给 0 | 新增 `highlight_label_balanced`（判对 1.0 + 事件词 0.2，整批预测 >90% 同一标签时整体 −0.3）+ `highlight_format` 0.2 + `highlight_evidence` 0.3 ✅ |
| 评测 balanced accuracy / P / R，n≥200 | 200 条分层子集（正负各 100）✅ |

## 二、结果（200 条分层验证样本，判定口径一致）

| 模型 | balanced accuracy | accuracy | Precision | Recall | 格式 | 行为 |
| --- | --- | --- | --- | --- | --- | --- |
| base Qwen2.5-VL-3B | 50.0% | 50.0% | 50.0% | 100% | 0% | 全部答 yes |
| SFT（LoRA 200 步） | 50.0% | 50.0% | 0% | 0% | 100% | 全部答 no |
| GRPO（100 步，从 SFT 继续） | 50.0% | 50.0% | 0% | 0% | 100% | 全部答 no（与 SFT 完全一致） |

（在完整 924 条验证集上，base 33.3% / SFT 66.7%——因为验证集是 1:2 正负比，恒答 no 恰好拿 66.7%；
换成平衡子集后两者都回到 50%。）

## 三、为什么没学会：证据与归因

1. **SFT 阶段就塌到了多数类**：训练数据正负比 1:2，SFT 学到"理由 + `<answer>no</answer>`"这个模板，
   在**不看图**的情况下也能拿 66.7% 的 token 准确率（teacher forcing 的 token_acc 0.955 就是这么来的）。
   输出 100% 合规但没有任何判别力。
2. **GRPO 阶段完全没有梯度**：训练日志里 `frac_reward_zero_std = 1.00`（最后 10 步全部为 1.0），
   即每个 prompt 的 4 条 completion 答案完全一样 → advantage 恒为 0 → 策略不动。
   佐证：KL 全程只有 0.002~0.003（几乎没偏离 SFT），reward 从 0.83 微降到 0.73。
3. **反退化惩罚救不了它**：我们加的"整批同一标签扣 0.3"只是让退化策略的奖励更低，
   但**均匀扣分不会产生组内方差**，所以还是没有梯度。RL 需要的是"同一条 prompt 的不同回答得分不同"。
4. **输入信息可能本来就不足**：5 秒片段抽 8 帧、448px、固定宽视角，而负例是"离高光很近的窗口"
   （画面几乎一样），模型很难从这几帧里看出"球有没有进 / 这次进攻是否成功"。

## 四、下一步要改什么（按优先级）

1. **先修 SFT**：训练集按正负 1:1 采样（或对正例加权），并**在 SFT 阶段就报 balanced accuracy**，
   只有 SFT 的 balanced accuracy 明显 > 50%（比如 0.65+）才值得进 RL。当前 50% 说明 SFT 没学会，
   这时候上 GRPO 是浪费时间。
2. **给 RL 留出探索空间**：`temperature` 1.0 → 1.3~1.5，`num_generations` 4 → 8~16，
   并在训练时监控 `frac_reward_zero_std`；如果 > 0.5，先别指望 RL 会动。
3. **加大视觉信息**：16 帧 / 672px，或改用"出手瞬间"关键帧 + 计分板/球筐区域裁剪；
   也可以把 4 个机位都作为输入（多视角同时给模型）。
4. **换更省数据的两段式**：先用规则引擎/时序模型（VideoMAE + 分类头）做候选召回，
   再让 VLM 只做"二选一/打分"，把难度降下来；或用 RFT（拒绝采样）先筛出判对的轨迹做 SFT。
5. 数据量：即使按 playbook 的建议，784 个正例对"从视频判断高光"仍然偏少；建议通过主动学习
   把正例扩到 3000+（模型预标 + 人工修正），再做 RL。

## 五、本次产物位置

* 数据：`data/highlight_cls_v2/`（12 个 jsonl，1428 训练 / 924 验证，25 分钟切好）
* SFT：`output/sft_highlight_cls/v0-20260913-094230/checkpoint-200`（LoRA 172MB）
* GRPO：`output/grpo_highlight_cls/v2-20260913-102644/checkpoint-100`
* 结果：`results/highlight_cls_v2/`（`metrics/eval_200.json`、`metrics_logging_*.jsonl`、
  `vlm_outputs/*.jsonl`、`samples/vlm_answers_200_readable.md`、训练参数 `configs_args_*.json`）
* 训练日志：`logs/sft_highlight_v2.log`、`logs/grpo_highlight_v2.log`
