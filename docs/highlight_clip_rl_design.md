# 篮球高光剪辑：reward 设计 + 训练方案

面向 `/data/ljy23/data/videodata/11.19` 的数据，基于已经跑通的 swift GRPO 环境
（Qwen2.5-VL-3B + vLLM rollout + SwanLab）。

## 0. 数据盘点（实测，2026-09-12）

| 项目 | 实测值 |
| --- | --- |
| 视频 | 12 个 mp4，1920×1080@30fps，**4 个机位（A1/A2/B3/B4）× 3 节**，每节 10:58 / 8:10 / 6:29，共 9.5G |
| 标注 | `highlight/{1,2,3}-3v3.json`，键 `000…`，字段 `StartTime/EndTime/Description/ID` |
| 高光段 | 55 段可用（56 段中 1 段 `EndTime` 字段损坏：`"00:01:"`），单段 3–25s，均值 5.5s |
| 标注时长 | 302s / 1537s ≈ 20% |
| 滑窗候选 | 单机位 5s 窗 / 1s 步长 → 1525 候选 / 196 正例（12.9%）；**四机位合计 6100 候选 / 784 正例** |

数据组织（已确认）：

1. **A1/A2/B3/B4 是同一场比赛的四个机位**，`-1/-2/-3` 是三节（10:58 / 8:10 / 6:29）。
   所以 `1-3v3.json` 标的是第 1 节，**四个机位的 -1 视频共用同一条时间轴**（2/3 节同理）。
   脚本默认映射就是「一个 json → 四个机位视频」，可用 `--views A1,B3` 限定。
2. 由此带来两个直接好处：① 数据量 ×4（784 个正例窗口），② 可以做**多视角一致性** reward
   （同一时刻不同机位应给出相同判断）。
   注意：**划 train/val 必须按「节 / 事件」划分，不能按机位划分**，否则同一事件的不同视角
   会同时出现在训练和验证集里，指标虚高。
3. `ID` 字段是参与球员的球衣号（1/5/6/7/8/9/10），可作为“球员归因”类 reward 的弱标签。

结论：**55 段标注够做 SFT 和少量 RL 微调，但远不够“从零 RL 训一个 VLM”**。正确顺序是
「候选生成 → SFT 对齐格式与口径 → GRPO 对齐“你的剪辑口味”」，并且 RL 阶段必须靠
密集/可验证 reward 补足标注量。

## 1. 先把任务定义清楚（三选一）

| 方案 | 输入 | 输出 | 训练方式 | 适合 |
| --- | --- | --- | --- | --- |
| **A. 候选二分类/打分（推荐起步）** | 一个 5s 候选片段（8 帧） | yes/no + 理由 | 有监督 SFT + GRPO | 数据少、要快速见效 |
| **B. 列表式剪辑（RL 主战场）** | 一段 30–120s 视频帧 | 时间区间列表 JSON | GRPO（reward 可验证） | 端到端出剪辑点 |
| **C. 成对偏好排序** | 两个候选片段 | 谁更精彩 | RM / DPO | 回答“哪段更好”最稳 |

建议：**A 做主线**（已有脚本 `prepare_highlight_dataset.py --task classify`），
**B 作为端到端目标**（`--task clip_list`），C 等标注扩到 300+ 段后再上。

## 2. Reward 设计

已在 `scripts/highlight_reward_plugin.py` 实现，通过 `--external_plugins` 挂载。
总 reward = 加权和（swift 用 `--reward_weights`，或内部归一化）：

```
R = Σ wᵢ·rᵢ / Σ wᵢ
```

| reward | 作用 | 形式 | 建议权重 | 备注 |
| --- | --- | --- | --- | --- |
| `highlight_format` | 保证输出可解析 | 0 / 0.5 / 1 | 0.2 | 没有 `<answer>` 直接 0 |
| `highlight_label` | **分类主奖励**：yes/no 与标注一致；理由里事件关键词也命中再 +0.2 | 0~1.2 → 归一 | 1.0 | 抓“是不是高光” |
| `highlight_reason` | 理由与 `Description` 的事件词重合度 | 0~1 | 0.3 | 把短标注变成密集信号 |
| `highlight_temporal` | **列表主奖励**：预测区间与标注区间匹配的 F1，tIoU≥0.3 起给分、0.5 满分 | 0~1 | 1.0 | 抓“剪在哪” |
| `highlight_length` | 时长占比（理想 8%~30%）、片段数上限、重叠惩罚 | 0~1 | 0.3 | 反“全剪/乱剪” |
| （可选）规则事实一致性 | 与现有规则引擎/轨迹给出的事实（命中、距离、抢断）对齐 | 0 / 1 | 0.2 | 你已有规则引擎，可复用 |
| （可选）球员 ID | `ID` 集合是否体现在理由里（或轨迹/球衣号识别命中） | 0 / 1 | 0.1 | 弱标签，别给太大 |

关键公式（`highlight_temporal`）：

```
tIoU(p,g) = |p∩g| / |p∪g|
score(p)  = 1                      if tIoU ≥ 0.5
          = (tIoU-0.3)/(0.5-0.3)   if 0.3 ≤ tIoU < 0.5
          = 0                      otherwise
precision = 命中数/预测段数, recall = 命中数/标注段数
r_temporal = 0.5·F1 + 0.5·(Σscore/预测段数)
```

### 反 reward hacking（这几个坑一定要堵）

* **全剪**：预测 `[[0, 600]]` → tIoU 很低 + `highlight_length` 扣分（已验证：该输出 temporal=0，length=0.6）。
* **只剪一段**（保 precision）：`recall` 项会掉分，F1 拉住。
* **刷数量**（保 recall）：`precision` 项 + 片段数上限 + 重叠惩罚。
* **复读格式**：`highlight_format` 只占 0.2，判对/剪对才拿大分。
* 建议 reward 分档不要用二值（tIoU 分段给分），否则 GRPO 组内全 0 没梯度。

## 3. 数据准备

```bash
cd /data/ljy23/basketball_project/cuttingclip/rl_basketball_video_highlight_detection
source env.sh

# 先确认映射与数据量
python3 scripts/prepare_highlight_dataset.py --mode stats

# 分类任务（RL 数据 + 可选 SFT 的 assistant 目标）
python3 scripts/prepare_highlight_dataset.py --mode cut --task classify \
    --window 5 --stride 1 --frames 8 --neg-ratio 2 --with-answer \
    --out-dir data/highlight_cls

# 列表任务（30s 一段，模型输出该保留的区间）
python3 scripts/prepare_highlight_dataset.py --mode cut --task clip_list \
    --window 30 --stride 30 --frames 16 --out-dir data/highlight_list
```

产物：`data/highlight_cls/<场次>.jsonl`，每行含 `messages`（+ `images` 帧路径）+ `solution`
（JSON：`is_highlight/interval/max_iou/description/ids`）。**按场次划分 train/val**
（例如 `1-3v3` 训练、`3-3v3` 验证），同场次内的窗口不要跨集合，否则泄漏。

抽帧策略：每个 5s 候选 8 帧、宽 448（`MAX_PIXELS=262144`），单条约 8 张图；实测
720P 帧裁剪后 1 条样本 ≈ 300KB，1525 条候选 ≈ 0.5GB，完全可承受。

## 4. 训练流程

### Stage 1：SFT（LoRA）——教格式和判断口径

```bash
NPROC_PER_NODE=4 CUDA_VISIBLE_DEVICES=4,5,6,7 swift sft \
  --model "$SWIFT_RL_MODEL" \
  --tuner_type lora --lora_rank 8 --target_modules all-linear \
  --dataset data/highlight_cls/1-3v3.jsonl data/highlight_cls/2-3v3.jsonl \
  --val_dataset data/highlight_cls/3-3v3.jsonl \
  --max_length 4096 --max_pixels 262144 \
  --per_device_train_batch_size 1 --gradient_accumulation_steps 4 \
  --num_train_epochs 3 --learning_rate 1e-4 \
  --freeze_vit true --freeze_aligner true \
  --report_to swanlab --swanlab_mode local --swanlab_project highlight-clip-sft \
  --output_dir output/sft_highlight_cls
```

### Stage 2：GRPO——用可验证 reward 对齐剪辑口味

```bash
# rollout 服务（占 2,3 卡）
VLLM_GPUS=2,3 bash scripts/01_serve_vllm.sh > logs/rollout_server.log 2>&1 &

# 训练（占 4-7 卡）；LoRA 时加 --vllm_enable_lora true --vllm_max_lora_rank 8
CUDA_VISIBLE_DEVICES=4,5,6,7 NPROC_PER_NODE=4 swift rlhf --rlhf_type grpo \
  --model "$SWIFT_RL_MODEL" \
  --adapters output/sft_highlight_cls/v0-*/checkpoint-* \
  --tuner_type lora --lora_rank 8 \
  --dataset data/highlight_cls/1-3v3.jsonl data/highlight_cls/2-3v3.jsonl \
  --external_plugins scripts/highlight_reward_plugin.py \
  --reward_funcs highlight_format highlight_label highlight_reason \
  --reward_weights 0.2 1.0 0.3 \
  --use_vllm true --vllm_mode server --vllm_server_host 127.0.0.1 --vllm_server_port 8000 \
  --max_pixels 262144 --max_length 4096 --max_completion_length 512 \
  --num_generations 4 --temperature 1.0 --beta 0.001 --learning_rate 5e-7 \
  --per_device_train_batch_size 2 --gradient_accumulation_steps 2 \
  --deepspeed zero3 --report_to swanlab --swanlab_mode local \
  --swanlab_project highlight-clip-grpo --output_dir output/grpo_highlight_cls
```

列表式任务把 `--reward_funcs` 换成 `highlight_format highlight_temporal highlight_length`
（权重 0.2/1.0/0.3），并把 `--task clip_list` 的数据接上。

参考成本（昨天同机型实测）：Qwen2.5-VL-3B、4 卡全参 GRPO ≈ 12.3 s/iter、23 GiB/卡；
LoRA 会更省。1525 条候选按 batch 8 计算，1 epoch ≈ 190 iter ≈ 40 分钟。

### 超参建议

| 参数 | 建议 | 说明 |
| --- | --- | --- |
| `num_generations` | 4~8 | 组内比较，太小优势估计噪声大 |
| `temperature` | 0.8~1.0 | 探索剪辑点 |
| `beta`(KL) | 0.001~0.01 | 防止跑偏；输出格式类任务可小一些 |
| `learning_rate` | 5e-7~1e-6（LoRA）/ 1e-6（全参） | |
| `max_completion_length` | 512（分类）/ 1024（列表） | 列表要放得下 JSON |
| `max_grad_norm` | 0.5 | 数学/多模态 GRPO 常见不稳，官方实践也这么设 |

## 5. 评估（离线 + 人工）

* 定位类：`mAP@tIoU{0.3,0.5}`、`Recall@5`、`Precision@1`；
* 选择类：剪出的高光总时长 vs 标注总时长（误差越小越好）、命中率（剪出的段有多少落在标注里）；
* 一致性：让模型对同一片段采样 8 次，看 yes/no 的方差（RL 后应明显下降）；
* 人工抽检：抽 20 段看「是否会漏掉明显好球 / 是否剪进垃圾时间」，这是最终验收标准。

## 6. 如果 RL 效果不稳，这些更省数据的办法优先做

1. **滑窗基线**：视频编码器（VideoMAE/InternVideo2）+ 时序头做 5s 窗二分类。
   1525 正负样本就够训一个可用的召回器，用来做候选生成，再交给 VLM 打分。
2. **弱标签扩样**：若已有“剪好的成片/发布视频”，用它和整场做时间对齐，秒级弱标签
   能把标注从 55 段扩到几千段（这是最划算的一步）。
3. **主动学习**：用当前 SFT 模型给未标注视频打伪标签 → 人工只改错的部分，
   一轮就能把标注翻几倍。
4. **排序优先**：把“给绝对分”换成“A 比 B 更精彩”，标注一致性更高，
   也天然适合 GRPO/DPO。

## 7. 待确认 / 下一步

- [ ] 确认 `highlight/*.json` 到底对应哪些视频（影响全部训练数据）。
- [ ] 修 `2-3v3.json` 里那条坏时间字段 `"00:01:"`。
- [ ] 决定主任务是 A（候选二分类）还是 B（列表剪辑）。
- [ ] 跑 `prepare_highlight_dataset.py --mode cut` 生成正式数据（当前 `data/_smoke` 只是
      6 条样本的连通性验证）。
- [ ] 先 SFT，再 GRPO；两条曲线都进 SwanLab 对比。
