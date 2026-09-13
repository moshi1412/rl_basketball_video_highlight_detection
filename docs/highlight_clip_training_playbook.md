# 企业数据（篮球高光）训练手册

这份手册是把「ClevrCount → GEOQA」两次实验 + 今天在 11.19 数据上跑通的
「切片 → SFT → GRPO」串起来，给下一次正式训练用的操作手册。所有坑都是实测踩过的。

## 0. 先明确该用什么训练方式（SFT / RL 各干什么）

| 阶段 | 作用 | 何时必须做 | 实测耗时 |
| --- | --- | --- | --- |
| **数据准备**（切片段+抽帧） | 把整场视频变成 VLM 能吃的样本 | 总是 | 360 条样本 ≈ 3 分钟 |
| **SFT（LoRA）** | 教「输出格式 + 任务口径」，让模型会用标注的写法回答 | 当 base 输出格式不对（实测 base 的 format reward = 0）时**必做** | 30 步 ≈ 2 分钟 |
| **GRPO（RL）** | 用可验证 reward 把你真正的偏好刷上去（能超过标注答案） | SFT 之后；有「难但可验证」的指标时 | 30 步 ≈ 13 分钟 |

为什么不是一上来就 RL：GRPO 的第一步是**采样 + 组内比较**，如果模型连格式都输出不对，
组内 reward 全是 0（`frac_reward_zero_std` 高）→ 没有梯度，RL 的前几十步都在替你「补 SFT 的课」。
GEOQA 实测：base 的 format reward = 0，GRPO 前 30 步主要在学格式，50 步之后 accuracy 才开始涨（0.177→0.242）。
先 SFT 2 分钟把格式教会，RL 的算力才能花在真正难的部分上。

## 1. 数据准备：切候选片段 + 抽帧

```bash
python3 scripts/prepare_highlight_dataset.py --mode stats            # 先看数据量
python3 scripts/prepare_highlight_dataset.py --mode cut --task classify --with-answer \
    --window 5 --stride 1 --frames 8 --width 448 --neg-ratio 1 \
    --out-dir data/highlight_cls
```

要点与实测：

* **A1/A2/B3/B4 是同一场比赛的 4 个机位**，`1/2/3-3v3.json` 分别标的是第 1/2/3 节；
  同一事件的 4 个视角画面不同、时间轴相同 → **划 train/val 必须按「节/事件」切，
  绝不能按机位切**，否则同一时刻的不同画面会同时出现在训练和验证集里（指标虚高）。
  本工程默认：1、2 节训练，3 节验证。
* 抽帧数与分辨率决定"看不看得懂"：本次为省时间只用了 **4 帧 / 336px**，
  结果三个模型都学不会分辨高光（见第 4 节）。**正式训练建议 8~16 帧 / 448~672px**，
  必要时把 5s 片段截成「出手前→进球后」的关键窗口。
* 负例不要纯随机抽：优先用**时间相邻**的窗口（如高光前后 10~30s），才是「像高光但不是」的难例。

## 2. SFT：教格式与口径（LoRA，多卡）

```bash
TRAIN_GPUS=4,5,6,7 MAX_STEPS=30 bash scripts/10_sft_highlight_cls.sh
```

实测（240 条训练 / 120 条验证，4 卡 LoRA r=8，max_pixels 100352）：

| 指标 | 数值 |
| --- | --- |
| loss | 2.69 → 0.14~0.25 |
| token_acc（训练） | 0.53 → 0.96 |
| token_acc（第 3 节验证） | **0.911** |
| 显存 / 产物 | 8.6 GB/卡，LoRA 权重仅 172 MB |
| 耗时 | 30 步 ≈ 2 分钟 |

## 3. GRPO：用可验证 reward 对齐偏好

```bash
VLLM_GPUS=0,1 MAX_PIXELS=100352 VLLM_LIMIT_MM='{"image":4}' bash scripts/01_serve_vllm.sh
VLLM_PORT=8001 TRAIN_GPUS=2,3,4,5,6,7,8 NUM_GEN=4 MAX_STEPS=30 bash scripts/11_grpo_highlight_cls.sh
```

实测（30 步，7 卡全参，28 completions/步）：reward 0.66→0.65，其中
`HighlightFormat` 0.86 → **0.94**（格式在学），`HighlightLabel` 0.45 → 0.41（没动），
`HighlightReason` 0.13 → 0.16（很弱），KL 0.0008 → 0.015（很稳），显存 17.8 GB/卡。

### 必踩的 5 个坑（都已在本工程脚本里修好）

1. **`swift rollout` 而不是 `swift deploy`**：GRPO 训练端要调 `/init_communicator`、
   `/update_named_param` 等权重同步接口，`swift deploy` 不注册这些路由 → 404。
2. **多帧片段要放开 vLLM 的图片数限制**：默认 `limit_mm_per_prompt={"image":1}`，
   送 4 帧会 500 Internal Server Error → `VLLM_LIMIT_MM='{"image":4}'`。
3. **`per_device_eval_batch_size` 也要能被 `num_generations` 整除**，否则直接报
   `global eval batch size (7 x 1) must be evenly divisible by num_generations_eval (4)`。
4. **显存**：24G 卡上 3B 全参 GRPO 很容易贴到 23.5/24 GB，rank0 OOM 后整个 job 卡死
   （我们 GEOQA 那次就是这样挂的）。降显存三件套：`num_generations` 8→4、
   改 `tuner_type lora`、`--offload_optimizer true`；监控 `memory(GiB)` 别超 21。
5. **每次 run 单独日志文件**：否则并发/重试的日志混在一起，排查时很难看清
   （GEOQA 那次日志被 NCCL 心跳刷到 298 MB）。

## 4. 今天在 11.19 数据上的可行性结论（重要）

在留出的第 3 节（120 个候选片段，正负各 60）上评测：

| 模型 | 分类准确率 | 行为 |
| --- | --- | --- |
| base | 50.0% | 120/120 全答 yes（恒正） |
| SFT-30 步 | 50.0% | 120/120 全答 no（恒负） |
| GRPO-30 步 | 55.0% | 114/120 答 yes（基本恒正） |

**结论：训练链路完全跑通了（切片→SFT→GRPO→评测），但当前任务设定下模型还分辨不出高光**，
三个模型都退化成了「恒答 yes/no」。原因按影响排序：

1. **输入信息不够**：5s 片段只抽 4 帧、336px，宽视角下看不清「球有没有进、防守多紧」；
2. **数据太少**：每节每机位只取了 15 个正例（合计 240 条训练样本）；
3. **reward 可被常数答案刷到 50%**：验证集正负各半，恒答 yes/no 就拿 50 分 → GRPO 组内几乎没有
   区分度（`HighlightLabel` 一直在 0.5 上下抖）；
4. **负例太容易**：随机抽的负例往往是远景空场，模型学会「看着像不像比赛画面」而不是「是不是高光」。

## 5. 下一次正式训练的推荐配置

| 项目 | 建议 |
| --- | --- |
| 帧 | 8~16 帧 / 448~672px，必要时用「出手瞬间」关键帧 |
| 样本量 | 每个机位每节**全部正例**（55 段 × 4 视角 ≈ 220 正例/节）+ 2~3 倍**时间相邻**难负例 → 2000+ 条 |
| 划分 | 按节或按场次切分（同事件多视角必须同侧） |
| 先 SFT | LoRA r=8，1~3 epoch，教格式与口径（本次 0.911 token_acc，够用） |
| 再 GRPO | `num_generations=4~8`、`beta=0.01~0.03`、`lr=1e-6`、`max_completion_length=256`、LoRA 或 `offload_optimizer` |
| reward | 分类主奖励换成**组内平衡指标**（balanced accuracy / F1），并对「整组答案相同」直接给 0；再叠加事件词匹配、时长先验；列表式任务用 `highlight_temporal` 的 tIoU-F1 |
| 评测 | 固定留出某一场/某一节，报 balanced accuracy + Precision/Recall；n≥200 |
| 监控 | SwanLab 看 `rewards/Highlight*`、KL、completion 长度；KL 突增或长度塌缩就早停 |

短期更稳的替代路线（数据量还没上来时）：

* **RFT / 拒绝采样**：用 base 或 SFT 模型对候选片段采样，挑出判对的轨迹做 SFT，比直接 GRPO 稳；
* **两段式**：先用规则引擎/时序模型（VideoMAE + 分类头）做候选召回，再让 VLM 只负责
  「二选一/打分」，把 VLM 的任务难度降下来。
