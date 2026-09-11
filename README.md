# swift-rl：VLM 强化学习（GRPO）环境与框架验证

> 当前工程目录已改名为 `rl_basketball_video_highlight_detection`；
> 面向篮球高光剪辑的 reward 设计与训练方案见
> [`docs/highlight_clip_rl_design.md`](docs/highlight_clip_rl_design.md)，
> 配套脚本：`scripts/prepare_highlight_dataset.py`（造数据）、
> `scripts/highlight_reward_plugin.py`（GRPO reward 插件）、
> `scripts/06_plot_curves.py`（离线画训练曲线）。

## 仓库结构

```
env.sh                         # 公共环境变量（路径自动推导，source 后用）
scripts/
  00_setup_env.sh              # uv 建环境 / 装依赖（含 TE 编译等踩坑命令）
  01_serve_vllm.sh             # GRPO 的 vLLM rollout 服务（swift rollout，不是 swift deploy）
  02_grpo_multimodal.sh        # ms-swift 多模态 GRPO 训练
  03_watch_swanlab.sh          # 打开 SwanLab 本地看板
  04_megatron_grpo.sh          # Megatron-SWIFT GRPO（mcore-bridge）
  05_infer_grpo_checkpoint.sh  # 用训练出的 checkpoint 推理
  06_plot_curves.py            # 从 logging.jsonl 画训练曲线
  07_make_clevr_demo.py        # 抽样 + 渲染 ClevrCount 对比 demo 视频
  prepare_highlight_dataset.py # 篮球高光数据集：滑窗候选 / 正负标注 / 抽帧 / jsonl
  highlight_reward_plugin.py   # 高光剪辑的 5 个 GRPO reward（格式/判对/理由/时序/时长）
docs/
  highlight_clip_rl_design.md  # 高光剪辑 reward 设计与训练方案（含数据盘点）
  grpo_clevr_reward_curve.png  # 训练曲线图
results/clevr_grpo_3b/         # 本次任务的完整结果包（指标/抽样/VLM 回答/demo 视频）
```

> 上传仓库时不需要（也已被 `.gitignore` 排除）：`.venv/`、`.cache/`（模型与数据集缓存）、
> `output/`（checkpoint）、`logs/`、`swanlog/`、`result/`、`data/`。
> 换机器后先 `bash scripts/00_setup_env.sh` 重建环境。

在 8×RTX 3090（24G）上用 **uv** 搭好环境，并用 **ms-swift** 和 **Megatron-SWIFT** 两套
RLHF 后端各跑通了一次多模态 GRPO（ClevrCount 任务 / Qwen2.5-VL-3B-Instruct），
训练过程用 **SwanLab（local 模式，无需账号）** 记录。

## 1. 环境

目录：`/data/ljy23/basketball_project/cuttingclip/swift-rl`（工程可从任意路径 `source env.sh`，路径自动推导）

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Python | 3.12.12（uv 托管） | `uv venv --python 3.12 .venv` |
| torch | 2.13.0+cu130 | 驱动 580.82.09 / CUDA 13.0 |
| ms-swift | 4.5.3 | `ms-swift[swanlab,megatron]==4.5.3` |
| vllm | 0.29.0 | GRPO rollout（server 模式） |
| deepspeed | 0.19.6 | `--deepspeed zero3` |
| transformers | 5.16.1 | |
| megatron-core | 0.19.0 | Megatron-SWIFT |
| mcore-bridge | 1.6.4 | HF 权重免转换直接训练 |
| transformer-engine | 2.18.0（cu13） | 源码编译 torch binding |
| swanlab | 0.10.0（+swanboard） | 本地看板 |

模型：`Qwen/Qwen2.5-VL-3B-Instruct`（多模态、小模型，swift rlhf 与 megatron rlhf 都支持），
缓存在 `.cache/modelscope/models/Qwen--Qwen2.5-VL-3B-Instruct/snapshots/master`。

环境重建：

```bash
uv venv --python 3.12 .venv
export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
uv pip install --python .venv/bin/python "ms-swift[swanlab,megatron]==4.5.3" vllm deepspeed math-verify qwen_vl_utils decord
uv pip install --python .venv/bin/python --prerelease=allow "swanlab[dashboard]"
# megatron 额外依赖见 scripts/00_setup_env.sh 与「踩坑记录」
```

## 2. 目录结构

```
swift-rl/
├── env.sh                    # 公共环境变量（先 source 再用）
├── scripts/
│   ├── 00_setup_env.sh       # 环境安装步骤（可重复执行）
│   ├── 01_serve_vllm.sh      # 启动 GRPO 的 vLLM rollout 服务（swift rollout）
│   ├── 02_grpo_multimodal.sh # ms-swift 多模态 GRPO
│   ├── 03_watch_swanlab.sh   # 打开 SwanLab 本地看板
│   └── 04_megatron_grpo.sh   # Megatron-SWIFT GRPO（mcore-bridge）
├── logs/                     # 各次运行日志
├── output/                   # 训练输出与 checkpoint
├── swanlog/                  # ms-swift 的 SwanLab 日志
└── .cache/                   # modelscope 模型/数据集缓存
```

## 3. 跑一次 ms-swift 多模态 GRPO

```bash
cd /data/ljy23/basketball_project/cuttingclip/swift-rl
source env.sh

# 1) 起 rollout 服务（占 GPU 2,3；注意是 swift rollout 而不是 swift deploy）
VLLM_GPUS=2,3 bash scripts/01_serve_vllm.sh > logs/rollout_server.log 2>&1 &

# 2) 训练（占 GPU 4,5,6,7）
MAX_STEPS=20 bash scripts/02_grpo_multimodal.sh 2>&1 | tee logs/grpo_run.log
```

实测（20 步 + 10 步验证，4 卡全参 + deepspeed zero3）：

| 指标 | 数值 |
| --- | --- |
| 速度 | ~12.3 s/iter（含 rollout） |
| 显存 | 23.0 GiB / 卡 |
| reward | 1.0 ~ 1.75（accuracy 0.25~0.875 + format 0.75~1.0） |
| kl | ~1e-3，grad_norm 2~8 |
| completion | 平均 90~130 token |
| checkpoint | `output/grpo_clevr_3b/v1-*/checkpoint-20`（43G，含优化器状态） |

后来又把步数放大到 100 步跑了一轮（`run-20260911_140252`，总时长约 23 分钟），
accuracy reward 从 0.25 稳定涨到 0.6~0.75，format reward 基本恒为 1.0，completion 从 ~154
token 收敛到 ~85 token，说明 reward 与训练链路是通的：

| step | reward | acc | fmt | kl | len |
| --- | --- | --- | --- | --- | --- |
| 1/100 | 1.125 | 0.250 | 0.875 | 0.000 | 154 |
| 26/100 | 1.375 | 0.375 | 1.000 | 0.012 | 81 |
| 52/100 | 1.625 | 0.625 | 1.000 | 0.023 | 88 |
| 100/100 | — | 0.625 | 1.000 | 0.042 | 84 |

最终 eval：`eval_reward 1.6 / eval_accuracy 0.6 / eval_format 1.0`，
checkpoint 在 `output/grpo_clevr_3b/v2-20260911-140217/checkpoint-100`。

用该 checkpoint 推理（`scripts/05_infer_grpo_checkpoint.sh`，取 4 条验证样本）输出格式正确：

```
<think> ... 逐个物体数数 ... </think><answer>3</answer>
```

4 条里 2 条完全正确（20 步版本）、100 步版本精度更高。

## 4. 跑一次 Megatron-SWIFT GRPO

```bash
source env.sh
VLLM_GPUS=2,3 bash scripts/01_serve_vllm.sh > logs/rollout_server.log 2>&1 &
TRAIN_ITERS=5 bash scripts/04_megatron_grpo.sh 2>&1 | tee logs/megatron_grpo.log
```

采用 **mcore-bridge**（不需要先把 HF 权重转成 Megatron 格式）。实测 5 iter：

| 指标 | 数值 |
| --- | --- |
| 速度 | 11.4 s/iter（首步 29s 含编译） |
| 显存 | 21.2 GiB / 卡 |
| reward | 1.22 ~ 1.25（eval_reward 1.5） |
| checkpoint | `output/megatron_grpo_clevr_3b/v3-*/checkpoint-5`（7.1G，HF safetensors） |

显存关键参数（24G 卡必需）：

```
--main_grads_dtype bf16 --use_precision_aware_optimizer true \
--optimizer_cpu_offload true --optimizer_offload_fraction 1.0 \
--recompute_granularity full --recompute_num_layers 1 \
--no_gradient_accumulation_fusion --freeze_vit true --freeze_aligner true
```

不加前两项时，DDP 会一次性申请 11.5 GiB 的 fp32 grad buffer → OOM。

## 5. 用 SwanLab 看训练过程

ms-swift 走 `--report_to swanlab --swanlab_mode local --swanlab_project swift-grpo-vlm`，
日志写到 `swanlog/`；Megatron 的写在 `output/megatron_grpo_clevr_3b/v*/swanlab/`。

```bash
source env.sh
bash scripts/03_watch_swanlab.sh     # 默认 http://127.0.0.1:5092
# 若在本地机器上，先做端口转发：ssh -L 5092:127.0.0.1:5092 <user>@<host>
# Megatron 的曲线：swanlab watch output/megatron_grpo_clevr_3b/v3-*/swanlab
```

想改用云端 SwanLab：`swanlab login` 后把 `SWANLAB_MODE` 设为 `cloud` 并传
`--swanlab_token/--swanlab_project/--swanlab_workspace`。

## 6. 踩坑记录（重要）

1. **GRPO 的 vLLM 服务要用 `swift rollout`，不是 `swift deploy`。**
   `swift deploy` 只注册 `/v1/*` 和 `/infer/`，训练端调用 `/init_communicator/`、
   `/update_named_param/`、`/get_world_size/` 会 404（`Server 0 failed: {"detail":"Not Found"}`）。
   `swift rollout` 会把这些路由注册好，并把 `WeightSyncWorkerExtension` 注入 vLLM 的 Worker
   （日志里能看到 `Injected ... extended collective_rpc calls [...]`）。vLLM 0.29 虽然删掉了
   TRL 风格的 `update_named_param`，但 swift 自己 patch 了，所以 0.29 也能用。
2. **SwanLab 0.10 的环境变量坑**：`SWANLAB_PROJECT` / `SWANLAB_RUN` 等是 pydantic 的
   *嵌套* 字段，env 值必须是 JSON，否则报
   `SettingsError: error parsing value for field "project"`。
   → 只设 `SWANLAB_MODE=local`、`SWANLAB_LOG_DIR=...`，project 名用 `--swanlab_project` 传。
3. **TransformerEngine**：`pip install transformer-engine[pytorch]` 会去 GitHub releases 下预编译
   wheel（本机不通），只能源码编译：
   - `uv pip install transformer-engine-cu13==2.18.0 transformer-engine==2.18.0`
   - 再编译 torch binding：`NVTE_FRAMEWORK=pytorch NVTE_WITH_NCCL_EP=0 MAX_JOBS=24 CUDA_HOME=<venv>/lib/python3.12/site-packages/nvidia/cu13 CPLUS_INCLUDE_PATH=<venv>/.../nvidia/cudnn/include:<venv>/.../nvidia/nccl/include uv pip install --no-build-isolation --no-cache-dir transformer_engine_torch==2.18.0`
   - `NVTE_WITH_NCCL_EP=0` 用来绕开缺 `nccl.h`；`CPLUS_INCLUDE_PATH` 里放 cuDNN 头文件绕开缺 `cudnn.h`。
4. **LD_LIBRARY_PATH**：TE/NCCL/cuDNN 的 `.so` 都在 venv 的 `nvidia/*/lib` 下，
   import 前必须加入 `LD_LIBRARY_PATH`（`04_megatron_grpo.sh` 已自动处理）。
5. **Apex 没装**：mcore 会回退到 Torch Norm（日志有 warning），配合
   `--gradient_accumulation_fusion false` 可以正常训练。
6. **flash-attn 没编译**：vLLM 自带 FA 用于推理；Megatron 侧用 `--attention_backend auto`
   （mcore 在缺 flash-attn 时会回退），需要更快的训练可以再编 flash-attn 2.8.3。
7. **数据集**：`AI-ModelScope/clevr_cogen_a_train` 虽然只需要 2000 条，但会下载全量
   （27 个 ~490MB 分片，约 13GB），首次预处理 7 万条约 20s；之后走缓存。
8. **GPU 分配**：0/1 号卡常有其他任务；本工程默认 rollout 用 2,3、训练用 4,5,6,7，
   可用 `VLLM_GPUS` / `TRAIN_GPUS` 覆盖。

## 7. 放大到正式训练

```bash
# 更多步数 & 更大 batch（数值按显存/卡数调整）
MAX_STEPS=2000 bash scripts/02_grpo_multimodal.sh
# 换数据集/奖励函数：--dataset 'AI-ModelScope/GEOQA_R1V_Train_8K' --reward_funcs accuracy format
# LoRA：脚本里 --tuner_type full 换成 lora，并加 --vllm_enable_lora true --lora_rank 8
# Megatron 并行：--tensor_model_parallel_size 2 --sequence_parallel true
```

常用命令备忘：

```bash
# 用训练好的 checkpoint 推理
swift infer --adapters output/grpo_clevr_3b/v1-*/checkpoint-20 --stream true --max_new_tokens 512
# 查看某个 run 的完整参数
cat output/grpo_clevr_3b/v1-*/args.json
```
