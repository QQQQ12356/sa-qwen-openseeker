# OpenSeeker 实验工作流

## 环境与入口

沿用原项目依赖和训练架构。当前可用环境是 `search-comp-qwen3.5`，CPU 测试使用其中的 PyTorch/Transformers；没有加载 4B 模型执行训练。新处理脚本还需要 `pyarrow`（Parquet，datasets 的依赖）；绘图脚本需要 `matplotlib`。

```bash
export PYTHON=/home/huangzj/miniconda3/envs/search-comp-qwen3.5/bin/python
export TOKENIZERS_PARALLELISM=false
```

所有命令从项目根目录执行。`scripts/openseeker.sh` 统一调用 Python；不设置 `PYTHON` 时继承原项目的 conda 环境查找规则。

## 1. 下载并保留原始数据

```bash
bash scripts/openseeker.sh download --source all
# 网络慢时手动切换镜像，不修改数据 ID：
bash scripts/openseeker.sh download --source all --endpoint https://hf-mirror.com
```

脚本先解析仓库 commit，再下载该 commit，目录为 `outputs/data/raw/{polar,zephyr}/<commit>/`。保留 Parquet 和 README，并逐批导出 `train.jsonl` / `val.jsonl`；下载 manifest 记录源文件 SHA256、字节数、commit、endpoint。可通过 `--revision <commit>` 固定版本。JSONL 转换先写临时文件，完成后改名。

本次已从本地 Hugging Face 缓存恢复 Polar 官方快照 `5970e581dc1548f37d846a1955cb8b651d2b657e`，并逐行比较原项目 JSONL 与官方快照的解析结果。源文件无改写。Zephyr 仍需在可联网环境执行上述命令。

## 2. 全量分析、规范化、长度与质量筛选

可编辑 `scripts/prepare_openseeker.sh` 顶部的输入路径、输出目录、tokenizer、长度和划分变量，然后执行：

```bash
bash scripts/prepare_openseeker.sh
```

其中 `VAL_FRACTION=0.02`、`TEST_FRACTION=0.02` 默认对应训练 / 验证 / 测试 = 96% / 2% / 2%；训练集自动取剩余比例。若需 80% / 10% / 10%，将两个变量都改为 `0.10`。两个值必须非负且和小于 1。划分在质量、长度筛选和问题去重后进行，哈希阈值控制的是近似比例，不保证精确条数；已有 val/validation/test 输入题目优先保留原归属，因此这类输入的最终比例还会受原划分影响。`SEED` 控制可复现划分，最终条数见输出 `manifest.json` 的 `counts`。调整变量后需生成新目录，已有数据不会自动更新。

也可继续使用命令行参数 `--val-fraction`、`--test-fraction`：

```bash
# MODEL ID 是精确的 tokenizer 来源，不用字符数估算 token。
bash scripts/openseeker.sh prepare \
  --source polar \
  --input outputs/data/raw/polar/5970e581dc1548f37d846a1955cb8b651d2b657e/openseeker_v1_data.jsonl \
  --output outputs/data/openseeker/polar_correct_dual_16k \
  --models Qwen/Qwen3-4B-Thinking-2507 Qwen/Qwen3.5-4B \
  --max-length 16384 --workers 8 \
  --val-fraction 0.02 --test-fraction 0.02

# Zephyr：两个 split 必须一起传入，保留已有 validation 题目归属。
bash scripts/openseeker.sh prepare --source zephyr \
  --input outputs/data/raw/zephyr/<commit>/train.jsonl \
  --input outputs/data/raw/zephyr/<commit>/val.jsonl \
  --output outputs/data/openseeker/zephyr_correct_dual_16k \
  --max-length 16384 --workers 8
```

`--offline` 只使用本地 tokenizer 缓存；本次只有 Qwen3.5-4B 的精确 tokenizer 缓存，因此实测命令指定 `--models Qwen/Qwen3.5-4B --offline`。不要把其他 Qwen tokenizer 的测量冒充指定模型的结果。

脚本依次进行格式校验、规范化、两个 tokenizer 的实际编码、质量与长度筛选、按问题去重和划分。任一 tokenizer 超长就淘汰；不截断、不开滑窗、不丢弃最终答案。默认 `--require-correct` 剔除官方 `Incorrect`；Zephyr 没有此标签时保留，并记为 unspecified，不能据此声称已验证正确。`--no-require-correct` 用于显式研究错误轨迹。

生成文件：

- `normalized.jsonl`：全部可解析轨迹，含来源、问题 hash、工具 schema、质量和每模型长度；不按质量/长度删除。
- `train.jsonl`、`validation.jsonl`、`test.jsonl`：新建的筛选子集。
- `rejected.jsonl`：源行 ID、淘汰原因；原始内容仍在 raw / normalized 中。
- `measurements.jsonl`：每条轨迹的 token、工具次数与淘汰原因。
- `manifest.json`：完整配置、计数、分位数、源 SHA256、tokenizer 指纹；只在成功完成后写入。

输出目录必须不存在，避免覆盖已有实验。分割由规范化问题的 SHA256 + seed 决定，默认 96% / 2% / 2%；同一问题不会跨 split。已发布的 val/test 问题优先保留其归属；同题训练样本也归入该 holdout。Zephyr train 中会再留出一小部分 validation/test。两来源可能重叠，联合实验应一次传入两者全部文件，并用 `--source combined` 去重，而不能直接拼接各自已经划分的文件。

已有实测长度可快速改阈值，无需重复分词：

```bash
"$PYTHON" scripts/refilter_openseeker.py \
  --prepared outputs/data/openseeker/polar_qwen35_16k \
  --output outputs/data/openseeker/my_correct_32k --max-length 32768
```

`polar_qwen35_16k` 是本次保留全部质量标签的分析父目录；默认训练指向后续生成的 `polar_correct_qwen35_16k`。refilter 记录父目录，完整数据和分布报告留在父目录中。

复现报告：

```bash
"$PYTHON" scripts/audit_openseeker.py \
  --input outputs/data/raw/polar/openseeker_v1_data.jsonl \
  --output docs/openseeker/polar_structure.json
MPLCONFIGDIR=/tmp/openseeker-mpl "$PYTHON" scripts/report_openseeker.py \
  --prepared outputs/data/openseeker/polar_qwen35_16k --output docs/openseeker
```

## 3. 实验参数与训练

只需编辑 `configs/openseeker.yaml`：

- `model_family`: qwen3 / qwen35；模型 ID 已按要求设定。
- `kind`: beacon / plain，复用原训练器的压缩组和未压缩对照组。
- `dataset`: 处理后的目录；切换 32k 时同步把 `training.max_length` 和 `evaluation.max_context` 改为 32768。
- `training`: 学习率、epoch、步数、LoRA、梯度累积、随机种子。
- `beacon`: 压缩比、续写损失、输出蒸馏权重、激活卸载。
- `evaluation`: replay / http、工具服务地址、上下文/生成/轮数预算。

```bash
bash scripts/openseeker.sh plan
# 复制其输出的 run_dir；plan 不加载模型。
bash scripts/openseeker.sh train --run-dir "$RUN_DIR"
```

实验名包含模型族、plain/beacon、数据模式、长度、压缩比、学习率、seed、配置/数据/代码 hash 和 UTC 时间。一个目录包含 `experiment.json`、`train.yaml`、原始配置、训练/评测日志、命令、退出码、耗时、数据和代码 SHA256。修改数据或代码后使用旧计划会报错；重新 plan 生成新实验。训练后的 checkpoint 在该目录的 `models/model/`，模型加载路径取 `train_summary.json`，无需手工猜测 LoRA merge 的目录。

同一训练 run 不允许重复写入。当前统一入口不提供 Beacon 断点续训快捷方式；原 plain 训练器的 `--resume_from_checkpoint` 仍可单独使用。`max_train_steps: null` 表示按 epoch，设置正数时按优化器步数。

80GB 建议先用 16k、batch=1、LoRA r=16、BF16、梯度累积 4，输出蒸馏先置 0。压缩历史 KV 不代表前向激活、保留的 reasoning、完整 teacher 都同步缩小；32k 或输出蒸馏应在目标 GPU 上实测再扩大。原 Beacon 不支持 Trainer 的普通层级 gradient checkpointing，依靠自身 loss 分块与激活 offload；没有宣称本配置必定满足 80GB 峰值。

### 一键训练（在脚本中修改参数）

编辑 `scripts/train_openseeker.sh` 顶部的 Python、GPU、ACTION，以及其中的 YAML 参数块，然后执行：

```bash
bash scripts/train_openseeker.sh
```

默认直接创建新实验并启动训练，不需要先运行 plan 或手动填写 run_dir。数据集必须事先准备好；相对路径以项目根目录为准。脚本支持从其他工作目录调用。

脚本中的参数块是独立的实验入口配置，不读取 `configs/openseeker.yaml`；两者不需要同步修改。它仍复用 `base_configs` 指定的基础训练配置，最终生效的参数保存在实验目录的 `train.yaml`，脚本配置保存在 `source_config.yaml`。每次执行都会创建新实验，不用于断点续训。

将脚本顶部 `ACTION="train"` 改为 `ACTION="plan"`，可只生成配置而不加载模型。OpenSeeker 支持 `beacon.loss_segments` 分段权重；默认入口提供 final_answer 加权起始值，设为 `{}` 可恢复全部启用且权重为 1。详见 [分段损失配置](openseeker_loss_segments.md)。日志和模型输出沿用上述实验目录结构。

## 4. 测试与推理

完整配置、训练集拟合诊断、多卡并行、语料构建及三种后端的操作步骤见 [一键测试操作说明](openseeker_testing.md)。

```bash
bash scripts/openseeker.sh test
OMP_NUM_THREADS=2 "$PYTHON" -m pytest tests -q
bash scripts/openseeker.sh eval --run-dir "$RUN_DIR"
```

### 一键测试（在脚本中修改参数）

编辑 `scripts/test_openseeker.sh` 顶部的变量，然后执行：

```bash
bash scripts/test_openseeker.sh
```

填写 `RUN_DIR="outputs/experiments/<训练实验名>"` 后，脚本自动从该实验的 `train_summary.json` 读取最终模型，从 `experiment.json` 读取模型类型和测试集。也可以单独填写 `MODEL_PATH` 测试已有的合并 checkpoint；`KIND`、`DATA` 支持覆盖自动读取的值，无实验目录时默认使用 Beacon 和 `polar_correct_qwen35_16k/test.jsonl`。

`SPLIT="train"` 可选训练集检查拟合，`validation` / `test` 分别选择验证 / 测试集；DATA 非空时覆盖自动路径。多卡设置 `CUDA_VISIBLE_DEVICES="0,1"` 和 `NUM_WORKERS=2`，每卡一个完整模型副本并行处理不同题目。Python 环境、GPU、`BACKEND`、`ENDPOINT`、`CORPUS`、样本数及各项生成预算均在脚本顶部配置。HTTP 后端必须填写 `ENDPOINT`；本地语料后端必须填写 `CORPUS`。评测参数直接取自测试脚本，无需修改训练配置中的 `evaluation` 块。

将 `DRY_RUN=1` 可先检查配置并打印实际命令，不加载模型或创建输出。每次正式运行使用新的时间戳保存预测 JSONL、指标 JSON 和日志，默认写入 `RUN_DIR/evaluation/`；单独指定模型时写入 `outputs/results/openseeker/`，可用 `OUTPUT_DIR` 修改。相对路径以项目根目录为准，支持从其他工作目录调用。终端会实时显示模型/后端初始化、逐题进度、累计 EM/F1 和预计剩余时间；长时间操作默认每 15 秒打印阶段心跳，可用 PROGRESS_INTERVAL 调整。多卡进度也会转发到主终端。此脚本执行模型评测；代码单元测试仍使用上方的 pytest 入口。

默认 replay 后端只在模型生成的**整批工具名与完整参数**和记录一致时返回下一批历史观察，失配立即记 `tool_error/replay_miss`。不会拿原轨迹的下一条回复冒充任意新查询的结果；它只能测动作复现与协议执行，不能代表真实联网能力。

在线评测使用 `backend: http` 和 `endpoint`。工具服务接口：

```text
POST /tool
Content-Type: application/json
{"name":"search","arguments":{"query":["query A","query B"]}}

200 OK
{"content":"真实搜索结果正文，保留 URL 和摘要"}
```

`visit` 请求为 `{"name":"visit","arguments":{"url":["https://..."],"goal":"..."}}`，服务应执行网页访问与目标相关内容提取。HTTP 后端按生成的调用顺序提交并包装回复；工具服务需自行配置搜索提供商、网页提取和授权。仓库不附带第三方 API 密钥，也没有执行外网工具质量测试。自定义工具可以使用相同 HTTP 接口，或通过 `Registry({name: callable})` 注入；不限定只有检索工具。

单独评测已有 checkpoint：

```bash
"$PYTHON" scripts/evaluate_openseeker.py \
  --model /path/to/merged_checkpoint --kind beacon \
  --data outputs/data/openseeker/polar_correct_qwen35_16k/test.jsonl \
  --backend http --endpoint http://127.0.0.1:8080/tool \
  --output outputs/results/my_run/predictions.jsonl
```

记录每题预测、完整交互、工具参数、状态、耗时和 token 数，另存汇总。EM 是忽略大小写与多余空格的严格文本匹配，支持中文但不识别语义等价答案；不是论文使用的模型裁判指标。另输出答案 token F1（英文 SQuAD 风格归一化，按空白分词），总 F1 按所有题目平均，失败/空答案记 0；中文不额外分词。超预算或调用错误不会被当作成功答案。保留完整工具结果且超过预算即停止，不在推理端引入训练端没有的截断策略。GPU 在线结果须实际运行后才能报告。

## 5. 原有检索任务

原 `scripts/20_plain_train.sh`、`22_beacon_train.sh`、`24_beacon_train_searchr1.sh` 和 `21_plain_eval.sh`、`23_beacon_eval.sh` 保留；原配置的 `data_mode: searchr1` 仍走原数据类、裸 `<information>` 和 BM25 路径。OpenSeeker 配置走 `data_mode: openseeker`，不会自动把旧任务改为多工具协议。历史文档中的本地数据路径需要按使用者环境配置。

## 可执行的离线工具环境（无需 replay 或外部 API）

针对 OpenSeeker 历史轨迹，可先运行 `bash scripts/build_openseeker_corpus.sh`，按脚本顶部 INPUTS / OUTPUT_DIR 生成 visit 证据语料。默认路径为 `outputs/data/openseeker_corpus/polar_correct_qwen35_16k_evidence/corpus.jsonl`，然后在测试脚本设置 BACKEND="corpus" 和对应 CORPUS。此库来自历史问题关联证据，属于受控诊断；详见 [建库操作说明](openseeker_corpus.md)。

`evaluation.backend: corpus` 配合 `evaluation.corpus: /path/to/corpus.jsonl` 可以运行真正的 search→visit 循环：search 支持模型生成的新批量查询，visit 根据 URL 读取独立语料的全文。JSONL 每行为 `{"id":"doc1","url":"https://example.org/page","title":"Title","text":"Full document"}`；兼容原项目只有 id/title/text 的语料，会为其分配 corpus:// URL。

```bash
"$PYTHON" scripts/evaluate_openseeker.py --model /path/to/checkpoint --kind beacon \
  --data outputs/data/openseeker/polar_correct_qwen35_16k/test.jsonl \
  --backend corpus --corpus /path/to/independent_corpus.jsonl \
  --output outputs/results/local_corpus/predictions.jsonl
```

这一路径明确标为 `local corpus tool rollout`，使用本地 BM25 和原始全文，不模拟在线 Google 排名或论文的 goal-conditioned 网页摘要器。用于协议测试和受控检索实验；不要从 held-out 轨迹中提取答案文档后再宣称是无泄漏的开放网络测试。

绘图是可选步骤：安装 matplotlib 后执行 `python scripts/plot_openseeker.py --measurements outputs/data/openseeker/polar_qwen35_16k/measurements.jsonl --output docs/openseeker/length_distribution.svg`。本次 SVG 使用独立的已安装 matplotlib 的环境生成；核心处理和训练不依赖绘图。

淘汰记录按长度→正确性→重复问题保留首个原因，因此 manifest 中 `incorrect` 表示通过长度条件后才因错误标签淘汰的条数，不等于原始数据全部 Incorrect 数量。全量质量计数应读取 audit 报告或父目录 manifest 的 `correctness`。
