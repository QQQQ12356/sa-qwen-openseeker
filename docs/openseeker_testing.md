# OpenSeeker 一键测试操作说明

本文对应 `scripts/test_openseeker.sh`，执行已训练模型的交互式评测。命令示例从项目根目录执行：

```bash
cd /home/huangzj/proj/searchagent/sa-qwen-openseeker
```

## 1. 先确定测试方式

| BACKEND | 工具结果来源 | 是否需要语料文件 | 用途 |
|---|---|---|---|
| replay | 测试轨迹中记录的历史工具回复 | 否 | 检查动作复现和协议；模型生成的新查询无法执行 |
| corpus | 本地 JSONL 文档，BM25 search + 全文 visit | 是 | 离线运行真实查询与访问循环，衡量指定语料下的表现 |
| http | 自建 HTTP 网关执行搜索与网页访问 | 否，由服务端管理 | 联网工具评测，需要真实工具服务 |

建议先用 replay 跑 1～2 题确认模型与协议，再按目标选择 corpus 或 http。三种后端的任务环境不同，分数应分别报告。

## 2. 测试数据用什么

测试集是“问题、答案、工具定义/轨迹”的集合；语料库是供检索的文档集合，两者分别由 DATA 和 CORPUS 指定。

已有数据：

| DATA | train / validation / test | 建议 MAX_CONTEXT |
|---|---|---|
| outputs/data/openseeker/polar_correct_qwen35_16k/test.jsonl | 644 / 9 / 11 | 16384 |
| outputs/data/openseeker/polar_correct_qwen35_32k/test.jsonl | 1696 / 23 / 40 | 32768 |

这些是本仓库由 Polar 筛选、按问题划分的测试子集，并非完整的公开基准榜单测试集。其长度使用 Qwen3.5-4B tokenizer 测量。默认训练使用 16k 数据，因此应优先使用同一实验记录的 16k test；32k 子集不能直接视为该 16k 实验的无重叠测试集，因为重新筛选、去重可能改变样本集合，使用前应核对 question_id 与训练集无重叠。

- SPLIT="test"（默认）、"train"、"validation"：分别选择测试、训练、验证数据。
- RUN_DIR 非空且 DATA 留空：从 experiment.json 的 training_config 读取 test_data_path / train_data_path / val_data_path。
- RUN_DIR 为空且 DATA 留空：使用 16k 数据目录下的对应 <SPLIT>.jsonl。
- DATA 非空：覆盖自动选择路径；请将 SPLIT 设为该文件真实划分，保证结果标签准确。
- LIMIT 按文件顺序取前 N 行，不随机抽样。LIMIT=100 在 11 题文件上仅评测 11 题；建议正式测试设置为测试集总题数。
- train.jsonl 用于训练，validation.jsonl 用于调参，最终结果使用 test.jsonl。
- corpus/http 仍读取相同的 OpenSeeker 测试文件。模型推理从 system 和最初 user 问题开始；历史 assistant 答案不作为生成上下文。只有 replay 使用后续历史动作和回复。
- 当前 normalize 要求完整的 system/user/assistant 轨迹，不能直接输入原始 HotpotQA 行或只有 question/answer 的文件。自定义数据要适配 OpenSeeker 格式并保留真实答案；replay 还必须有真实工具轨迹。
- 分数依赖非空 answer 字段；无答案的样本 EM 和 F1 均记 0。

## 3. 修改测试脚本

直接编辑 scripts/test_openseeker.sh 顶部赋值。脚本不是命令行参数解析器；外部 export 同名变量会被这些赋值覆盖。

先采用以下配置（把 RUN_DIR 的占位符换成实际实验名）：

```bash
PYTHON="/home/huangzj/miniconda3/envs/search-comp-qwen3.5/bin/python"
CONDA_ENV="search-comp-qwen3.5"
export CUDA_VISIBLE_DEVICES="0"
NUM_WORKERS=1
DRY_RUN=1

RUN_DIR="outputs/experiments/<实际实验名>"
MODEL_PATH=""
KIND=""
SPLIT="test"
DATA=""

BACKEND="replay"
ENDPOINT=""
CORPUS=""
LIMIT=2
MAX_CONTEXT=16384
MAX_NEW_TOKENS=8192
MAX_TOOL_TOKENS=16384
MAX_TURNS=32
OUTPUT_DIR=""
```

RUN_DIR 是包含 experiment.json 的实验根目录，不是 models/model 子目录。训练成功后，脚本从 RUN_DIR/models/model/train_summary.json 的 final_model_path 获取模型，从 experiment.json 获取 kind 和 SPLIT 对应数据集。只有计划配置、没有训练完成产物的实验不能直接评测。

查看实验目录及模型路径：

```bash
ls outputs/experiments
cat outputs/experiments/<实际实验名>/models/model/train_summary.json
```

若没有 train_summary.json，先检查训练是否成功。也可以单独指定一个已保存的完整模型：

```bash
RUN_DIR=""
MODEL_PATH="/absolute/path/to/merged_checkpoint"
KIND="beacon"  # 普通 SFT 模型填 plain
DATA="outputs/data/openseeker/polar_correct_qwen35_16k/test.jsonl"
```

MODEL_PATH 同时覆盖 RUN_DIR 中的模型；KIND 和 DATA 非空时分别覆盖实验配置。Beacon 应使用保存了 Beacon 参数和配置的 checkpoint；LoRA 请使用已合并的完整模型。恢复 Beacon adapter 合并可使用：

```bash
bash scripts/30_merge_adapter.sh outputs/experiments/<实际实验名>/models/model
```

该命令要求模型目录有 resolved_config.json 和对应 adapter；以命令输出的实际路径为准，再填 MODEL_PATH。正常训练收尾会执行合并。

运行：

```bash
# DRY_RUN=1：检查元数据/测试文件并打印命令，不加载模型。
bash scripts/test_openseeker.sh

# 确认打印的模型、数据和后端正确后，在脚本中改 DRY_RUN=0，再运行同一命令。
bash scripts/test_openseeker.sh
```

DRY_RUN 不验证模型能否加载、显存是否足够或 HTTP 服务是否可用。首次正式运行用 LIMIT=2，跑通后将 LIMIT=11 测完整 16k 测试集。


### 3.1 在训练集上判断拟合能力

对同一训练实验修改：

```bash
RUN_DIR="outputs/experiments/<实际实验名>"
MODEL_PATH=""
SPLIT="train"
DATA=""
BACKEND="replay"
LIMIT=644
DRY_RUN=0
```

然后运行：

```bash
bash scripts/test_openseeker.sh
```

16k 训练集为 644 题，32k 训练集为 1696 题；以当前实验实际 train.jsonl 行数为准。先用 LIMIT=2 冒烟测试，再设置为全量题数。LIMIT 是全局样本上限，多卡时不会每卡重复评测 LIMIT 题。

无 RUN_DIR、只有完整模型目录时，也可设 SPLIT="train"，默认读取 16k/train.jsonl；其他训练集请显式设置 DATA。评估验证集时设 SPLIT="validation"。

训练集 EM/F1 衡量模型在已见问题上的自由生成表现，不等同于 teacher-forced loss 或逐 token 准确率。replay 同时要求精确复现工具动作，因此还需结合 statuses 和 replay_miss 比例判断；工具动作不同可能阻断后续答题。训练集表现只用于拟合诊断，正式泛化结果使用 test，并保持后端与预算一致。

输出文件名包含 train / validation / test，metrics 中的 data_split 和 arguments.data 保存划分与实际文件路径。

### 3.2 多卡并行测试

编辑脚本：

```bash
export CUDA_VISIBLE_DEVICES="0,1,2,3"
NUM_WORKERS=4
SPLIT="train"  # 也可选 test 或 validation
LIMIT=644
DRY_RUN=0
```

运行命令相同：

```bash
bash scripts/test_openseeker.sh
```

每张卡启动一个独立 Python 进程、加载一个完整模型副本，按原始行号取模分配前 LIMIT 条样本。所有卡合计评测这批问题一次，最终按原始顺序合并预测，并按实际样本数计算 EM/F1。工作进程失败时主进程停止其他进程并报错，不生成看似完整的总指标；分片文件和 worker 日志保留供排查。

- NUM_WORKERS 必须为正整数，多卡时 CUDA_VISIBLE_DEVICES 必须显式列出至少同等数量的互不重复 GPU。使用列表前 NUM_WORKERS 张卡，例如 "2,5" 配合 NUM_WORKERS=2。
- 这是按题目并行，每张卡都需要容纳完整模型及单题推理开销；不将一个模型拆到多卡。仅修改 CUDA_VISIBLE_DEVICES 不会自动增加工作进程。
- 单卡设 NUM_WORKERS=1；多卡不会改变每题生成预算。少量题目不一定受益于多卡，建议工作进程数不超过题数。
- corpus 模式每个工作进程各加载一份语料和索引，CPU 内存需求增加；http 模式工具请求并发增加，服务端需能承受。
- 各 worker 的输出和日志位于 predictions_<SPLIT>_<时间戳>.workers/；总日志会打印该路径。
- 多卡 CPU 自动化测试覆盖了分片、GPU 分配、汇总和失败处理；真实 GPU 吞吐与数值一致性需在目标机器实测。

也可直接调用 Python：

```bash
CUDA_VISIBLE_DEVICES=0,1 /home/huangzj/miniconda3/envs/search-comp-qwen3.5/bin/python \
  scripts/evaluate_openseeker.py \
  --model /absolute/path/to/merged_checkpoint --kind beacon \
  --data outputs/data/openseeker/polar_correct_qwen35_16k/train.jsonl \
  --data-split train --num-workers 2 --limit 644 --backend replay \
  --output outputs/results/train_two_gpu/predictions.jsonl
```

直接调用时 --data 决定实际数据，--data-split 只做结果标记，必须与数据来源一致；一键脚本会根据 SPLIT 自动选文件。

## 4. replay 后端

脚本修改：

```bash
BACKEND="replay"
ENDPOINT=""
CORPUS=""
```

执行：

```bash
bash scripts/test_openseeker.sh
```

无需构建语料或启动服务。每批工具调用的工具名、完整参数、批内顺序都必须与历史记录匹配，才返回该批历史回复。查询措辞不同也会触发 tool_error，events 中包含 replay_miss；这表示动作未复现，不能直接解释为在线搜索失败。

## 5. corpus 后端：构建本地语料并测试

### 5.1 准备文档 JSONL

每行一个 JSON 对象：

```json
{"id":"doc1","url":"https://example.org/a","title":"文档标题","text":"这篇文档的完整正文"}
{"id":"doc2","url":"https://example.org/b","title":"另一篇文档","text":"另一篇文档的完整正文"}
```

id 使用唯一字符串；url 如提供也必须唯一。没有 url 时自动生成 corpus:// URL。正文应为可读纯文本，使用 JSON 序列化转义换行；不能把 HTML 文件路径当成正文。为测试 Polar 问题，应收集覆盖相应主题的独立网页/文档正文并写入上述格式。仓库没有附带覆盖 Polar 全部问题的独立网页库，也没有从任意 URL 自动抓取建库的一键脚本。

不要将测试轨迹里的工具回复直接抽出来充当独立检索语料；那属于使用题目已知证据的受控实验，结果需要单独说明。

### 5.2 可直接执行的建库示例：HotpotQA 文档

仓库已有 HotpotQA 建库模块，命令如下（下载数据需要网络或已有缓存）：

```bash
/home/huangzj/miniconda3/envs/search-comp-qwen3.5/bin/python \
  -m search_comp.data.build_corpus \
  --output_path outputs/data/hotpotqa_corpus.jsonl \
  --splits train,validation \
  --max_per_split 7405 \
  --seed 42
```

此命令每个 split 最多随机采样 7405 道问题，提取 context 文档并去重；7405 不是文档总数。去掉 --max_per_split 可构建所有指定 split 的文档，内存需求随文档规模增加。

生成文件只有 id/title/text 也可直接使用，工具会分配 corpus:// URL。**HotpotQA 文档不保证覆盖 Polar 测试问题**：它适合跑通本地工具流程；Polar 正式语料评测需另建覆盖所需知识的独立库。原始 HotpotQA 问题格式也不能直接填 DATA。

已有 Search-R1 训练轨迹还可按下列命令抽取训练观察文档：

```bash
/home/huangzj/miniconda3/envs/search-comp-qwen3.5/bin/python \
  -m search_comp.data.build_corpus_searchr1 \
  --jsonl_path outputs/data/searchr1/qwen3-4b-instruct-sft.jsonl \
  --output_path outputs/data/searchr1_corpus.jsonl
```

仅在该输入文件实际存在时使用。这是 Search-R1 的 <information> 格式抽取器，不能拿来解析 OpenSeeker 的 <tool_response> 轨迹。用训练文档作知识库时需报告语料来源，且不保证覆盖 Polar 问题。

### 5.3 修改测试脚本

```bash
BACKEND="corpus"
ENDPOINT=""
CORPUS="outputs/data/hotpotqa_corpus.jsonl" # 或自建 independent_corpus.jsonl
DATA="outputs/data/openseeker/polar_correct_qwen35_16k/test.jsonl"
LIMIT=2
```

先 DRY_RUN=1 检查，再 DRY_RUN=0 执行：

```bash
bash scripts/test_openseeker.sh
```

无需单独建索引命令或启动服务。程序启动时加载整个语料，在内存中建立 BM25 索引，每次启动重新构建。search 当前固定每个查询最多返回 10 篇文档及每篇前 600 字符；visit 根据 URL 返回全文，goal 只展示在结果中，不做目标相关摘要。只支持 search 和 visit。索引在 CPU/内存，模型在其加载器分配的设备上。

## 6. http 后端：调用真实工具服务

脚本修改：

```bash
BACKEND="http"
ENDPOINT="http://127.0.0.1:8080/tool"
CORPUS=""
LIMIT=2
```

ENDPOINT 是工具网关地址，不是模型推理接口。仓库有 HTTP 客户端，没有可直接启动的联网搜索网关，因此需使用已有服务或自行实现：search 对接真实搜索提供商；visit 访问网页并提取正文/相关内容；服务端自行管理搜索 API 密钥及鉴权。

网关收到 POST JSON：

```json
{"name":"search","arguments":{"query":["query A","query B"]}}
```

visit 示例：

```json
{"name":"visit","arguments":{"url":["https://example.org/page"],"goal":"提取与问题有关的证据"}}
```

成功返回 HTTP 200 和 JSON，content 必须是字符串，直接返回正文即可，客户端会包装 <tool_response>：

```json
{"content":"真实工具结果，保留必要的 URL、标题及正文"}
```

先验证网关（此命令仅在服务已启动时执行）：

```bash
curl -sS -X POST http://127.0.0.1:8080/tool \
  -H 'Content-Type: application/json' \
  -d '{"name":"search","arguments":{"query":["OpenSeeker"]}}'
```

确认有真实结果后执行：

```bash
bash scripts/test_openseeker.sh
```

HTTP 客户端当前逐个提交调用，单次超时 90 秒；不提供自定义 Authorization header 配置。需要鉴权时可让本地网关代管上游凭据，或扩展客户端。连接失败、非预期返回及超时会记录为 tool_error/gateway_error。

## 7. 预算、结果与排查

| 变量 | 实际含义 |
|---|---|
| MAX_CONTEXT | 每轮序列化后的完整历史 + 本轮生成的 token 总上限；Beacon 同样按逻辑历史计数 |
| MAX_NEW_TOKENS | 每轮 assistant 生成上限，还会被剩余上下文预算限制；不是整题总生成上限 |
| MAX_TOOL_TOKENS | 单条工具回复（含包装标签）的 token 上限，超出即 tool_budget，不截断 |
| MAX_TURNS | assistant 生成循环次数上限，最终答案也占一次 |
| LIMIT | 从文件开头最多读取多少行；多卡合计的上限 |
| SPLIT | train / validation / test，选择数据划分并标记输出 |
| NUM_WORKERS | 并行进程数，每个进程绑定一张 GPU；默认 1 |
| OUTPUT_DIR | 输出目录；留空则用 RUN_DIR/evaluation，无 RUN_DIR 时用 outputs/results/openseeker |

切换 32k 时将 DATA 和 MAX_CONTEXT 一起修改；它不会自动扩大显存，也不会改变已保存模型的训练配置。显存不足可先降低 MAX_CONTEXT/MAX_NEW_TOKENS 并说明预算变化。

每次运行保存：

- predictions_<SPLIT>_<UTC时间戳>.jsonl：逐题 prediction、answer、exact_match、f1、sample_index、messages、events、status、generated_tokens 和耗时。
- predictions_<SPLIT>_<UTC时间戳>.metrics.json：data_split、count、exact_match、f1、statuses、总耗时以及实际评测参数。
- eval_<SPLIT>_<UTC时间戳>.log：程序标准输出和错误日志。

EM 仅在 status=completed、参考答案非空，且双方忽略大小写/合并空白后完全相同时计正确。F1 复用 search_comp/evaluation/em_f1.py：答案转小写、去英文冠词和 ASCII 标点、合并空白后按空白分词，根据重复 token 的交集计算 precision / recall 的调和平均。逐题 f1 在 [0,1]，总 f1 是所有题目的算术平均；空预测、空参考答案或非 completed 样本记 0，并保留在分母。F1 主要适合英文，中文不做额外分词（连续中文会作为一个 token），不衡量语义等价，也不使用 LLM judge。EM 仍保留原来的严格口径，因此 F1=1 不一定 EM=1，例如 "The red!" 对 "red"。多卡汇总按样本累加后求平均，不直接平均各卡均值。逐题 turns 字段当前取 events 长度，不能直接视为 assistant 生成次数。

| 状态/报错 | 排查 |
|---|---|
| 找不到 train_summary.json | 训练未完成或 RUN_DIR 选错；检查训练日志，或指定真实完整 MODEL_PATH |
| tool_error / replay_miss | 生成调用与记录不匹配；检查 events 中 calls |
| tool_error / gateway_error | 检查服务、网络、响应 JSON 和超时 |
| tool_error / URL is not present | corpus 中没有该 URL；检查模型是否使用 search 返回的地址 |
| context_limit | 历史已占满上下文，无法继续生成 |
| generation_limit | 本轮预算耗尽前未遇到结束标记 |
| tool_budget | 某条工具回复过长 |
| turn_limit | 生成循环达到 MAX_TURNS |
| invalid_call | 工具调用标签或 JSON 不符合协议 |
| empty_answer | 结束但没有提取出最终答案 |

对比 plain 与 Beacon 时应固定测试问题、后端、语料版本和预算，分别选择正确的 KIND、模型路径及输出目录。

## 8. 代码单元测试

它与上面的模型评测分别运行，不需要加载 4B 模型：

```bash
bash scripts/openseeker.sh test
OMP_NUM_THREADS=2 /home/huangzj/miniconda3/envs/search-comp-qwen3.5/bin/python -m pytest tests -q
```
