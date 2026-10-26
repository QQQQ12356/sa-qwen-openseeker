# OpenSeeker 历史证据语料库

本方案将 OpenSeeker 轨迹中的 visit 网页证据整理为本地 JSONL，供 corpus 后端重新搜索和访问。无需下载网页、不加载模型、不使用 GPU；使用项目现有 search-comp-qwen3.5 环境。

这是**根据原问题/访问目标选择过的历史证据集合**，用于训练集拟合诊断和受控本地检索评测，不是独立网页库，也不代表开放网络搜索能力。语料只包含被访问且可识别的证据，不保证所有问题均有答案。

## 1. 一键构建

从项目根目录运行：

```bash
bash scripts/build_openseeker_corpus.sh
```

先编辑该 Shell 脚本顶部：

```bash
PYTHON=""  # 可填写当前服务器 Python 绝对路径
CONDA_ENV="search-comp-qwen3.5"

INPUTS=(
  "outputs/data/openseeker/polar_correct_qwen35_16k/train.jsonl"
  "outputs/data/openseeker/polar_correct_qwen35_16k/validation.jsonl"
  "outputs/data/openseeker/polar_correct_qwen35_16k/test.jsonl"
)
OUTPUT_DIR="outputs/data/openseeker_corpus/polar_correct_qwen35_16k_evidence"
PROGRESS_EVERY=100
```

默认合并同一 16k 数据版本的三个 split，便于模型在训练、验证和测试问题上共用固定证据库。可以只保留其中一个或多个文件。输入的 split 是语料来源，不会改变测试脚本的 SPLIT。合并 test 来源后必须在实验记录中说明库包含测试轨迹关联证据；不能将结果标为独立语料检索结果。

其他服务器的 4k 数据示例：

```bash
INPUTS=(
  "outputs/data/openseeker/my_correct_4k/train.jsonl"
  "outputs/data/openseeker/my_correct_4k/validation.jsonl"
  "outputs/data/openseeker/my_correct_4k/test.jsonl"
)
OUTPUT_DIR="outputs/data/openseeker_corpus/my_correct_4k_evidence"
```

如服务器不存在某个 split，请从 INPUTS 删除对应条目。路径相对于项目根目录；支持绝对路径以及从其他工作目录调用 Shell 脚本。

输出目录必须不存在，避免覆盖已有语料版本。脚本中变量直接赋值，外部 export 不会覆盖这些赋值。进度按输入轨迹条数打印。

## 2. Python 入口及统一入口

可以直接重复传入 --input：

```bash
python scripts/build_openseeker_corpus.py \
  --input outputs/data/openseeker/polar_correct_qwen35_16k/train.jsonl \
  --input outputs/data/openseeker/polar_correct_qwen35_16k/validation.jsonl \
  --input outputs/data/openseeker/polar_correct_qwen35_16k/test.jsonl \
  --output-dir outputs/data/openseeker_corpus/polar_correct_qwen35_16k_evidence \
  --progress-every 100
```

支持已规范化的 OpenSeeker JSONL，以及 normalize 可处理的原始 trajectory/messages 和结构化 tool_calls/tool 消息。它不是 Search-R1 的 <information> 抽取器。

也可使用统一入口：

```bash
bash scripts/openseeker.sh corpus \
  --input outputs/data/openseeker/polar_correct_qwen35_16k/test.jsonl \
  --output-dir outputs/data/openseeker_corpus/polar_16k_test_evidence
```

--progress-every 必须为正整数，默认 100。输入不存在或重复列出同一个文件会在创建输出目录前报错。

## 3. 提取和去重规则

1. 使用项目共享 normalize / parse_calls 读取轨迹及工具调用。
2. 先收集同一 assistant 批次的全部调用及后续回复。仅当调用数与回复包装块数一致时按原顺序配对，防止把 search 结果错归到 visit。
3. 仅处理 visit，且 url 必须是单个 HTTP(S) URL，或只含一个 URL 的数组。包含多个 URL 的一次 visit 因不能可靠区分正文而跳过。
4. 识别回复中独立一行的 Evidence in page:，仅保留它之后、Summary: 之前的证据。原文为列表字符串或 Markdown 时保留其内容。
5. 去除包含 goal 的前言及 Summary；不提取 system、用户问题、assistant 推理/最终答案或参考 answer 字段。
6. 以精确 URL 合并文档，不改写 URL。同一 URL 内，以去掉首尾空白后的证据 SHA256 去重，不做语义去重；不同证据按输入顺序连接。
7. 文档 title 暂以 URL 填充，不猜测网页标题；当前 BM25 按 text 建索引。

失败访问、未知回复格式、空证据、非法 URL、多个 URL 的 visit、调用/回复数量不匹配、无法解析的输入行都会记录到 skipped.jsonl。无法可靠提取时不会将任意工具回复当作网页正文。

输出不会包含 goal 前言或最终 Summary，但保留的证据本身仍是历史工具按目标选择过的内容。

## 4. 输出文件与核对

输出目录包含：

| 文件 | 用途 |
|---|---|
| corpus.jsonl | CorpusTools 可直接加载的 id/url/title/text 文档 |
| manifest.json | 成功标记、来源文件路径/SHA256/行数、提取规则、计数、语料 SHA256 和局限 |
| provenance.jsonl | 每条接受的观察对应输入行、样本 ID、消息/调用位置、URL、证据 hash、重复标记 |
| skipped.jsonl | 跳过原因及原始输入位置，便于回查 |

provenance / skipped 的 line 从 1 开始，message_index / call_index 从 0 开始。溯源文件记录 ID 和位置，不复制完整问题或答案正文。

corpus.jsonl 示例：

```json
{"id":"<URL的SHA256>","url":"https://example.org/page","title":"https://example.org/page","text":"证据段一\n\n证据段二","source":"openseeker_recorded_visit_evidence","evidence_count":2}
```

检查统计：

```bash
cat outputs/data/openseeker_corpus/polar_correct_qwen35_16k_evidence/manifest.json
wc -l outputs/data/openseeker_corpus/polar_correct_qwen35_16k_evidence/corpus.jsonl
```

主要计数：

- input_rows / valid_rows / invalid_rows：输入轨迹和格式校验情况。
- visit_calls：解析成功轨迹中的 visit 调用数。
- accepted_observations：成功提取的观察数，包含重复。
- unique_evidence / duplicate_observations：URL 内的独立证据数和重复次数。
- documents：去重后的 URL 文档数。
- rows_with_evidence：至少提取一条可用证据的轨迹数，**不是答案覆盖率**。
- skipped_*：各类跳过原因及数量。

某个输入文件损坏时可查看 invalid_row；大量 missing_evidence_marker 表示该数据的工具回复格式需要专门适配。没有提取到任何文档时退出码为 2，保留跳过报告，不写成功 manifest。意外中断也可能留下部分产物，不能仅凭 corpus.jsonl 存在判断成功，应检查 manifest.json。重试时使用新输出目录。

## 5. 配置一键评测

编辑 scripts/test_openseeker.sh：

```bash
RUN_DIR="outputs/experiments/<实际训练实验名>"
MODEL_PATH=""
SPLIT="test"  # 训练拟合诊断改成 train
DATA=""       # 默认从 RUN_DIR 选择 SPLIT 对应文件

BACKEND="corpus"
CORPUS="outputs/data/openseeker_corpus/polar_correct_qwen35_16k_evidence/corpus.jsonl"
ENDPOINT=""
LIMIT=2
MAX_CONTEXT=16384
DRY_RUN=1
```

4k 数据对应改成：

```bash
CORPUS="outputs/data/openseeker_corpus/my_correct_4k_evidence/corpus.jsonl"
MAX_CONTEXT=4096
```

先检查命令，再改 DRY_RUN=0：

```bash
bash scripts/test_openseeker.sh
```

跑通后设置合适的 LIMIT。建库没有模型/tokenizer 依赖的长度筛选；同 URL 多段证据合并后可能超出 MAX_TOOL_TOKENS 或 MAX_CONTEXT，应结合 tool_budget/context_limit 排查。评测不会静默截断 visit 回复。

当前 corpus search 每查询最多返回 10 篇及每篇前 600 字符，visit 返回该 URL 合并后的全部证据；不会按当前 goal 再做摘要。系统提示词仍来自 DATA，默认描述网页搜索和目标相关摘要，因此工具内容分布仍与训练记录不同。更换后端不修改模型或提示词。

## 6. 如何报告结果

保留 manifest 与原始轨迹，明确说明语料来自哪些 split、URL 数、证据数和筛选规则。plain 与 Beacon 对比使用同一份语料、问题和预算，分别报告 replay 与 corpus 的 EM/F1、状态分布。

此库比随意使用 HotpotQA 更贴近这些 OpenSeeker 轨迹涉及的知识，但包含被历史轨迹选择过的证据，且缺少 search-only 页面及访问失败页面。不要将其视为完整网页快照或开放网络基准的复现。

## 7. 回归测试

```bash
python -m pytest tests/test_build_openseeker_corpus.py tests/test_openseeker.py -q
```

测试覆盖正文边界、失败和批量调用、URL 合并/去重、溯源、禁止覆盖、结构化消息、Shell 入口和 CorpusTools 的 search/visit。CPU 即可运行，不加载真实模型。

## 8. 本地实际构建结果

已用默认 polar_correct_qwen35_16k 的 train / validation / test 实际构建，并完成 CorpusTools 的 search 和 visit 冒烟检查，无需模型或网络。

| 项目 | 数量 |
|---|---:|
| 输入轨迹 | 664（644 / 9 / 11） |
| visit 调用 | 893 |
| 接受的观察 | 788 |
| 去重证据段 | 787 |
| URL 文档 | 745 |
| 至少贡献一段证据的轨迹 | 409 |
| 多 URL visit 跳过 | 11 |
| 访问失败跳过 | 6 |
| 空证据跳过 | 10 |
| 调用/回复数不匹配跳过 | 71 |
| 非法 URL 跳过 | 7 |

语料文件为 1,299,050 字节，SHA256：

```text
04afb7a94713e343f257bbba477e0b4ac837c3fa4f619e4536899532ade3ed47
```

结果位于 `outputs/data/openseeker_corpus/polar_correct_qwen35_16k_evidence/`。outputs 下的生成文件由项目忽略规则管理；在另一台服务器同步脚本后可重新构建。以上数值仅对应当前本地输入，4k 等其他数据版本应以自己的 manifest 为准。409 只表示贡献证据的轨迹数，不能解释为答案覆盖率；只有 search 的轨迹也可能在共享库中找到相关资料。

相关 CPU 回归检查通过 58 项，覆盖建库、旧语料构建、OpenSeeker 协议和评测。没有运行真实模型质量评测。
