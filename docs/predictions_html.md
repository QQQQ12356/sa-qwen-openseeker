# 离线轨迹检查报告

使用 Python 3 标准库，将评测 JSONL 转成一个包含全部数据、样式和交互的 HTML 文件，无需安装依赖或连接网络。

推荐直接运行 `scripts/render_predictions_html.sh`。所有参数都在脚本顶部：`INPUTS` 填一个或多个 JSONL 路径，`OUTPUT_DIR` 设置输出目录（默认 `outputs/reports`，留空则保存到输入旁边），`TITLE` 可选。相对路径以项目根目录为准，从其他工作目录运行脚本也可。保存后在编辑器运行该脚本，或执行一次：

```bash
bash scripts/render_predictions_html.sh
```

无需再填写命令行参数。脚本会调用共享 Python 转换器和 HTML 模板；下面保留直接调用 Python 的用法，供需要时使用。

```bash
# 默认写到输入文件旁边，同名 .html
python scripts/render_predictions_html.py \
  outputs/data/tmp/sa-qwen3-v1-16-intersect-test_predictions-100qa.jsonl

# 指定输出和标题
python scripts/render_predictions_html.py results.jsonl \
  -o outputs/reports/results.html --title 'Qwen3 测试轨迹'

# 批量转换，每个输入生成一个独立 HTML
python scripts/render_predictions_html.py outputs/data/tmp/*predictions*.jsonl \
  --output-dir outputs/reports
```

在服务器上生成后，将 HTML 下载到本机，直接用浏览器打开。相同目标路径会覆盖旧报告；不同目录的同名输入不能批量写入同一个输出目录。

报告支持全文搜索、状态和 EM 筛选、F1/耗时/token 排序、上一条/下一条，以及展开/折叠轨迹。顶部显示当前筛选结果的指标均值；缺失指标不参与计算，EM 还会显示有效样本数。指标直接读取原文件，不重新评分；F1 概览按原始 0–1 分值显示为百分比，单条记录保留原值。

字段约定：

| 内容 | 字段 |
| --- | --- |
| 问题 | `question` / `query`，或首条 user 消息的 content |
| 模型答案 | `prediction` / `predicted_answer` / `response` |
| 参考答案 | `answer` / `answers` / `reference_answer` / `ground_truth` |
| 轨迹 | `messages`，支持 `role`、`content`、`reasoning_content` / `reasoning`、`tool_calls` |
| 工具事件 | `events`，独立折叠显示 |
| 指标 | `exact_match`、`f1`、`seconds`、`generated_tokens`、`turns`、`status` |

`<think>`、`<thinking>`、`<tool_call>`、`<tool_calls_begin>`、`<search>`、`<visit>`、`<tool_response>`、`<information>`、`<answer>` 和 `<final_answer>` 会分块标注。轨迹按原始消息顺序展示；system 默认折叠并按完整提示词原文展示，其中的格式示例不会被识别成实际工具调用。只有 assistant 内容会拆分为语义块；普通 user 消息和工具返回内部的标签按原文保留。工具调用按工具名和格式化 JSON 展示，思考和工具返回正文默认折叠，final_answer 独立高亮。正文作为纯文本显示，保留换行和 Markdown/公式源文，避免模型输出中的 HTML 被执行。所有原始字段均保存在“完整原始记录”中，包括额外指标。

共享模板位于 `scripts/templates/predictions.html`，可直接修改配色和布局。也可用 `--template path/to/custom.html` 指定模板；模板必须恰好包含一个 `__REPORT_DATA__` 占位符，脚本在此注入 JSON 数据（结构为 `title`、`source`、`rows`）。

文件以 UTF-8 读取，兼容 BOM 和空行；无效 JSON 或非对象记录会报出文件和行号。转换和浏览均会将整份数据载入内存，适合人工检查的评测结果；大型轨迹文件建议先拆分。生成的 HTML 包含全部原始数据，共享时等同于共享原 JSONL。
