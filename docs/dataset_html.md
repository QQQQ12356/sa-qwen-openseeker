# 数据集离线 HTML 浏览器

用于查看训练数据，支持准备后的 OpenSeeker / Search-R1 `messages` JSONL，以及原始
Polar `trajectory`（列表或 JSON 字符串）。仅需 Python 3 标准库，无需模型、GPU、网络
或第三方前端资源。生成的是自包含 HTML，下载到本机后双击打开。

## 一键生成

在项目根目录运行，默认读取现有 `polar_correct_qwen35_16k/train.jsonl`：

```bash
bash scripts/render_dataset_html.sh
```

查看另一台服务器的 6k 数据集：

```bash
bash scripts/render_dataset_html.sh \
  outputs/data/openseeker/polar_correct_qwen35_6k \
  outputs/reports/polar_6k_train.html
```

也可直接传入 JSONL，原始数据不做归一化、不改变参考答案：

```bash
bash scripts/render_dataset_html.sh outputs/data/raw/polar/openseeker_v1_data.jsonl
SPLIT=all bash scripts/render_dataset_html.sh outputs/data/openseeker/polar_correct_qwen35_6k
LIMIT=100 OFFSET=0 bash scripts/render_dataset_html.sh /path/to/train.jsonl
```

默认导出全部记录；数据很大时建议用 `LIMIT` 和 `OFFSET` 分批导出，避免 HTML 和浏览器
内存过大。`SPLIT=all` 合并存在的 train、validation、test，不包含 rejected；可单独设置
`SPLIT=rejected`。第二个位置参数或 `OUTPUT_HTML` 可指定输出路径；重复运行会覆盖同名
HTML。`PYTHON` 可指定解释器路径。

## 查看内容

- 全文搜索 ID、问题、参考答案和原始轨迹；筛选有/无非空观测正文的样本。
- 按角色查看消息；区分思考、工具调用、观测内容及最终答案，支持展开和折叠。
- 查看原始 token 长度字段、协议、来源行号、数据集 manifest 和完整原始 JSON。
- 样本列表每页 50 条；上一条/下一条按当前筛选结果导航。

观测识别按消息角色处理：OpenSeeker 的后续 user 消息中的 `<tool_response>` 正文、
Search-R1 在 assistant 之后完整 user `<information>` 消息的正文，以及原始 `tool`
结果。system/assistant 中的标签示例不计入观测。原始 `tool` 消息未运行训练归一化。
空正文不计数；无观测只表示该文本规则未找到非空正文，不直接判定数据无效。

页面统计的字符数不是 token 数；已有 `lengths` 字段原样展示，未重新调用 tokenizer。
本工具不验证协议、截断、监督标签或实际训练 token 区间。数据中的 HTML 和标签以安全
文本展示，不作为网页执行。

## Python 入口与模板

```bash
python scripts/render_dataset_html.py /path/to/dataset \
  --split train --limit 100 --offset 0 \
  --output outputs/reports/dataset.html --title '训练数据检查'
```

模板为 `scripts/templates/dataset.html`，可修改样式或通过 `--template` 指定副本。
保留唯一的 `__DATASET_DATA__` 占位符。输入格式错误会报告文件名和行号。
