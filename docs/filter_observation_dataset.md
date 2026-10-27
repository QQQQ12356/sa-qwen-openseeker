# 剔除无观测正文的轨迹

无观测样本可能不使用 Beacon embedding、投影和 linear writer；不同 rank 使用不同样本，
可能让当前 ZeRO-2 归约流程错位。剔除这些样本是有用的排查和数据选择方法，但不能保证
消除所有动态计算图问题，也会改变训练和评测的数据分布。

## 一键执行

默认输入为目标服务器上的 `outputs/data/openseeker/polar_correct_qwen35_6k`：

```bash
bash scripts/filter_observation_dataset.sh
```

自定义输入、输出（输出目录必须不存在）：

```bash
bash scripts/filter_observation_dataset.sh \
  outputs/data/openseeker/polar_correct_qwen35_6k \
  outputs/data/openseeker/polar_correct_qwen35_6k_with_observations
```

只过滤单个文件时也输出到新目录，保留原文件名：

```bash
bash scripts/filter_observation_dataset.sh /path/to/train.jsonl /path/to/new_dataset
DATA_FORMAT=searchr1 bash scripts/filter_observation_dataset.sh /path/to/searchr1.jsonl
```

仅需 Python 3 标准库；可用 `PYTHON=/path/to/python` 指定解释器。不加载模型、不重新分词。

## 过滤规则

- OpenSeeker：只统计索引大于 1 的 user 消息中的完整 `<tool_response>...</tool_response>`
  正文，与 `search_comp/data/openseeker.py` 的 `pieces()` 边界一致。
- Search-R1：首个 assistant 后的 user 消息必须是完整 `<information>...</information>`
  块，与 Search-R1 数据加载器一致；格式不符时报告文件和行号，不静默删除坏数据。
- 至少有一个非空正文才保留。system/assistant 中的标签示例不算观测；只有空标签会被剔除。
- 空格或换行正文仍保留，因为它们可能编码为训练 token。未实际调用 tokenizer，不能据此
  保证 token 级 `regions` 非空；已有 `lengths` 原样保留。
- 输入必须已经准备为 `messages` 格式。原始 `trajectory` 或 `tool` 角色先运行项目数据准备
  流程，不能将原始工具消息直接当作训练可压缩区。

## 输出与追溯

目录输入会分别过滤现有的 train、validation、test，不重新划分。输出包括：

| 文件 | 内容 |
|---|---|
| `train.jsonl` 等 | 保留的完整原始记录，不改答案、消息或测量字段 |
| `removed/train.jsonl` 等 | 被剔除的完整原始记录 |
| `removed_records.jsonl` | 剔除原因、原始文件、行号和 ID |
| `filter_report.json` | 各 split 输入/保留/剔除数、规则和源文件 SHA256 |
| `manifest.json` | 输入目录有 manifest 时生成，保留协议、tokenizer 指纹和长度参数，更新计数 |
| `parent_manifest.json` | 原始 manifest 备份，原来的聚合统计不会被误认为新数据统计 |

不会复制旧的 rejected 文件，也不会把旧协议升级为 v2。旧协议数据仍需重新 prepare，
不能通过过滤绕过训练入口的协议检查。输出已存在则报错；数据格式错误时不留下半成品目录。

## 过滤后训练与查看

检查 `filter_report.json`，尤其是训练集和验证集是否为空。当前实验入口要求两者非空。
然后把训练脚本 YAML 中的 `dataset` 修改为新目录。脚本不会自动改动训练配置。

```bash
bash scripts/render_dataset_html.sh \
  outputs/data/openseeker/polar_correct_qwen35_6k_with_observations \
  outputs/reports/filtered_dataset.html
```

若仍出现同一通信序号、不同张量大小的 ALLREDUCE 超时，应继续排查 ZeRO-2 动态参数参与
和梯度归约顺序；不要只增加 NCCL 超时时间。可使用支持 unused parameters 的 DDP 做对照，
但需检查显存占用。过滤后评测集与原评测集不同，指标不宜直接比较。
