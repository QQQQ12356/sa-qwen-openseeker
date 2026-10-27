# sa-qwen-openseeker

在 `../sa-qwen` 基础上做增量适配：保留 Qwen3 / Qwen3.5 的 Beacon 压缩模型、蒸馏损失、LoRA、Search-R1 训练和评测；新增 OpenSeeker 多工具轨迹数据层和完整实验入口。**没有启动真实模型训练。**

默认模型为 `Qwen/Qwen3-4B-Thinking-2507` 和 `Qwen/Qwen3.5-4B`。实验参数集中在 [configs/openseeker.yaml](configs/openseeker.yaml)。

- [使用指南](docs/openseeker_workflow.md)：下载、JSONL 转换、筛选、训练、测试、归档。
- [多卡训练](docs/multi_gpu_training.md)：Beacon / plain 的 torchrun、DeepSpeed、CPU 卸载和显存边界。
- [一键测试说明](docs/openseeker_testing.md)：EM/F1、训练集拟合诊断、多卡并行、三种后端、语料构建与结果排查。
- [OpenSeeker 历史证据建库](docs/openseeker_corpus.md)：一键抽取 visit 证据、合并去重、溯源及 corpus 评测。
- [数据分析](docs/openseeker_analysis.md)：全量结构、质量标签、工具、token 分布、筛选偏差及来源限制。
- [实现说明](docs/openseeker_implementation.md)：协议、压缩区域、训练推理一致性、代码改动边界。
- [OpenSeeker 分段损失](docs/openseeker_loss_segments.md)：分别配置思考、工具调用、原始回答和最终答案的训练权重。
- [最终精简答案协议](docs/openseeker_final_answer.md)：`final_answer` 训练目标、数据升级及评测提取。
- [离线轨迹 HTML 报告](docs/predictions_html.md)：统一转换评测 JSONL，浏览问题、答案、轨迹和指标。
- [数据集 HTML 浏览器](docs/dataset_html.md)：一键查看训练数据、工具观测、长度统计和无观测样本。
- [过滤无观测轨迹](docs/filter_observation_dataset.md)：保留原始数据和划分，一键生成含观测块的数据集及剔除审计。
- [原项目文档](docs/README.md)：原有 Search-R1 和压缩算法说明。

## 已生成的数据

`outputs/data/raw/polar/` 保留官方快照与可读 JSONL；`outputs/data/openseeker/` 包含完整规范化数据、逐条长度记录，以及两个正确性筛选版本：

| 数据集 | train | validation | test |
|---|---:|---:|---:|
| `polar_correct_qwen35_16k`（默认） | 644 | 9 | 11 |
| `polar_correct_qwen35_32k` | 1,696 | 23 | 40 |

这些 token 长度是 **Qwen3.5-4B 的实测值**。切换 Qwen3 时应先用对应 tokenizer 重新处理；入口会阻止使用尚未测量的模型 tokenizer。原数据共 11,677 条，包含 6,728 条 Incorrect，不能直接全部当作正确 SFT 轨迹。

Zephyr 数据的下载/Parquet→JSONL/协议适配已实现并以结构化测试覆盖，但当前终端 DNS/网络限制导致无法下载该版本；其全量统计尚未实测。详情见数据分析文档。

## 快速使用

```bash
# 使用已有环境；也可在匹配 CUDA 的环境中安装 requirements.txt。
export PYTHON=/home/huangzj/miniconda3/envs/search-comp-qwen3.5/bin/python
bash scripts/openseeker.sh test

# 只生成实验计划和配置快照，不加载模型，不训练。
bash scripts/openseeker.sh plan

# 后续由实验人员执行；RUN_DIR 使用 plan 输出的路径。
bash scripts/openseeker.sh train --run-dir "$RUN_DIR"
bash scripts/openseeker.sh eval --run-dir "$RUN_DIR"

# 历史 visit 证据建库；先编辑脚本顶部 INPUTS 和 OUTPUT_DIR。
# bash scripts/build_openseeker_corpus.sh

# 构建数据集；先编辑输入/输出路径及 VAL_FRACTION、TEST_FRACTION（训练集取剩余比例）。
# bash scripts/prepare_openseeker.sh

# 一键训练 / 测试：先编辑各脚本顶部参数；测试时填写 RUN_DIR 或 MODEL_PATH。
# bash scripts/train_openseeker.sh
# bash scripts/test_openseeker.sh
```

评测默认是严格动作匹配的离线 replay 诊断；真实在线评测应配置 HTTP 工具服务。两者结果明确区分，不将 replay 分数标为在线搜索能力。默认 16k、LoRA、batch=1 是 80GB 卡上的保守起点；未在 80GB GPU 上实测显存或吞吐。
