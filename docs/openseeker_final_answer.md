# OpenSeeker 最终精简答案协议

协议版本为 `openseeker-chatml-v2-final-answer`。预处理和推理共同使用 `FINAL_ANSWER_INSTRUCTION`，要求模型在最后输出一个完整的 `<final_answer>...</final_answer>`，其中只包含精简答案，不写解释、开场白或 Markdown。

## 训练数据与损失

保留原始思考、工具调用、工具返回，以及原有 `<answer>` 长解释。预处理在最后一条 assistant 消息末尾追加标准答案，例如：

```text
<think>原始推理……</think>
<answer>Euler's theorem 的原始解释……</answer>

<final_answer>Euler's theorem</final_answer>
```

标准答案来自独立的 `answer` 字段，必须是非空字符串；缺失答案、未明确选择的列表/对象答案及包含保留 final_answer 标签的答案会拒绝处理，原因写入预处理的 rejected.jsonl。系统提示词只写格式说明，不写标准答案。已有完整 final_answer 在升级时会被移除并以参考答案重建；不闭合的标签会报错。

所有 assistant 内容均可计算主 CE，包括原始思考、工具调用、长解释、新 final_answer 的标签和短答案及消息结束符；system、问题和工具返回仍不计算主 CE。各类别现在支持独立权重，配置见 [分段损失](openseeker_loss_segments.md)。final_answer 协议训练模型“额外输出精简结论”，并不直接减少原始思考和解释的生成长度。

## 重新预处理

旧的 v1 数据可以作为预处理输入；升级保留原 ID 和元数据，新增共用提示词和答案块，并丢弃旧长度。prepare 会重新使用指定 tokenizer 测量。v2 数据重复 normalize 不会重复追加提示词或标签。

例如，用已有训练/验证/测试文件生成新的 Qwen3 数据目录（在具有对应 tokenizer 的服务器执行）：

```bash
python scripts/prepare_openseeker.py \
  --input outputs/data/openseeker/polar_correct_qwen35_16k/train.jsonl \
  --input outputs/data/openseeker/polar_correct_qwen35_16k/validation.jsonl \
  --input outputs/data/openseeker/polar_correct_qwen35_16k/test.jsonl \
  --source polar \
  --output outputs/data/openseeker/polar_correct_qwen3_16k_final_answer_v2 \
  --models Qwen/Qwen3-4B-Thinking-2507 \
  --max-length 16384 --offline
```

这只重处理已有筛选集；如需重新选择全量数据，请改用原始输入。用其他模型时修改 `--models`，保持其标识与训练模型一致。新增提示词和标签后，部分原本贴近长度上限的样本可能被排除。

将 `train_openseeker.sh` 内的 `dataset` 改为新目录，并选择对应 `model_family`，再训练。原文件不被覆盖。实验入口和训练数据加载器会拒绝旧协议；refilter 也会拒绝用旧协议长度构造新训练集，并将当前协议写入派生 manifest。

## 推理与指标

推理只从数据取前两条 system/user 消息，使用相同的格式提示词，绝不取最后的标准答案消息；模型生成内容写入轨迹。在工具调用结束、ChatML 消息结束或 `</final_answer>` 闭合时停止当前轮生成。

评测只接受恰好一个完整且非空的 final_answer 块，并将其内容写入 prediction。没有该标签、标签为空、未闭合或重复时 prediction 为空，状态为 empty_answer；被生成预算截断时仍为 generation_limit。不会从 `<answer>` 或整段解释兜底取答案。

EM/F1 的数学定义保持原有 OpenSeeker 口径：EM 忽略大小写及空白差异，F1 复用现有标准化后的词重叠实现；失败和空参考答案计 0。此改动对齐答案提取目标，没有改变 EM 标准化规则。指标 JSON 记录 protocol 和 prediction_source，便于区分新旧结果。

HTML 模板会将 `<answer>` 标为“原始回答”，将 `<final_answer>` 标为“最终精简答案”。旧预测文件仍可展示。旧 checkpoint 未接受新协议训练，可能只生成 `<answer>`，在新评测下会被判未作答；历史指标保持不变，不应直接将格式变化解释为模型能力变化。
