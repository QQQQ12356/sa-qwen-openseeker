# OpenSeeker 分段损失权重

OpenSeeker 复用 Search-R1 的逐 token 加权机制，支持 Beacon 和 Plain SFT。设置在训练配置的 `beacon.loss_segments`；Plain 也读取这个位置，另支持顶层 `loss_segments`（顶层优先）。`scripts/train_openseeker.sh` 使用脚本内嵌 YAML，直接改该文件；通过 `scripts/openseeker.sh` 运行时修改 `configs/openseeker.yaml`。

```yaml
beacon:
  loss_segments:
    think_tag: 1.0
    think_content: 1.0
    tool_call_tag: 1.0
    tool_call_content: 1.0
    answer_tag: 1.0
    answer_content: 0.2
    final_answer_tag: 2.0
    final_answer_content: 10.0
    other: 1.0
```

以上为入口默认提供的可调整起始值，尚未通过训练实验确定最优值。降低原始长解释权重、提高 final_answer 权重，同时保留思考和工具调用的监督。标签和正文分开，是为了能独立训练格式正确性与答案内容。

| 类别 | 对应 token |
| --- | --- |
| `think_tag / think_content` | `<think>`、`<thinking>` 的开闭标签 / 思考正文 |
| `answer_tag / answer_content` | `<answer>` 的开闭标签 / 原始回答和解释 |
| `final_answer_tag / final_answer_content` | `<final_answer>` 的开闭标签 / 最终精简答案 |
| `tool_call_tag / tool_call_content` | 工具调用外层及各调用开闭标签 / 调用 JSON、参数和外层块内空白 |
| `search_tag / search_content` | Search-R1 `<search>` 标签 / query，OpenSeeker JSON 工具调用不用这两类 |
| `other` | 其他 assistant 文本、块间换行及 ChatML 消息结束符 |

system、问题、工具返回保持 label=-100，类别为 ignore，不受这些权重影响。OpenSeeker 的 search 与 visit 调用均归为 tool_call，暂不按工具名称细分。

## 权重语义

```text
L_CE = Σ(有效目标 token 的权重 × CE) / Σ(有效目标 token 的权重)
```

这是逐 token 相对权重，不是各类损失固定占比。例如思考有 1000 token、权重 1，最终答案有 5 token、权重 10，那么答案的归一化权重占比约为 `50 / 1050 = 4.76%`（此例忽略其他类别）；实际损失贡献还取决于各 token 的 CE。

未配置的类别默认为 1。`loss_segments: {}` 恢复全部等权，与旧 CE 行为一致。`0` 或 `{enabled: false}` 关闭该类直接损失，不占分母；该类文本仍保留在输入中，仍可能通过后续目标参与上下文梯度。不要把降低解释权重理解成自动停止生成解释。

例如，保留 final_answer 格式训练，但关闭原始 answer 正文监督：

```yaml
answer_content: {enabled: false}
final_answer_tag: {enabled: true, weight: 2.0}
final_answer_content: {enabled: true, weight: 10.0}
```

只监督 final_answer 时，还需要显式将 think、tool_call、answer、search、other 的标签和正文权重设为 0。通常建议保留格式和工具调用监督。所有类别均关闭时主 CE 是可反向传播的 0；辅助损失是否存在取决于其独立配置。

## 实现与兼容性

- Token 类别跟随目标标签一起做因果移位，位置 i 的预测使用位置 i+1 的类别权重。
- 新类别追加在原有类别之后，不改变旧 checkpoint 的类别编号；旧配置未写的新类别默认为 1。
- OpenSeeker 加权训练要求支持 offset_mapping 的 tokenizer，并严格校验分类分词结果与训练 token ID 一致；无法对齐时报错，不静默退化为等权。
- Beacon 沿用原有稀疏分块加权 CE。Plain 将类别传入独立 Trainer，使用同一套归一化公式；全默认时仍走原标准 Trainer。
- 修改权重不会改变 token 序列、loss mask 或压缩区，因此不必重新测量已准备好的 v2 数据；从 v1 升级到 final_answer v2 仍需重新预处理。
- 分段权重控制主 CE；Beacon 的 information 续写、readout 蒸馏和 output KL 有自己的全局权重，不使用上述数值乘子。output KL 会排除分段权重为 0 的位置；正数权重大小不直接缩放 KL。stratified 采样新增 final_answer 和工具调用正文队列，final_answer 优先参与轮询。
- 新训练 YAML/实验快照会记录配置，Beacon 权重随模型配置保存；Plain 同样保存配置和 dataset_summary。推理和 EM/F1 定义不受分段权重影响。
