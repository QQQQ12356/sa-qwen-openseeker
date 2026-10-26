# 实现边界与代码阅读路线

基于 `/home/huangzj/proj/searchagent/sa-qwen` 工作树复制代码、配置、测试和历史文档，原仓库 HEAD 为 `b7356a6a4b03013a06b0d91bec170a2c852da031`。未复制原模型输出、Git 元数据或环境。Beacon 模型、Qwen3/Qwen3.5 分派、LoRA 合并和蒸馏计算均沿用原实现；核心旧文件只修改两个训练器的数据类选择。

## 数据和协议

`search_comp/data/openseeker.py` 的 `normalize` 处理两类来源：

1. 官方 `trajectory` 保持 system 和 assistant 原文，工具观察仍是带 `<tool_response>` 的 user 消息。
2. OpenAI 风格 `messages/tools` 把结构化函数调用渲染为官方调用包裹。JSON 字符串形式的 arguments 转为对象；独立 reasoning_content/reasoning 放入 `<think>`；tool 角色转换为包裹后的观察；系统提示追加工具定义及调用格式。

工具参数列表的顺序与批量调用顺序保留。结构化 tool_call_id 用于校验是否存在对应调用、是否重复/遗漏，当前对携带 ID 的乱序观察明确拒绝；不会丢弃关联后静默绑定到错误调用。转换后的执行协议按顺序关联函数与观察，不要求模型重新生成提供商的 opaque ID。未知工具名仍被保留，执行由 registry/网关决定。

`validate` 检查开头 system/question、工具动作后的观察、观察包装完整性和末尾非工具 assistant 回答。多模态内容、孤立观察、损坏 JSON、不完整动作显式报错。正确性标签独立于语法校验：语法合法不表示推理正确。

`pieces` 和 `encode` 是训练/推理共享的唯一序列化入口。采用显式 ChatML；不调用可能随模型模板删去历史 thinking 的 `apply_chat_template`。每条 assistant 的内容及结束符参与 loss；system、问题、观察、角色头全部 mask。只有 `<tool_response>` 标签之间的正文形成半开 token 区间，标签和角色边界不压缩。工具正文中出现 `<tool_call>` 文本不会被当作模型新动作。

工具调用仍进入 assistant 的 CE 和原续写/蒸馏训练路径。OpenSeeker 现在支持 think、answer、final_answer、tool_call 的标签/正文分段权重，并支持 stratified 输出蒸馏；不将工具调用 JSON 错分成 Search-R1 的 search query，也不启用基于 Doc 编号的子文档切分。详见 [分段损失配置](openseeker_loss_segments.md)。

`OpenSeekerDataset` 仅在内存保存 JSONL 字节偏移，逐条读取和编码。超出 max_length 直接报错，避免训练阶段静默切断。实际数据流程先全量筛选，训练端的判断是第二道约束。

## 训练集成

`search_comp/data/factory.py` 对 `searchr1/openseeker` 显式分派。原 `SearchR1SFTDataset` 完全保留。

- `beacon_trainer.py` 改用 factory，向数据类传入 data_mode / max_length；继续使用原 collator 的 regions、原 Beacon 模型与 loss。
- `plain_sft_trainer.py` 的 wrapper 选择相同数据类、相同编码/labels，丢弃 regions，因此 plain/beacon 能进行一致的数据对照。
- 原 native trainer、检索评测和 BM25 接口保持原有语义；主 OpenSeeker 路径使用上述 plain/beacon 训练器。

没有更改原算法中的 attention、GatedDeltaNet、beacon 位置、KV 状态或蒸馏方程。`beacon_output_distill_weight: 0` 是新实验配置中的省显存默认选项，改为正数即可启用原 teacher 路径。

## 推理执行

`search_comp/evaluation/tool_agent.py` 每轮通过同一 `encode` 重建完整历史及压缩区，然后调用标准 `generate` 或 `beacon_generate(reuse_cache=False)`。完整历史重算的代价更高，但避免新增多工具增量缓存协议；这里优先保证可核验的上下文一致性。多轮状态由历史消息显式维护。

生成工具包裹结束或 assistant 结束符时停止。解析整个批次，再执行所有函数；工具返回转换为与训练相同的观察。回放只匹配下一批完整动作；HTTP 网关逐条发送函数名和参数；Registry 可直接绑定 Python 函数。工具失败、上下文耗尽、单观察超预算、生成截断、轮数耗尽都独立记录状态，未完成内容不计为成功答案。

`evaluate_openseeker.py` 加载已合并 checkpoint，不直接误载 LoRA adapter；统一实验入口从训练 summary 获取最终模型路径。推理和 checkpoint 实际表现仍需目标环境跑完整评测，CPU 小模型测试只证明数据/接口/计算路径可工作。

## 复现和测试

脚本分工：download 保留原文件并转 JSONL；audit 检查原始结构；prepare 实际分词并筛选；refilter 复用测量换阈值；report 做质量/长度分层；plot 可选绘图；experiment 保存配置与数据/代码指纹；evaluate 运行模型与工具。

新增测试覆盖原文保留、结构化 arguments/reasoning、调用 ID、loss masking、观察区边界、每个 assistant 位置的训练/推理前缀一致、错误协议、严格 replay、未知工具、长度上限、截断状态，以及 Qwen3/Qwen3.5 小模型真实 Beacon 前向与生成。已有 Search-R1 回归测试全部保留。测试不构成 4B 模型训练或 80GB GPU 验证。

最终本机回归结果：**335 passed，2 skipped**（约 26 秒）；完整记录见 [validation.json](openseeker/validation.json) 与 [pytest.log](openseeker/pytest.log)。还实跑了 7 行带重复题和 Incorrect 标签的数据处理夹具，验证实际 Qwen3.5 tokenizer、published validation 优先归属、去重和默认质量筛选。80GB 实验前可从 [示例计划](openseeker/example_plan.json) 查看生成的命令；该计划仅写出配置，没有执行 train。
