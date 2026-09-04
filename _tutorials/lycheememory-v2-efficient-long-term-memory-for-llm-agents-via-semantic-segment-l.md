---
layout: default
title: "LycheeMemory V2：告别逐轮蒸馏，写入Token骤降86%还拿下双榜第一"
description: "要让大语言模型（LLM）从只会单轮对话的聊天机器人，蜕变为能长期伴随用户的智能体，外部记忆系统不可或缺。然而，当前主流智能体记忆系统的运行逻辑存在一个隐形却极其昂贵的代价：每产生一轮对话，系统就急于调用一次LLM去抽取实体、总结事实并更新记忆库。"
arxiv_id: "2608.12990"
published_at: "2026-09-04T11:26:52.370742+08:00"
topics:
  - "知识系统"
  - "AI Agent"
tags:
  - "LLM agents"
  - "LoCoMo"
  - "LongMemEval-S"
  - "LycheeMemory V2"
  - "lightweight structured indexes"
  - "query-planned evidence retrieval"
related_tutorials:
  - "agentic-memory-learning-unified-long-term-and-short-term-memory-management-for-l"
  - "simplemem-efficient-lifelong-memory-for-llm-agents"
  - "dynamic-affective-memory-management-for-personalized-llm-agents"
  - "forgetful-but-faithful-a-cognitive-memory-architecture-and-benchmark-for-privacy"
---

<p class="paper-original-title" lang="en">LycheeMemory V2: Efficient Long-Term Memory for LLM Agents via Semantic Segment-Level Consolidation</p>

<img src="/images/2608.12990v1/A__title.webp" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

要让大语言模型（LLM）从只会单轮对话的聊天机器人，蜕变为能长期伴随用户的智能体，外部记忆系统不可或缺。然而，当前主流智能体记忆系统的运行逻辑存在一个隐形却极其昂贵的代价：每产生一轮对话，系统就急于调用一次 LLM 去抽取实体、总结事实并更新记忆库。

> ArXiv URL：https://arxiv.org/abs/2608.12990v1

这种被称为“急迫整合”（eager consolidation）的机制，导致智能体对话越长，写入端的 Token 消耗就越失控。若为了省钱改用粗粒度的整段总结，关键的细粒度时间线和指代线索又会被抹杀；若依赖查询时的多跳推理去弥补，又会使推理成本大幅飙升。来自哈尔滨工业大学（Harbin Institute of Technology）的研究团队给出了新的破局思路：他们推出了长期记忆框架 **LycheeMemory V2**，核心判断是——**智能体记忆的精度与成本平衡，不仅取决于记住了什么，更取决于以何种粒度进行整合。**

<img src="/images/2608.12990v1/motivation.webp" alt="Motivation of LycheeMemory" style="width:85%; max-width:600px; margin:auto; display:block;">

该方案彻底放弃了昂贵的逐轮调用，转而采用轻量级 Embedding 驱动的“语义分段级整合”（Semantic Segment-Level Consolidation）。评测显示，在以 GPT-4.1-Mini 为底座的基准测试中，LycheeMemory 在 LoCoMo 上达到 89.22% 的准确率，在长上下文的 LongMemEval-S 上达到 92.20%，均位列第一；与代表性记忆框架 A-Mem 相比，其记忆构建阶段的 Token 消耗分别暴跌 86.0% 和 75.9%，且查询端 Token 同样保持更低水平。

### 记忆系统不该沦为逐轮调用的“碎纸机”

现有的长期记忆架构大致遵循几种路径：Mem0 与 A-Mem 等系统习惯在每一轮交互后立即唤醒 LLM，以此保证记忆单元的高度提纯；而 MemoryOS 这类方案则引入多层分级架构，但在吸纳新对话时依然严重依赖高频生成模型。在长期运行场景下，这种高频调用的开销令人难以承受。

更深层的缺陷在于信息孤岛效应。用户在与智能体交流时，一个完整事件或决策往往跨越 3 到 5 个来回。逐轮急迫整合相当于把连贯的对话切碎成碎片，模型不得不频繁处理上下文残缺的代词（如“它”、“那个地方”）和相对时间（如“明天”、“下周三”）。许多系统随后不得不依赖复杂的跨节点链接或查询时的反复迭代检索去“打补丁”，把写入端的偷懒变成了查询端的负担。

直接换成固定轮数的窗口打包同样行不通。机械地每 5 轮切一刀，往往会把一句话切成两半，或者把上一件事的结尾与下一件事的开头拼在一起，破坏事件的语义完整性。LycheeMemory 的立足点在于：必须在对话流中找到自然的语义停顿点，在单次批处理中低成本地完成高质量沉淀。

### 语义分段与结构化索引：两端兼顾的协同架构

LycheeMemory V2 构筑了一条从在线自适应分段、段落级编码，到多路结构化索引与规划检索的闭环流程。整个系统将昂贵的 LLM 生成调用严格限制在必要节点上，其余环节全部交由轻量级 Embedding 和确定性算法处理。

<img src="/images/2608.12990v1/pipeline.webp" alt="Overview of LycheeMemory" style="width:85%; max-width:600px; margin:auto; display:block;">

#### 1. 无 LLM 介入的在线语义边界检测

系统在对话推进时维护一个活跃分段（Active Segment）。针对每一个到来的新回合 $x_t$，系统利用小型向量模型 `text-embedding-3-small` 计算其与当前分段的语义突变程度（Semantic Surprise）以及分段内聚度的衰减量：




{% raw %}$$ s_{t}=1-\max\bigl(\mathrm{sim}(\mathbf{e}_{t},\mathbf{c}_{k}),\mathrm{sim}(\mathbf{e}_{t},\mathbf{h}_{k})\bigr) $${% endraw %}






{% raw %}$$ d_{t}=\max\bigl(0,\mathrm{Coh}(S_{k})-\mathrm{Coh}(S_{k}\cup\{x_{t}\})\bigr) $${% endraw %}



结合归一化的分段 Token 长度压力 $L_t$ 与轮数压力 $N_t$，系统通过逻辑斯蒂函数输出综合切分概率 $p_t$：




{% raw %}$$ p_{t}=\sigma\bigl(b+w_{s}\phi(s_{t})+w_{c}d_{t}+w_{l}L_{t}+w_{n}N_{t}\bigr) $${% endraw %}



只要分段尚未饱和且未发生主题转移，中间回合就会持续缓存。整个切分判定不调用任何生成式 LLM，使得写入调用的总次数从原本的交互总轮数 $T$ 骤降到实际分段数 $\lvert \mathcal{S} \rvert$。

#### 2. 段落级去耦编码与有界指代消歧

当一个语义段落封包后，系统才会发起一次 LLM 编码调用，将其蒸馏为一组自包含的结构化记忆记录。每条记录均显式包含自身陈述、记忆类型、涉及实体、所属主题、时间范畴以及原始对话索引：




{% raw %}$$ r_{i}=\bigl(\mathrm{id}_{i},\tau_{i},\mathrm{text}_{i},\mathcal{E}_{i},\mathcal{K}_{i},\mathcal{T}_{i},\mathrm{src}_{i}\bigr) $${% endraw %}



为了防止分段切分造成跨段上下文断裂，LycheeMemory 并不会将全量历史回传，而是维持一个轻量级的动态消歧上下文 $\rho_{k+1}$。该上下文只包含上一段提炼出的实体别名、指代关系及最近的记忆摘要，并设定了严格的 Token 上限。这让当前段落内的相对时间（如“昨天发给张三”）能够在编码阶段被直接还原为绝对客观事实，消除了查询时再去多跳反推的麻烦。

#### 3. 确定性元数据索引与规划驱动的单步多路召回

记忆落库后，系统无需额外的 LLM 总结开销，而是直接依据记录中现成的元数据，自动衍生构建出实体索引、主题索引、精确时间范围索引以及保留同段共现关系的事件框架（Event-frame）节点。

面对用户的复杂提问，LycheeMemory 摒弃了消耗巨大、多轮往复的 Agent 自主发散搜索，转而采用“单次规划 + 确定性多路召回”。LLM 仅在最开头被调用一次，将问题解析为具体的检索路由需求；紧接着，系统并行从直接向量库、结构化节点、时间范围与原始片段中抽取候选，最后经由倒排秩融合（Reciprocal Rank Fusion, RRF）以及多样性重排算法输出紧凑上下文：




{% raw %}$$ \mathrm{RRF}(d)=\sum_{j=1}^{m}\frac{1}{\kappa+\mathrm{rank}_{j}(d)} $${% endraw %}



这种设计将非线性的复杂检索收敛为纯粹的向量检索与算术运算，从根源上锁死了查询端的计算消耗。

### 评测落地：双基准登顶与断崖式成本缩减

评测选用了两个具代表性的长期对话基准：侧重生活陪伴、多会话（平均约 600 轮）交叉提问的 **LoCoMo**，以及侧重长程任务执行、上下文高达 11.5 万 Token 的高负载基准 **LongMemEval-S**。

#### 1. 复杂推理场景下的准确率跃升

在 GPT-4.1-Mini 作为底座的实验中，LycheeMemory V2 全面刷新了各项指标。

* 在 **LoCoMo** 上，LycheeMemory 取得了 89.22% 的整体准确率，相比 A-Mem 实现了超过 20 个百分点的断崖式领先。尤其是在最具挑战的多跳推理（87.23% vs. 59.93%，提升 27.3 个百分点）和开放域问题（67.71% vs. 42.71%，提升 25.0 个百分点）上表现尤为亮眼。甚至相较于把所有历史暴力塞进窗口的 Full Context 方案（84.80%），也高出了 4.42 个百分点。

* 在 **LongMemEval-S** 上，对话长度急剧扩展，Full Context 因受制于“大海捞针”的位置衰减效应，准确率大幅滑落。而 LycheeMemory 仍稳定拿下 92.20% 的最高分。其中，时间推理维度从 A-Mem 的 52.63% 大幅拉升至 87.22%，多会话交叉推理达到 87.97%，知识更新题型更是斩获了 97.44% 的高分。这证明其基于语义段落固化时间戳的设计，精准切中了长期记忆的关键痛点。

#### 2. 写入 Token 骤降八成以上，查询端同样节省

对比记忆构建与实际查询两个维度的消耗，可以看到该机制在效率上的绝对优势：

* **构建成本（Construction Tokens）**：在 LoCoMo 测试中，LycheeMemory 平均每段对话仅耗费 204.1K 写入 Token，相较于 A-Mem 的 1459.9K 骤降了 **86.0%**，相较于 Mem0 同样节省了 86.6%；在超长文本的 LongMemEval-S 上，其构建开销同样比 A-Mem 降低了 **75.9%**。

* **查询成本（Query Tokens）**：许多主打精简写入的框架往往会将开销转嫁给查询端。而得益于写入时事实已被充分消歧、索引架构高度结构化，LycheeMemory 在查询时只需极窄的上下文即可完成高精度回答，其实际查询 Token 消耗相较 A-Mem 不升反降。

### 对未来智能体架构的技术启示

LycheeMemory V2 的实验数据破除了一种行业惯性，即“记忆要好，就必须每轮都调用 LLM 深度蒸馏”。通过将批处理思想与语义边界检测结合，不仅能够直接砍掉 80% 以上无意义的写入开销，还能因为保护了事件上下文的自然连续性，反向带来推理准确率的大幅提升。

对于工程落地而言，这种将昂贵计算（LLM 语义结构化提炼）限定在“语义完备点”、将高频调度（边界切分、多路召回、重排过滤）托付给“低成本算法模型”的解耦模式，为真正能够全天候低成本运行的伴随式个人智能体提供了一套极具实操价值的系统范式。
