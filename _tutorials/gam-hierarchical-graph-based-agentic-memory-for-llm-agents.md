---
layout: default
title: "告别记忆污染！GAM分层图记忆让Agent长程推理提升86%"
description: "当前的大语言模型虽然能在单次对话中对答如流。但一旦进入长期交互，它们常常会陷入“记了新的，忘了旧的”的窘境。现有的记忆系统大多采用统一流式更新。这意味着每一次日常闲聊的噪音，都可能直接污染已经建立的核心知识库。如何让Agent既能快速吸收新信息，又能稳固保持长期记忆？"
arxiv_id: "2604.12285"
topics:
  - "AI Agent"
  - "基础模型"
tags:
  - "GAM"
  - "LLM agents"
  - "context precision enhancement"
  - "event progression graph"
  - "graph-guided multi-factor retrieval"
  - "hierarchical graph-based agentic memory"
related_tutorials:
  - "agentic-memory-learning-unified-long-term-and-short-term-memory-management-for-l"
  - "from-experience-to-strategy-empowering-llm-agents-with-trainable-graph-memory"
  - "back-to-basics-let-conversational-agents-remember-with-just-retrieval-and-generation"
  - "skillos-learning-skill-curation-for-self-evolving-agents"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">GAM: Hierarchical Graph-based Agentic Memory for LLM Agents</p>

当前的大语言模型虽然能在单次对话中对答如流。
但一旦进入长期交互，它们常常会陷入“记了新的，忘了旧的”的窘境。
现有的记忆系统大多采用统一流式更新。
这意味着每一次日常闲聊的噪音，都可能直接污染已经建立的核心知识库。

> **ArXiv URL**：http://arxiv.org/abs/2604.12285v1

如何让Agent既能快速吸收新信息，又能稳固保持长期记忆？
这项由浙江大学等机构提出的最新研究给出了明确的答案。
该研究提出了`**分层图代理记忆**（**Hierarchical Graph-based Agentic Memory, GAM**）`框架。
通过精准解耦记忆的“编码”与“巩固”过程，彻底解决了这一冲突。
实验表明，GAM在复杂多方交互中推理准确率最高提升达86%。

### 核心痛点：记忆污染与架构僵化

要理解GAM框架的理论突破，首先要审视现有架构的底层局限性。

目前的流式记忆系统追求极致的上下文更新速度。
诸如MemGPT或Mem0等架构，像流水账一样处理输入。
它们将所有新信息直接追加或融合到长期记忆的流中。
这种“不设防”的直接写入机制极易引发严重的`**记忆污染**（**Memory Contamination**）`。
随着对话的不断拉长，瞬时噪音不断累积。
最终会导致已有知识的语义发生漂移，或者关键节点变成结构上的孤岛。

<img src="/images/2604.12285v1/x1.webp" alt="Refer to caption" style="width:85%; max-width:450px; margin:auto; display:block;">

另一种技术路线是离散的结构化记忆体系。
比如GraphRAG，它将知识严格组织为静态图谱，抗干扰能力极强。
但这类架构的更新成本高昂，且往往过于僵化。
它们无法适应自然对话中频繁跳跃的叙事节奏。
最终产生的往往是碎片化的话语表征，难以捕捉用户状态的连续演变。
本文提出的GAM框架，正是为了打破这种鱼与熊掌不可兼得的困局。

### 核心机制：解耦缓存与巩固

GAM的设计灵感来自于人类基于睡眠的记忆巩固机制。
它将记忆的生命周期明确划分为两个物理隔离的阶段。

这里引入一个克制的比喻，来解析这个稍显复杂的双层架构。
我们可以将GAM系统想象为一个分工明确的档案管理室。

**1. 局部工作台：情景缓冲状态**
档案室里有一块专门用于打草稿的“白板”。
这就是GAM的`**事件演进图**（**Event Progression Graph**）`，用 $\mathcal{G}_{\text{event}}$ 表示。
在对话进行时，系统处于情景缓冲状态。
所有的高频交互和琐碎细节，都会作为原子事件单元 $e_t$ 被记录在这块白板上。
由于这块白板与核心档案库是物理隔离的。
瞬时的对话噪音被严格限制在局部作用域内。
这确保了全局长期记忆网络不会受到任何未经验证的污染。

**2. 核心档案库：语义巩固状态**
那么，系统何时才会清理白板并将信息归档呢？
GAM创新性地引入了基于状态的记忆巩固机制。
它不依赖生硬的固定字数或时间步触发更新。
而是通过大模型作为神经代理，动态检测语义边界。
当系统检测到一个二值触发器 $b_t$ 等于1时，代表语义距离超过了稳定阈值 $\epsilon$。
这通常意味着用户已经结束了当前的话题。

此时，系统便会触发合并操作，进入语义巩固状态。
它会将白板上的局部图转化为一个具备双重粒度的巩固语义节点 $v_{\text{new}}$。
随后，这些被验证过的完整语义单元，才会被写入全局图谱。
即被存入`**主题关联网络**（**Topic Associative Network**）` $\mathcal{G}_{\text{topic}}$ 中。
完成归档后，白板会被清空，准备迎接下一个话题的演算。

<img src="/images/2604.12285v1/x2.webp" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

### 图引导的多因子检索策略

将记忆分层存储固然保证了稳定性，但如何在推理时精准调用？
如果单纯依赖传统的扁平向量检索，很容易丢失上下文脉络。
为此，本文设计了一种自上而下的多因子图引导检索策略。

首先，系统在全局的 $\mathcal{G}_{\text{topic}}$ 中进行语义锚定。
通过向量相似度找到最相关的核心主题节点。
接着，利用图结构展开“结构化下钻”。
通过跨层边 $\mathcal{E}_{\text{cross}}$ 直接穿透到那些已被归档的历史事件图中。
从原始档案 $\mathcal{S}_{\text{arch}}$ 中提取最底层的生动对话细节。

为了在海量候选集中找到最准确的上下文，系统应用了乘法调制重排序。


{% raw %}$$ \text{Score}(v,q)=P_{\text{sem}}(v|q)\cdot\prod_{k\in\mathcal{K}}\beta_{k}^{\mathbb{I}_{k}(v,q)} $${% endraw %}


该公式首先使用交叉编码器计算基础的语义概率 $P_{\text{sem}}$。
随后引入时间、角色和置信度等外部结构化信号进行调节。
这种机制确保了提取的记忆不仅在主题上高度契合。
在时序连贯性和逻辑上也保持着绝对的精准。

### 关键实验评估

研究团队在LoCoMo和LongDialQA两大长程交互基准上进行了详尽评估。
实验结果证明，GAM在推理准确率和运行效率上均实现了显著飞跃。

**复杂场景下的降维打击**
在LongDialQA这种多方长程对话的对抗性测试中，GAM表现出了极强的韧性。
当搭载较小规模的Qwen 2.5-7B模型时。
GAM依然取得了12.55的平均F1分数，超越MemoryOS高达86%。
这强有力地说明，优秀的结构化隔离记忆能极大弥补小模型上下文窗口的先天劣势。
这种通过架构优化而非单纯增加参数量带来的提升，对端侧部署意义重大。

**主题分割策略的降维对比**
为了验证语义边界触发的必要性，研究分析了不同的切分策略。
如图3所示，固定窗口（256 Token）的硬切分表现垫底（F1为34.23）。

<img src="/images/2604.12285v1/x3.webp" alt="Refer to caption" style="width:80%; max-width:300px; margin:auto; display:block;">

这是因为生硬的字数截断会无情割裂上下文，导致逻辑链条断裂。
而GAM的动态语义触发（平均F1达40.00）则完美保留了推理所需的连贯结构。
尤其是在时序推理任务上，性能提升更为显著。

**更低的算力消耗**
在效率方面，GAM由于避免了全局知识图谱的频繁重构，显著降低了开销。
平均每次查询的Token消耗仅为1370，比工业界标杆Mem0降低了11%。
在保持同等推理延迟的前提下，成功兑现了更优的性能。

### 局限性与工程启示

尽管GAM在文本模态下表现优异，但本文也坦诚了目前的局限性。
当前的系统尚无法处理现实交流中极为丰富的视觉或听觉线索。
例如，用户表情的变化或语气的起伏，同样标志着重要的话题转换。
未来向原生多模态记忆框架演进，是进一步提升Agent感知能力的必经之路。

从工程实践的角度来看，GAM的解耦架构带来了极具价值的启示。
对于需要在企业微信或智能客服系统中落地的Agent而言。
这种“先缓存，后巩固”的机制天然充当了极佳的隐私过滤器。
在未经验证的内容正式写入长期全局图谱之前。
开发者有充足的缓冲时机，利用规则引擎拦截敏感或有害数据。
此外，基于图结构的记忆允许在不重新微调大模型参数的情况下，直接删改错误的信念节点。
这为构建更安全、更可控的下一代AI应用提供了极其关键的架构参考。
