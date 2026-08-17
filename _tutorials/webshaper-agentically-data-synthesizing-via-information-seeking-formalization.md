---
layout: default
title: "WebShaper：集合论重构数据合成，打造最强开源搜索Agent"
description: "OpenAI的DeepResearch近期引爆了全网，展现了惊人的信息搜索（Information-Seeking,IS）能力。这类Agent系统能自主查阅海量资料，被视为迈向AGI的关键一步。然而，开源社区想要复现这类惊艳的系统，却常常卡在一个致命瓶颈上：高质量训练数据的极度匮乏。"
arxiv_id: "2507.15061"
topics:
  - "AI Agent"
  - "基础模型"
tags:
  - "Agentic Expander"
  - "GAIA Benchmark"
  - "IS Data Synthesis"
  - "Information-Seeking"
  - "Knowledge Projections"
  - "Large Language Model Agents"
related_tutorials:
  - "are-agents-just-automata-on-the-formal-equivalence-between-agentic-ai-and-the-ch"
  - "training-task-reasoning-llm-agents-for-multi-turn-task-planning-via-single-turn-"
  - "personal-llm-agents-insights-and-survey-about-the-capability-efficiency-and-security"
  - "a-language-for-describing-agentic-llm-contexts"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">WebShaper: Agentically Data Synthesizing via Information-Seeking Formalization</p>

OpenAI的Deep Research近期引爆了全网，展现了惊人的**信息搜索**（**Information-Seeking, IS**）能力。

> **ArXiv URL**：http://arxiv.org/abs/2507.15061v1

<img src="/images/2507.15061v1/p01_intro_page_1.jpg" alt="论文中的核心图示" style="width:85%; max-width:600px; margin:auto; display:block;">

*论文原图：用于辅助理解核心方法或实验结果。*

这类Agent系统能自主查阅海量资料，被视为迈向AGI的关键一步。

然而，开源社区想要复现这类惊艳的系统，却常常卡在一个致命瓶颈上：

高质量训练数据的极度匮乏。

纯靠人工标注复杂搜索轨迹不仅成本高昂，且难以规模化。

因此，通过大模型进行数据合成，成为了当前业界的主流探索方向。

传统的合成方法大多采用“信息驱动”范式。

系统会先去网上漫无目的地爬取网页信息，再让大模型根据这些素材反向“捏造”问题。

这种做法导致了严重的隐患：拼凑出的推理链路往往自相矛盾，且充斥着同质化的冗余数据。

为了打破这一僵局，阿里通义实验室提出了一种全新范式——**WebShaper**。

该研究彻底抛弃了先找材料再定框架的传统做法，转而采用一种严谨的“形式化驱动”。

本文将深入拆解WebShaper背后的核心技术，看看数学之美如何重塑Agent的认知基石。

### 核心痛点：为什么“信息驱动”行不通？

在深入新方法前，我们需要理解现有方法的缺陷。

诸如WebDancer或TaskCraft等现有的合成框架，主要依赖信息的线性拼接。

这就像是去荒野里随意捡砖头，捡到什么形状的砖，就勉强拼凑成什么样子的房子。

大模型在处理这些杂乱无章的原始网页时，极易产生理解偏差。

这就导致最终生成的自然语言问题，其内在逻辑与实际的答案根本对不上。

此外，无序的信息检索会导致系统反复收集极其相似的结构，严重限制了数据的多样性。

### 核心机制：集合论化身“建筑图纸”

WebShaper破局的关键，在于引入了**任务形式化**（**Task Formalization**）。

系统不再盲目搜集数据，而是先用严密的数学逻辑画好**“标准建筑图纸”**。

图纸明确规定了哪里需要承重墙，Agent就专门去寻找符合规格的知识素材。

这份图纸所使用的语言并非自然语言，而是严谨的**集合论**（**Set Theory**）。

该研究将复杂的搜索任务解构为统一的基础单元，称为**知识投影**（**Knowledge Projections, KP**）。

KP就是这份图纸上的**“标准预制件”**。

任何复杂的问题，都可以被拆解为实体集合之间的关系映射。

例如这样一个问题：
“2004-05赛季某支球队中，哪位球员是90后？这支球队成立于1966年，是一支东德足球队。”

在自然语言下，这显得错综复杂。但在WebShaper中，它被精准地表达为一个目标实体集 $T$。




{% raw %}$$T = R_{playIn}(T_1) \cap (R_{playAt}(\{2004\}) \cup R_{playAt}(\{2005\})) \cap \bigcup_{1900}^{1999} R_{bornIn}(\{y\}) $${% endraw %}



其中基础实体集 $T_1$ 的定义同样清晰：




{% raw %}$$T_1 = R_{foundIn}(\{1996\}) \cap R_{isA}(\{East German football team\})$${% endraw %}



images/page_3_Figure_0.jpg

通过对知识投影进行交集和并集的组合，系统就像搭积木一样，精确控制着推理的复杂度。

这种数学级别的严密性，从根本上消除了数据生成中“问题与答案不匹配”的顽疾。

### 数据合成流水线：从种子到深林

有了这套严密的数学框架，WebShaper设计了一套精巧的多步数据合成流水线。

整个过程无需人工干预，完全由Agent自主驱动。

#### 第一步：种子任务构建

万丈高楼平地起，合成的第一步是获取大量高质量的初始问题。

该研究利用离线的维基百科数据库，通过超链接进行随机游走。

基于这些关联文章，模型生成了完全以真实文本为支撑的种子问题 $q^1(T)$。

这一步确保了数据的根基是真实可信的，绝不凭空捏造。

#### 第二步：智能体扩展（Agentic Expansion）

这是WebShaper最精彩的环节。

为了让基础问题演化为深度搜索任务，研究引入了一个专属的扩展智能体（Expander）。

由于复杂的递归数学公式对大模型并不友好，研究团队设计了**KP表示法**（**KP Representation**）。

通过引入“常量”与“变量”的概念，将复杂的集合运算转化为清晰的三元组列表。

Expander能够完美读懂这些三元组要求，并被赋予了上网检索和验证工具。

更为关键的是，Expander采用了一种**逐层扩展策略**（**Layer-wise Expansion**）。

在每一次扩展中，Expander会挑选逻辑图纸上的一个叶子节点（常量节点）。

将其替换为一个新的变量节点，并连接出新的事实分支，从而生成更复杂的 $q^{n+1}(T)$。

这种像植物根系一样逐层延展的方式，完美避开了“逻辑短路”和无意义的冗余推理。

#### 第三步：轨迹构建与强化训练

获取高质量问题后，还需要收集Agent解决这些问题的完整思考过程。

该研究部署了一个基于QwQ模型的ReAct架构Agent。

Agent通过不断输出思考和动作，最终生成了数千条高质量的搜索轨迹。

随后，这些数据被用于**监督微调**（**Supervised Fine-Tuning, SFT**）。

但这还不算完，研究进一步引入了**群体相对策略优化**（**GRPO**）进行强化学习。




{% raw %}$$ \mathcal{J}_{\text{GRPO}}(\theta) = \mathbb{E} \left[ \frac{1}{\sum_{i=1}^G |y_i|} \sum_{i=1}^G \sum_{t=1}^{|y_i|} \min(...) \right] $${% endraw %}



WebShaper构造的高难度任务，极大激发了模型在强化学习阶段的动态决策能力。

### 实验验证：刷新开源性能天花板

WebShaper合成的数据集，在真实模型的训练中展现了压倒性的优势。

研究团队在极具挑战性的GAIA和WebWalkerQA基准上进行了严格测试。

images/page_0_Figure_9.jpg

上图直观展示了当今顶尖搜索Agent的实力梯队。

在使用Qwen-2.5-72B作为基座模型时，WebShaper取得了极其耀眼的成绩。

它不仅在两个榜单上横扫了所有开源方法（如WebDancer、WebThinker）。

更是目前唯一一个在此类评测中突破60分大关的开源方案。

这一成绩已经非常逼近OpenAI闭源的Deep Research系统。

实验数据充分证明，高质量的搜索数据能够深度激活大模型的长链路探索潜能。

### 深度剖析：机制为什么有效？

为了验证各模块的实际贡献，该研究进行了详尽的消融实验。

**形式化 vs 自然语言**：
如果仅仅依靠自然语言去指导Expander进行扩展，效果会怎样？

images/page_12_Figure_0.jpg

结果如上图左侧所示，无论是32B还是72B模型，基于形式化框架训练的模型性能均大幅领先。

数学框架有效减少了合成过程中的误差传递，保证了问答对的精准一致。

**逐层扩展 vs 顺序扩展**：
上图右侧则验证了扩展策略的有效性。

相比于简单粗暴地将条件首尾相接（顺序扩展），逐层扩展能构建出更立体的逻辑树。

这迫使Agent在搜索时进行真正的多跳推理，而非依赖简单的关键字捷径。

此外，对模型调用工具的统计分析显示，WebShaper训练出的模型明显具备更深度的工具使用习惯。

它能极其从容地处理复杂、冗长的工具调用链，展现出极强的任务拆解韧性。

### 局限性与工程启示

尽管WebShaper交出了一份惊艳的答卷，但这种全新范式也并非完美无缺。

首先，基于集合论的形式化框架高度依赖于实体和明确的客观关系。

对于一些缺乏明确边界的探索性诉求，如何将其转化为严密的KP表达式，仍是一大挑战。

其次，当前的扩展过程重度依赖在线检索的质量。

若互联网上存在事实冲突的信息，可能会在合成初期引入难以察觉的系统性偏差。

**对于开发者的启示**：

WebShaper的成功，给狂热的Agent赛道降了一丝清醒的温度。

当大家都在追求用“魔法打败魔法”，即用更强的大模型去直接生成数据时。

回归经典计算机科学，引入形式化验证与符号逻辑，或许才是提升上限的王道。

将自然语言的模糊性进行“数学降维”，为模型提供结构绝对正确的高优供给。

这不仅是构建下一代Deep Research的必由之路，更是通往可靠AGI的一把关键钥匙。
