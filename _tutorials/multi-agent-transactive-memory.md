---
layout: default
title: "突破智能体孤岛：MATM共享Agent轨迹记忆，任务成功率提升17%"
description: "当前的大语言模型智能体在执行各类复杂任务时，会产生大量高价值的中间轨迹。然而，这些包含丰富过程知识的交互数据，通常在单次使用后就被直接丢弃。或者仅被生产该轨迹的单一智能体私有保留。这种“阅后即焚”的孤岛模式，导致新实例化的智能体被迫反复重新探索已存在的解决方案。"
arxiv_id: "2606.19911"
topics:
  - "AI Agent"
  - "RAG"
tags:
  - "ALFWorld"
  - "LLM agents"
  - "MATM"
  - "RAG"
  - "WebArena"
  - "agent trajectories"
related_tutorials:
  - "retrieval-augmented-generation-rag-for-fintech-agentic-design-and-evaluation"
  - "remember-me-refine-me-a-dynamic-procedural-memory-framework-for-experience-drive"
  - "thought-retriever-don-t-just-retrieve-raw-data-retrieve-thoughts-for-memory-augmented-agentic-sy"
  - "dynamic-agent-skills-a-lifecycle-survey-and-taxonomy-of-evolving-skill-libraries"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">Multi-Agent Transactive Memory</p>

当前的大语言模型智能体在执行各类复杂任务时，会产生大量高价值的中间轨迹。

> **ArXiv URL**：http://arxiv.org/abs/2606.19911v1

然而，这些包含丰富过程知识的交互数据，通常在单次使用后就被直接丢弃。

或者仅被生产该轨迹的单一智能体私有保留。

这种“阅后即焚”的孤岛模式，导致新实例化的智能体被迫反复重新探索已存在的解决方案。

这不仅极大浪费了推理算力，也限制了 Agent 生态在种群级别的能力进化。

为了打破这种经验隔离，卡内基梅隆大学与加州大学伯克利分校的研究人员提出了全新框架。

该框架名为 **多智能体交互记忆**（**Multi-Agent Transactive Memory, MATM**）。

该研究将 Agent 产生的轨迹转化为整个智能体种群共享的知识库。

在无需联合训练的前提下，显著提升了下游任务的成功率并降低了交互步数。

### 交互记忆机制

在传统搜索系统中，人类通过搜索引擎检索其他人类编写的文档。

在 **检索增强生成**（**Retrieval-Augmented Generation, RAG**）中，Agent 检索人类生成的语料。

而 MATM 迈出了关键的下一步：让 Agent 检索其他 Agent 生成的交互轨迹。

<img src="/images/2606.19911v1/x1.webp" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

为了理解 MATM 的运作机制，我们可以引入一个贯穿始终的“共享航海日志”比喻。

在未知的海域（复杂任务环境）中航行时，不同的船只（Agent）会遇到各种洋流和暗礁。

传统的做法是，每艘船都靠自己的声纳盲目试探，前船走通的航线，后船一无所知。

MATM 就像是为所有船只建立了一个中央同步的“共享航海日志”。

这个日志分为两个核心角色：**生产者智能体**（**Producer Agents**）和 **消费者智能体**（**Consumer Agents**）。

生产者在成功完成任务后，将其走过的正确航线（动作-观察轨迹）上传到日志库。

当消费者在新的航程中遇到特定的坐标状态时，就可以从日志中调取前人的航线指导。

这两个角色并非互斥，一个 Agent 在某任务中是生产者，在另一任务中可能就是消费者。

### 状态条件索引与检索

不同于人类编写的静态文档，Agent 的交互轨迹具有极强的上下文依赖性。

直接使用传统的语义检索，往往无法精准匹配 Agent 当前面临的具体困境。

为此，MATM 采用了状态条件下的键值对（Key-Value）索引方案。

在这个方案中，Agent 最近的交互历史被作为“检索键”（Key）。

而紧随其后的交互片段（即前人接下来采取的动作和观察）被作为“存储值”（Value）。

回到“共享航海日志”的比喻，这意味着日志不是按任务名称分类的。

而是按照“当前所处的经纬度和风向”（最近的交互历史）来建立索引。

当消费者 Agent 检索时，系统会匹配出与当前处境最相似的历史记录。

从而直接提供下一步应该如何操作的具体指导，而不仅仅是初始任务维度的模糊提示。

### 轨迹排序学习机制

仅靠基础的向量嵌入模型 $f$ 进行相似度检索，往往不足以选出真正对下游任务有帮助的轨迹。

为此，研究团队引入了 **轨迹排序学习**（**Learning To Rank Trajectories, LTRT**）级联管道。

LTRT 管道分为两个阶段：候选生成和特征排序。

首先通过基础检索召回前 20 个候选轨迹块。

随后，一个轻量级的重排模型 $g_{\theta}$ 会对这些候选者进行重新打分并选出 Top-1。

关键在于，重排模型的训练标签并非基于语义相似度。

而是基于轨迹块的“边际效用”（Marginal Utility）。

具体而言，标签 $\ell$ 被定义为 $\ell=s_{t}^{(j)}-s_{\mathrm{base}}$。

即注入该轨迹块后，消费者 Agent 完成任务的表现与完全不使用检索时的基线表现之差。

这种设计确保了系统推荐的是“真正能提高任务成功率”的航线，而不仅是“看起来相似”的航线。

### 核心实验与指标表现

研究人员在两个高度交互的环境中实例化了 MATM：

一个是基于文本的家庭任务环境 ALFWorld。

另一个是基于网页导航的复杂任务环境 WebArena。

为了全面衡量模型表现，除了常规的 **成功率**（**Success Rate, SR**）和 **交互步数**（**# steps**）。

研究还引入了 **回报配对偏好**（**Return-Paired Preference, RPP**）指标。

该指标用于联合捕捉有效性（成功率）和效率（步数）的帕累托优势。

在 ALFWorld 中，单阶段的纯密集检索就已展现出明显优势。

成功率从无检索基线的 47% 跃升至 55%，平均交互步数从 11.77 降至 11.18。

而在加入了 LTRT 重排机制（特别是使用 SVMRank 模型）后，效果进一步爆发。

成功率飙升至 64.3%，相较于无检索基线提升了惊人的 17.2 个百分点。

RPP 指标也达到了 0.15，证明检索增强后的种群在效率和成功率上全面压制了基线。

在难度更高的 WebArena 环境中，虽然整体基线较低，但 MATM 依然有效。

使用前馈神经网络（FFN）作为重排器时，成功率从 18% 提升至 20.5%。

同时实现了最低的步数消耗（19.91步）和最高的 RPP 分数（0.04）。

### 跨任务泛化与规模缩放法则

MATM 的另一个核心亮点在于其强大的泛化能力和规模效应。

研究团队对候选轨迹池进行了严格的消融实验。

结果表明，即使强制限制检索池，仅允许跨任务类型的轨迹匹配。

ALFWorld 的成功率依然能达到 59.9%，远高于 47.1% 的基线。

这说明结构不同但底层逻辑相似的轨迹，依然蕴含着高度可迁移的程序性知识。

此外，系统的表现与记忆库的规模密切相关。

<img src="/images/2606.19911v1/x3.webp" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">
<img src="/images/2606.19911v1/x4.webp" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

如上图所示，在 ALFWorld 中，成功率和效率随着索引规模的扩大呈现单调递增。

这证实了记忆库越庞大，Agent 种群受益越深。

而在 WebArena 中，成功率在 50% 索引规模时出现了一个有趣的非单调下降。

但当索引规模达到 100% 时，成功率迅速反弹至最高点（20.9%）。

研究认为，这是由于中等规模的记忆库足以召回看似合理但实际无用的“误导性航线”。

只有当规模足够大、多样性足够丰富时，系统才能稳定覆盖并输出高质量的匹配结果。

### 局限与工程启示

尽管 MATM 展现了多智能体经验共享的巨大潜力，但该框架仍处于早期探索阶段。

当前的 LTRT 数据集在排位采样上较为稀疏（仅采样了 1, 5, 10, 15, 20 位）。

这在一定程度上限制了标签分布的完整性。

此外，当前的评估仅关注了消费者 Agent 的福祉。

在工程实践中，MATM 本质上是一个双边市场。

如何建立合理的归因机制和激励机制，鼓励生产者 Agent 持续贡献高质量轨迹。

以及如何防御恶意生产者注入的“投毒轨迹”，将是未来走向开放生态必须解决的难题。

总体而言，MATM 将工件复用从个体优化层面提升到了种群级的基础设施高度。

这不仅为减少冗余探索指明了方向，也为未来异构 Agent 之间的大规模协作提供了全新的设计模式。
