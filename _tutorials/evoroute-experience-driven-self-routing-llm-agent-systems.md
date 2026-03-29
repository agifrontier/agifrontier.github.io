---
layout: default
title: "EvoRoute: Experience-Driven Self-Routing LLM Agent Systems"
---

## 破解Agent“不可能三角”：EvoRoute实现成本降80%、速度提升3倍

<img src="/images/2601.02695v1/A__title.jpg" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

当我们在惊叹于 AgentOrchestra 或 Devin 等顶尖 AI Agent 处理复杂任务的能力时，往往忽略了一个尴尬的现实：这些系统正在疯狂地“烧钱”。

> ArXiv URL：http://arxiv.org/abs/2601.02695v1

为了追求极致的性能，现有的 Agent 系统通常会无脑调用最昂贵的模型（如 GPT-4 或 Claude-3.5-Sonnet）。这就导致了一个严峻的问题：虽然任务完成了，但单次执行成本可能高达数美元，且等待时间极长。这便是本文要探讨的核心痛点——**代理系统不可能三角**（**Agent System Trilemma**）。

如何在**性能**（Performance）、**成本**（Cost）和**效率**（Efficiency）这三者之间找到完美的平衡点？来自新加坡国立大学和通义实验室的研究团队提出了一种全新的解决方案——**EvoRoute**。这是一种基于经验驱动的自进化模型路由机制，它能够在保持甚至提升系统性能的同时，将执行成本降低高达 80%，并将延迟降低超过 70%。

### 什么是“代理系统不可能三角”？

在经济学中，有著名的“蒙代尔不可能三角”。研究团队敏锐地发现，在复杂的 Agent 系统中也存在类似的困境。

<img src="/images/2601.02695v1/x1.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

如上图所示，现有的 Agent 系统往往顾此失彼：

*   **高性能但昂贵**：为了保证准确率，全程使用 SOTA 模型，导致成本和延迟飙升。

*   **便宜但低能**：为了省钱使用小模型，结果任务频频失败。

传统的解决方法是“静态路由”，即人为规定“规划用 GPT-4，写代码用 GPT-3.5”。但这种方法太死板，无法应对动态变化的复杂任务。EvoRoute 的出现，正是为了打破这一僵局。

### EvoRoute：让 Agent 学会“看菜下碟”

EvoRoute 的核心理念不再是简单的“大模型 vs 小模型”，而是**基于过往经验的动态选择**。它不像传统路由那样只针对整个任务，而是深入到 Agent 工作流的每一个**子步骤**（Sub-task）进行微操。

EvoRoute 的工作流程可以概括为三个关键阶段：

#### 1. 多维度检索（Multi-Faceted Retrieval）

当 Agent 面临一个新的子任务时，EvoRoute 首先会问：“我们以前遇到过类似的情况吗？”

它利用一个不断增长的**经验库**（**Experience Base, $\mathcal{K}$**），通过三个维度来检索历史记录：

*   **Agent 角色**：当前是哪个 Agent 在工作？（如 Coder, Planner）

*   **语义相似度**：当前的任务指令和以前哪次最像？

*   **工具使用**：当前任务可能需要用到什么工具？

通过这种“联想记忆”，EvoRoute 召回了一批历史案例，这些案例记录了当时使用了什么模型、花了多少钱、耗了多少时间以及最终的效果如何。

#### 2. 帕累托最优过滤（Pareto-Optimal Filtration）

检索到的候选模型可能五花八门。EvoRoute 接下来会进行一轮残酷的淘汰赛。它使用**帕累托最优**（**Pareto-Optimality**）原则：如果模型 A 在成本、速度和性能上都比模型 B 差，或者在某一方面持平但在其他方面更差，那么模型 B 就被视为“被支配”的劣质选项，直接剔除。

这一步确保了留下的候选模型都是“各有千秋”的精英——要么极快，要么极准，要么极省。

#### 3. 基于效用的选择（Selection）

最后，系统需要从精英中选出唯一的“天选之子”。EvoRoute 使用了一个轻量级的决策模型，基于贝叶斯推断来估计每个候选模型的预期表现。其核心是一个效用函数 $U^{\prime}(l)$：




{% raw %}$$ U^{\prime}(l)=w\_{p}\cdot\tilde{x}\_{\mathbb{P},l}-w\_{c}\cdot\tilde{x}\_{\mathbb{C},l}-w\_{d}\cdot\tilde{x}\_{\mathbb{D},l} $${% endraw %}



其中 $\tilde{x}\_{\mathbb{P},l}$、$\tilde{x}\_{\mathbb{C},l}$、$\tilde{x}\_{\mathbb{D},l}$ 分别代表模型 $l$ 在性能、成本和延迟上的预估值，$w$ 为权重。系统会选择得分最高的模型 $l^{\*}$ 来执行当前步骤。

<img src="/images/2601.02695v1/x2.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

最精妙的是，EvoRoute 是**自进化**的。一旦选定的模型完成了任务，这次执行的真实结果（成功与否、耗时、花费）会被立即写回经验库 $\mathcal{K}$。这意味着，EvoRoute 用得越多，它就越聪明，越能精准地避开那些“又贵又笨”的坑。

### 实验结果：降本增效的实锤

研究团队在 GAIA、BrowseComp+ 等高难度 Agent Benchmark 上进行了广泛测试，对比了 ReAct、Smolagents 和 Cognitive Kernel-Pro (CK-Pro) 等主流框架。

**1. 惊人的成本与速度优化**

在 GAIA 基准测试中，当将 EvoRoute 集成到 CK-Pro 框架中时：

*   **成本降低**：相比全程使用 Claude-4，成本从 $359.32 降至 $85.40，降幅超过 **76%**。

*   **性能提升**：准确率反而从 58.28% 提升到了 **63.18%**。

*   **速度飞跃**：在 BrowseComp+ 测试中，相比 Claude-4，执行时间缩短了近 **50%**。

**2. 真正的“好钢用在刀刃上”**

通过可视化分析（如下图），我们可以清晰地看到 EvoRoute 的策略：

*   对于复杂的**规划代理**（Plan Agent），它倾向于调用 Gemini-1.5-Pro 或 Claude-3.5 等高智商模型。

*   对于简单的**文件处理代理**（File Agent），它则大量使用廉价的 Qwen-14B。

<img src="/images/2601.02695v1/x4.jpg" alt="Refer to caption" style="width:85%; max-width:450px; margin:auto; display:block;">

这种分配策略完美诠释了“该省省，该花花”的原则。

### 总结

EvoRoute 的出现证明了 AI Agent 系统不必在性能和成本之间做非此即彼的选择。通过精细化的、基于经验的动态路由，我们完全可以用 20% 的成本实现 SOTA 级别的性能。

对于正在构建复杂 Agent 应用的开发者来说，EvoRoute 提供了一个极具价值的启示：**不要盲目迷信单一的大模型，构建一个懂得从历史经验中学习、能灵活调度资源的“大脑”，才是通往高效 AI 的关键。**