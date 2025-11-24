---
layout: default
title: "Budget-Aware Tool-Use Enables Effective Agent Scaling"
---

# AI Agent只会“挥霍”算力？谷歌BATS框架教它精打细算，成本性能双优化

<img src="/images/2511.17006v1/A__title.jpg" alt="" style="width:85%; max-width:450px; margin:auto; display:block;">

当今的AI Agent越来越强大，我们习惯性地认为：给它更多的计算资源、更多的工具调用次数，它就应该表现得更好。但现实果真如此吗？谷歌的最新研究发现了一个反直觉的现象：简单地给Agent增加预算，其性能很快就会触及“天花板”，不再提升。问题出在哪？原来，这些Agent缺少一种关键能力——**预算意识**（budget awareness）。它们就像一个没有预算概念的员工，即使资源充足，也不知道如何深度挖掘或调整策略。

> **论文标题**：Budget-Aware Tool-Use Enables Effective Agent Scaling

> **ArXiv URL**：http://arxiv.org/abs/2511.17006v1

为了解决这个难题，谷歌DeepMind等机构推出了一个全新的智能框架BATS，教会Agent如何“精打细算”，在有限的预算内最大化性能。

### 性能瓶颈：只会“行动”却不懂“规划”的Agent

对于需要与外部环境交互的工具增强型Agent（tool-augmented agents）而言，其能力扩展不仅依赖于内部“思考”（消耗Tokens），更依赖于外部“行动”（调用工具，如网络搜索）。

工具调用的次数，直接决定了Agent探索外部信息的广度和深度。

然而，研究发现，标准的Agent（如基于ReAct框架的Agent）并不能有效利用增加的工具调用预算。它们往往进行浅层搜索，一旦觉得找到了“足够好”的答案或陷入困境，就会提前终止任务，全然不知还有大量资源闲置。

<img src="/images/2511.17006v1/x1.jpg" alt="标准ReAct Agent与BATS框架对比" style="width:90%; max-width:700px; margin:auto; display:block;">

*图1：预算追踪器（Budget Tracker）可应用于标准ReAct Agent（上）和更先进的BATS框架（下）。蓝色框表示根据预算进行调整的模块。*

这就引出了一个核心问题：如何让Agent在给定的资源预算下，实现最有效的性能扩展？

### 破局第一步：轻量级“预算追踪器”

研究团队首先提出了一个简单却极其有效的解决方案：**预算追踪器**（**Budget Tracker**）。

这是一个即插即用的轻量级模块，它在Agent的每一步行动后，都会通过Prompt明确告知Agent：“你还剩下多少次工具调用机会”。

<img src="/images/2511.17006v1/x2.jpg" alt="预算追踪器工作原理" style="width:90%; max-width:700px; margin:auto; display:block;">

*图2：在每一轮交互中，Agent在生成下一步思考和工具调用前，都会通过预算追踪器获知当前和剩余的预算。*

别小看这个简单的提醒！它让Agent对资源消耗和剩余预算有了明确感知，从而能够调整后续的推理和行动策略。

实验结果证明了它的威力。如下图所示，在没有预算意识时，标准ReAct Agent的性能在预算达到100后就饱和了。而加入了预算追踪器后，Agent能够持续利用增加的预算，性能也随之稳步提升，成功打破了性能天花板。

<img src="/images/2511.17006v1/x3.jpg" alt="预算追踪器打破性能瓶颈" style="width:85%; max-width:600px; margin:auto; display:block;">

*图3：在BrowseComp数据集上，标准ReAct Agent（蓝色虚线）性能很快饱和，而具备预算意识的Agent（橙色实线）能持续扩展性能。*

### BATS：动态规划与验证的智能框架

在证明了“预算意识”的有效性后，研究团队进一步开发了更先进的**BATS**（**Budget Aware Test-time Scaling**）框架，将预算意识深度融入Agent的整个工作流。

<img src="/images/2511.17006v1/x6.jpg" alt="BATS框架概览" style="width:85%; max-width:600px; margin:auto; display:block;">

*图6：BATS框架概览。Agent从预算感知的思考和规划开始，在迭代中不断根据新信息和预算更新策略。在提出答案后，BATS会进行验证，并根据剩余预算决定是继续、转向还是重新尝试。*

BATS的核心设计原则就是将预算意识贯穿始终，主要体现在两个智能模块中：

1.  **预算感知规划**（**Budget-Aware Planning**）：在任务开始时，BATS会引导Agent分解问题，识别出哪些是用于扩大搜索范围的“探索性”线索，哪些是用于验证具体信息的“验证性”线索。Agent会根据剩余预算，动态地决定是先广泛探索还是直接验证，避免在不确定的路径上过早耗尽资源。

2.  **预算感知自验证**（**Budget-Aware Self-verification**）：当Agent提出一个初步答案后，BATS不会草率结束。验证模块会回溯整个推理过程，检查是否所有问题约束都已满足。更关键的是，它会根据剩余预算做出决策：

    *   如果预算充足且当前路径很有希望，它会决定“**深入挖掘**”（dig deeper）。

    *   如果当前路径似乎走不通，但预算尚有，它会选择“**转换方向**”（pivot），开启新的探索路径。

    *   只有当答案可靠且预算紧张时，它才会确认并输出最终答案。

### 实验效果：更优的成本-性能曲线

为了公平地评估不同方法的效率，该研究提出了一个**统一成本度量**（unified cost metric），它同时考虑了Token消耗和工具调用的成本。




{% raw %}$$ C\_{\textit{unified}}(x;\pi)=\underbrace{c\_{\textit{token}}(x;\pi)}\_{\text{Token Cost}}+\underbrace{\sum\_{i=1}^{K}c\_{i}(x;\pi)\cdot P\_{i}}\_{\text{Total Tool Cost}} $${% endraw %}



在BrowseComp、BrowseComp-ZH和HLE-Search等多个高难度信息检索任务上，BATS的表现十分亮眼。

最值得注意的是，BATS是一个**完全无需额外训练**的框架。仅通过在推理时引入预算感知的智能策略，它就在严格的预算限制下取得了比许多经过专门微调的Agent更好的性能。例如，在使用Gemini-2.5-Pro模型时，BATS在BrowseComp上取得了24.6%的准确率。

下图清晰地展示了BATS在成本-性能权衡上的巨大优势。它推动了成本-性能的**帕累托前沿**（Pareto frontier），意味着在相同的成本下，BATS能达到更高的准确率；或者说，要达到相同的准确率，BATS所需的成本更低。

<img src="/images/2511.17006v1/x7.jpg" alt="BATS实现更优的成本-性能权衡" style="width:85%; max-width:600px; margin:auto; display:block;">

*图7：在统一成本度量下，BATS（橙色）相比基线方法（蓝色）展现出更优越的扩展曲线，实现了更高的性价比。*

### 结论

这项研究首次系统地探讨了预算约束下工具增强型Agent的性能扩展问题。它揭示了“预算意识”是解锁Agent潜力的关键。

从简单的“预算追踪器”到精密的BATS框架，该工作证明了让Agent学会“精打细算”，不仅能打破性能瓶颈，还能显著优化成本效益。这为未来构建更高效、更可靠、更可控的AI Agent系统指明了一个极具前景的方向。