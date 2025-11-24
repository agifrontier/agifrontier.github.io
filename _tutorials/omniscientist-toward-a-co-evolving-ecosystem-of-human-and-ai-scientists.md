---
layout: default
title: "OmniScientist: Toward a Co-evolving Ecosystem of Human and AI Scientists"
---

# OmniScientist：让AI科学家“组建团队”，成果超越NIPS最佳论文

<img src="/images/2511.16931v1/A__title.jpg" alt="" style="width:85%; max-width:600px; margin:auto; display:block;">

当前的AI Agent能写代码、做分析，甚至开始涉足科学研究，人们称之为“AI科学家”。但它们更像是孤军奋战的天才，缺乏一个关键要素：科学研究本质上是社会性与协作性的活动。

> **论文标题**：OmniScientist: Toward a Co-evolving Ecosystem of Human and AI Scientists

> **ArXiv URL**：http://arxiv.org/abs/2511.16931v1

如果AI科学家也能像人类一样，组建研究团队、进行同行评审、在庞大的知识网络中寻找灵感，并与人类科学家无缝协作，科学发现的范式是否会被彻底颠覆？

来自清华大学等机构的研究者们推出了**OmniScientist**框架，它不再将AI视为孤立的工具，而是构建了一个完整的、模拟人类科研体系的AI科学家“生态系统”。更惊人的是，它在一个真实案例中，其发现的解决方案性能超越了NIPS 2024的最佳论文方法！

<img src="/images/2511.16931v1/x1.jpg" alt="OmniScientist系统概览" style="width:85%; max-width:450px; margin:auto; display:block;">

### AI科学家的“社交困境”

现有的“AI科学家”系统，尽管功能强大，但大多将科学发现简化为一个孤立的搜索或优化问题。

它们缺少支撑人类科学发展的复杂基础设施：

*   **协作机制**：如何让多个AI Agent或人机团队有效合作？

*   **贡献归属**：谁的想法？谁的功劳？如何清晰界定？

*   **同行评议**：如何保证研究的严谨性和可靠性？

*   **知识网络**：如何理解思想的传承和概念的演变？

正是这些缺失，让AI科学家们始终是“局外人”，难以真正融入并推动人类科学共同体的演进。

### OmniScientist：模拟完整的科研生态

OmniScientist的核心思想，是把人类科研的基础设施“编码”到AI的工作流中。它不仅实现了从文献综述、研究构思、实验自动化到论文写作和同行评审的全流程自动化。

更重要的是，它提供了三大 infrastructural support，构建了一个微缩版的科学世界。

#### 1. 结构化知识系统：读懂学术传承

OmniScientist的基础是一个动态知识库，它不仅仅是论文的堆砌。

它基于引文网络和概念关联，构建了一个结构化的知识图谱。这使得AI Agent能够理解一篇论文的“学术地位”——它继承了谁的思想，又启发了哪些后续研究。

这种基于“关系”的检索，远比传统的关键词搜索更深刻、更精准。

<img src="/images/2511.16931v1/pipeline.jpg" alt="知识库构建流程" style="width:85%; max-width:600px; margin:auto; display:block;">

#### 2. Omni科学协议（OSP）：AI与人的协作规则

为了解决协作问题，该研究提出了**Omni科学协议**（**Omni Scientific Protocol, OSP**）。

OSP是一个专为科研场景设计的协作“骨架”，它有几个革命性的特点：

*   **人类成为参与者**：在OSP中，人类不再是外部操作员，而是系统内的最高决策实体。人的每个决策都是可追溯的协议事件。

*   **中心化Hub管理**：通过一个中心Hub，OSP能够管理“多对多”的复杂协作关系，例如一个项目包含多个人类科学家和多个AI Agent。

*   **贡献溯源**：最关键的一点，OSP实现了从数据溯源到贡献溯源。每个研究对象（如一个假设）都绑定了一个“贡献账本”，清晰记录了每个想法、数据集或实验结果的来源，无论是来自人类还是AI。

这套机制为透明、可信的人机协作奠定了基础。

#### 3. ScienceArena：AI科学界的“Elo排行榜”

如何评价一个开放性科学发现的优劣？这是一个难题。

OmniScientist为此推出了一个开放评估平台**ScienceArena**。它模拟了科学界的同行评议机制，采用盲审的“两两比较”投票方式，由人类专家对匿名的研究成果进行投票。

平台会根据投票结果动态计算每个“AI科学家”的Elo等级分，形成一个实时更新的排行榜。这使得AI Agent的进化方向能够被人类科学界的共识所引导。

### 实践出真知：闭环系统超越NIPS最佳论文

理论再好，不如实战。OmniScientist通过一个闭环多Agent系统，整合了文献研究、思想产生和自动化实验等模块，形成了一个强大的“AI研究小组”。

<img src="/images/2511.16931v1/x12.jpg" alt="闭环多Agent系统" style="width:90%; max-width:700px; margin:auto; display:block;">

研究团队选择了一个极具挑战性的任务：改进**随机泰勒微分估计器**（**STDE**），这是NIPS 2024的一篇最佳论文。STDE方法虽然强大，但其精度受限于蒙特卡洛（MC）采样带来的高方差。

*   **对照组（AlphaEvolve）**：作为一个强大的进化算法系统，它对STDE进行了深度优化，但只能在原有框架内进行微调，改进效果有限。

*   **实验组（OmniScientist）**：OmniScientist的文献综述Agent在知识库中检索时，发现“拟蒙特卡洛（Quasi-Monte Carlo）采样”是解决MC方差问题的经典外部知识。Ideation Agent据此提出将其与STDE结合的全新思路。

结果令人瞩目！

<img src="/images/2511.16931v1/x13.jpg" alt="实验结果对比" style="width:80%; max-width:300px; margin:auto; display:block;">

如上图所示，引入了外部知识的OmniScientist方案，在所有维度上都实现了解决方案误差的“巨大且一致的降低”，性能显著超越了原始的STDE方法。

这个案例完美证明了OmniScientist的价值：它不只是一个优化器，更是一个能够通过广泛学习和整合外部知识来实现概念性突破的创新引擎。

### 总结

OmniScientist标志着一个范式的转变：从设计孤立的AI研究工具，转向构建一个全面的、可进化的AI与人类共生的科学研究生态系统。

通过将人类科学界的协作规范、评价体系和知识结构融入AI，OmniScientist让AI Agent从单纯的任务执行者， evolve 成为能够理解科研范式、参与协作、并与人类科学家共同推动知识边界的真正“科学家”。

一个AI与人类科学家共同演化、加速创新的新时代，或许已经拉开序幕。