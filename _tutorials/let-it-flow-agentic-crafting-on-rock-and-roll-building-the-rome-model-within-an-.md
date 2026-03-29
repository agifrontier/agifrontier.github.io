---
layout: default
title: "Let It Flow: Agentic Crafting on Rock and Roll, Building the ROME Model within an Open Agentic Learning Ecosystem"
---

## 阿里开源ROME：SWE-bench胜率57%，揭秘打造顶尖Agent的“罗马”基建

<img src="/images/2512.24873v1/A__title.jpg" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

大模型的发展正在经历一场从“对话者”到“行动者”的深刻变革。

> ArXiv URL：http://arxiv.org/abs/2512.24873v1

过去，我们习惯于给模型一个Prompt，然后期待它吐出一个完美的答案。但在真实的软件工程或复杂任务中，这种“一锤子买卖”往往行不通。真正的**智能体构建**（**Agentic Crafting**）需要模型像人类工程师一样：规划方案、编写代码、观察报错、自我修正，并在多次交互中最终解决问题。

然而，开源社区一直缺乏一套像样的“基础设施”来支撑这种复杂的Agent开发。大家都知道“罗马不是一天建成的”（ROME wasn't built in a day），但如何系统性地建造它？

阿里巴巴团队近日发布了一篇重磅论文，不仅推出了名为 **ROME** 的高性能Agent模型，更重要的是，他们开源了背后的整套**智能体学习生态系统**（**Agentic Learning Ecosystem, ALE**）。这套系统在SWE-bench Verified榜单上助力ROME模型达到了 **57.4%** 的准确率，甚至逼近了千亿参数模型的表现。

今天，我们就来拆解一下，阿里是如何在“摇滚”（Rock and Roll）之上，构建起这座通往AGI的“罗马”城的。

### 这里的“摇滚”不仅仅是音乐

论文标题中的 "Rock and Roll" 其实是一个精彩的双关，它代表了ALE生态系统中两个最核心的基础组件：**ROLL** 和 **ROCK**。

要训练一个能在真实环境中干活的Agent，光有数据是不够的，你需要一个能让Agent“摸爬滚打”的训练场，以及一套高效的训练机制。ALE正是为此而生，它包含三个协同工作的组件：

1.  **ROLL**（**Reinforcement Learning Optimization for Large-Scale Learning**）：

    这是一个专为大规模RL设计的训练框架。它的核心亮点在于**动态GPU资源调度**。在Agent训练中，生成数据（Rollout）和更新模型（Training）对资源的需求是波动的。ROLL采用了一种“时分复用”策略，在Rollout需求高峰时全力生成数据，在数据攒够后迅速切换资源进行训练，极大地提高了GPU利用率。

2.  **ROCK**（**Reinforcement Open Construction Kit**）：

    这是Agent的“练功房”——一个安全的沙盒环境管理器。Agent在写代码或执行命令时，可能会产生危险操作（比如意外的rm -rf或网络攻击）。ROCK提供了严格隔离的容器环境，支持文件系统、网络控制等细粒度权限管理，确保Agent在“犯错”时不会炸毁服务器，同时保证了训练数据的纯净和安全。

3.  **iFlow CLI**：

    这是一个连接模型与环境的Agent框架。它负责管理复杂的上下文（Context），让开发者可以通过配置而非硬编码来定义Agent的行为流。

![ALE Ecosystem Overview](images/page_3_Figure_0.jpg)

### ROME：显然是一个Agent模型

基于上述强大的基建，阿里孵化出了 **ROME**（**ROME is Obviously an Agentic ModEl**）。这不仅仅是一个微调后的LLM，它经历了一个精心设计的“三部曲”训练流水线：

1.  **持续预训练（CPT）**：

    在这一阶段，模型不仅学习代码，还通过约3000亿Token的轨迹数据，学习如何像Agent一样思考。这些数据包含了由强力教师模型（如Claude等）生成的成功和失败的交互记录，让ROME学会了“意图形成”和“错误恢复”。

2.  **两阶段监督微调（SFT）**：

    为了避免模型在复杂的Agent任务中迷失，SFT被分为两个阶段。第一阶段使用启发式过滤的数据进行基础训练；第二阶段则引入了**自适应价值数据重访**，专门针对那些高质量、高难度的Agentic任务进行强化。

3.  **强化学习（RL）**：

    这是ROME“灵魂升华”的关键一步。但在长链路的Agent任务中，传统的RL面临巨大挑战：**信用分配难题**。

### 核心算法创新：IPA

在长达数十轮的交互中，Agent可能只在最后一步才成功。如果简单地奖励每一个Token，或者只奖励最后的结果，模型很难知道中间哪一步做对了，哪一步做错了。

为了解决这个问题，论文提出了一种新的策略优化算法：**基于交互感知的策略对齐**（**Interaction-Perceptive Agentic Policy Optimization, IPA**）。

IPA的核心洞察在于：**Agent的决策粒度不是Token，而是“交互块”（Chunk）。**

传统的Token级RL（如PPO或ReMax）往往过于细粒度，导致训练不稳定。IPA将多轮对话建模为 **Chunked MDP**，将每一次完整的“思考-行动-观察”循环视为一个语义单元。




{% raw %}$$ \nabla J_{\text{RL}}(\pi) = \underbrace{\sum_{\tau \in \mathcal{T}^{+}} \dots}_{\text{正样本加权更新}} + \underbrace{\sum_{\tau \in \mathcal{T}^{-}} \dots}_{\text{负样本截断更新}} $${% endraw %}



简单来说，IPA做到了以下几点：

*   **语义级信用分配**：它不是盲目地奖励每一个词，而是评估整个交互动作的价值。

*   **长程稳定性**：通过在语义块级别进行优势函数（Advantage）计算，IPA显著提升了长序列任务的训练稳定性。

*   **正负样本兼顾**：不仅学习成功的轨迹，还利用失败的轨迹（通过重要性采样截断）来明确“什么是不该做的”。

![IPA Algorithm Comparison](images/page_21_Figure_0.jpg)

### 实验结果：小模型的大爆发

在这些技术的加持下，ROME展现出了惊人的战斗力。

在 **SWE-bench Verified**（一个评估LLM解决真实GitHub问题的权威榜单）上，ROME取得了 **57.4%** 的解决率。这个成绩不仅碾压了同等规模的开源模型，甚至可以与参数量大数倍的闭源模型（如GPT-4系列）掰手腕。

此外，阿里还推出了一个新的基准测试 **Terminal Bench Pro**，相比之前的版本，它在规模、领域覆盖和防污染控制上都更加严格。即便在这个“地狱难度”的测试中，ROME依然保持了极具竞争力的表现。

### 总结

这篇论文最大的价值，或许不在于ROME模型本身，而在于它向社区展示了一套完整的**Agent生产流水线**。

从 **ROCK** 的安全沙盒，到 **ROLL** 的高效训练，再到 **IPA** 算法对长程交互的优化，阿里证明了：在Agent时代，模型能力的提升不再仅仅依赖于堆砌数据和参数，更依赖于**环境（Environment）**、**数据合成（Data Synthesis）**与**训练系统（System）**的深度协同。

正如论文所言：“ROME wasn't built in a day.” 想要构建通用的Agent，我们需要先构建好底层的“罗马基建”。