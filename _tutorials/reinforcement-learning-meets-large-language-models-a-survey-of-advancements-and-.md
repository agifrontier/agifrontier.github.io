---
layout: default
title: "Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle"
---

# Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle

- **ArXiv URL**: http://arxiv.org/abs/2509.16679v1

- **作者**: Dingkang Yang; Yuchi Wang; Ziyun Qian; Jun Liu; Peng Zhai; Hongsheng Li; Weijie Yin; Yang Liu; Lihua Zhang

- **发布机构**: ByteDance; Fudan University; Lancaster University; The Chinese University of Hong Kong; The University of Toronto; Tongji University

---

# Reinforcement Learning Meets Large Language Models: A Survey of Advancements and Applications Across the LLM Lifecycle

## 引言

近年来，以ChatGPT为代表的大语言模型（Large Language Models, LLMs）在通用对话、代码生成和数学推理等任务中展现了卓越的性能。然而，当前的LLMs仍存在关键缺陷：它们难以可靠地捕捉细微的人类意图，可能产生误导性或不安全的输出，并且其推理能力仍有显著不足。为了应对这些挑战，强化学习（Reinforcement Learning, RL）被引入作为一个强大的框架，通过交互式反馈和奖励信号直接优化模型行为，旨在使LLMs的生成能力与人类的偏好、价值观和特定任务需求对齐，并增强其解决复杂问题的推理能力。

自Ouyang等人提出基于人类反馈的强化学习（Reinforcement Learning from Human Feedback, RLHF）以来，基于RL的微调已成为提升LLM对齐能力的核心方法。近期，研究者们开始将RL范式应用于增强模型的推理能力，特别是通过一种名为**带可验证奖励的强化学习**（Reinforcement Learning with Verifiable Rewards, RLVR）的新范式。RLVR通过为模型提供客观、可自动验证的奖励信号（如代码单元测试或数学证明），直接激励模型生成可靠且逻辑正确的解决方案。这一方法已成为推动顶尖LLM（如GPT-o1、Claude 3.7/4、DeepSeek R1）推理能力突破的关键驱动力。

尽管取得了显著进展，但RL与LLM的结合仍面临诸多悬而未决的问题：
1.  RLVR在多大程度上真正扩展了LLM固有的推理能力，而非仅仅放大其预训练知识？
2.  在LLM生命周期的不同阶段（预训练、对齐微调、推理优化）应如何最佳地应用不同的RL技术？
3.  如何高效构建高质量的奖励数据集（无论是人类偏好、AI辅助偏好还是程序化奖励）？
4.  如何在大规模训练中高效实施RL微调，同时避免模型性能不稳定？

本综述旨在系统性地回顾RL增强LLM领域的最新进展，重点关注RLVR范式。本文将围绕LLM的整个生命周期，深入剖析RL在**预训练**、**对齐微调**和**强化推理**等不同阶段的应用策略、理论基础、数据集、基准以及开源工具。

### 相关综述

近年来已有多篇综述探讨了与LLM相关的RL研究，但它们通常范围有限。例如，部分研究仅关注基于RL的对齐技术，而忽略了新兴的方法。尽管2025年的一些工作开始总结推理时的RL应用，但其分析往往不够全面。相比之下，本综述系统地考察了RL在LLM整个生命周期（从预训练到推理）中的作用，并提出了一个更全面的组织框架。

**代表性综述对比分析表**


| ↓→ | 生命周期覆盖度 | 数据集与基准总结 | 工具/框架收集与实用性 | 引文广度与时效性 | 未来展望与挑战 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Wang et al. 2024a | | | | | |
| Srivastava and Aggarwal 2025 | | | | | |
| Wang et al. 2024d | | | | | |
| Cao et al. 2024 | | | | | |
| Chaudhari et al. 2024 | | | | | |
| Kaufmann et al. 2024 | | | | | |
| 本文 | | | | | |

### 贡献总结

本文的贡献主要体现在三个方面：
1.  **全生命周期覆盖**：系统性地覆盖了RL在LLM中的完整应用生命周期，详细阐述了从预训练、对齐到强化推理的每个阶段的目标、方法和挑战。
2.  **聚焦RLVR前沿**：重点介绍了RLVR的最新进展，深入分析了其确保奖励客观可验证的方法论，并探讨了它在提升模型性能和对齐方面的优势与局限。
3.  **整合实用资源**：总结了用于LLM中RL实验、评估和实践的关键数据集、基准及开源框架，为未来的研究和应用提供了宝贵的资源。

### 分类体系

为了提供一个清晰的组织路线图，本文提出了一个RL增强LLM的分类体系，如下图所示。该体系将现有方法和资源分为五个主要分支：
1.  **预训练 (Pre-training)**：在初始阶段引入RL。
2.  **对齐 (Alignment)**：使用RL使模型与人类偏好对齐。
3.  **带可验证奖励的强化学习 (RLVR)**：利用客观、可验证的奖励信号进行推理增强。
4.  **数据集与基准 (Datasets & Benchmarks)**：用于训练和评估RL微调模型的相关资源。
5.  **开源框架 (Open-source Frameworks)**：支持大规模RL训练的工具。

![RL增强LLM分类体系](https://s2.loli.net/2024/09/20/ChYxXz3pU9Jqf5T.png)

## 强化学习基础

RL使智能体通过与环境交互来学习最优策略，以最大化累积奖励。一个典型的RL问题可以建模为马尔可夫决策过程（Markov Decision Process, MDP）。RL算法沿着两个主要范式发展：基于策略的学习和基于价值的学习。

### 策略学习

策略学习方法直接优化策略 $$$\pi(a \mid s;\theta)$$$。
*   **REINFORCE**：最基础的蒙特卡洛策略梯度方法，其策略梯度的无偏估计为：
    

    {% raw %}$$
    \nabla_{\theta}J(\theta)=\mathbb{E}_{\tau\sim\pi_{\theta}}\left[\sum_{t=0}^{T}\nabla_{\theta}\log\pi_{\theta}(a_{t} \mid s_{t})R_{t}\right]
    $${% endraw %}


    其中 $$$R\_t$$$ 是从时间步 $$t$$ 开始的折扣回报。为减小方差，可以引入基线函数 $$$b(s\_t)$$$，此时梯度变为：
    

    {% raw %}$$
    \nabla_{\theta}J(\theta)=\mathbb{E}\left[\sum_{t}\nabla_{\theta}\log\pi_{\theta}(a_{t} \mid s_{t})\left(R_{t}-b(s_{t})\right)\right]
    $${% endraw %}


    其中 $$$A\_t = R\_t - b(s\_t)$$$ 是优势函数（Advantage Function）。

*   **Actor-Critic (AC)**：结合了策略梯度和价值函数近似。**Actor**（行动者）根据策略选择动作，**Critic**（评论家）使用价值函数评估策略，并为Actor提供低方差的梯度估计。

*   **信赖域策略优化 (Trust Region Policy Optimization, TRPO)**：通过将策略更新限制在与旧策略的KL散度（KL Divergence）的一个小“信赖域”内，来避免过大的策略更新导致的性能崩溃。

*   **近端策略优化 (Proximal Policy Optimization, PPO)**：通过一个截断的替代目标函数（clipped surrogate objective）来简化TRPO的约束优化，实现了稳定且高效的策略更新。其目标函数为：
    

    {% raw %}$$
    L^{\mathrm{PPO}}(\theta)=\mathbb{E}_{t}[\min(r_{t}(\theta)\hat{A}_{t},\mathrm{~clip}(r_{t}(\theta),1-\epsilon,1+\epsilon)\hat{A}_{t})]
    $${% endraw %}


    其中 $$$r\_t(\theta)$$$ 是新旧策略的概率比，$$$\hat{A}\_t$$$ 是优势估计。PPO是RLHF中的主流算法。

*   **组相对策略优化 (Group Relative Policy Optimization, GRPO)**：为解决PPO在LLM长序列推理中价值网络估计不准和计算成本高的问题，GRPO引入了一种新颖的基线思想。它为每个提示（prompt）采样一组（group）输出，并使用组内奖励的均值作为动态基线，从而无需训练一个独立的价值网络。其优势函数计算方式为：
    

    {% raw %}$$
    \hat{A}_{i,t}=\frac{r_{i}-\max(\{R_{i}\}_{i=1}^{G})}{\operatorname{std}(\{R_{i}\}_{i=1}^{G})}
    $${% endraw %}


    此方法简化了算法结构，提高了训练效率。

### 价值学习
价值学习方法通过估计价值函数来间接推导最优策略。
*   **Q-learning**：一种经典的离策略（off-policy）算法，通过贝尔曼最优方程（Bellman optimality equation）迭代更新动作价值函数 $$$Q(s, a)$$$。其更新规则为：
    

    {% raw %}$$
    Q_{new}(s_{t},a_{t})\leftarrow Q(s_{t},a_{t})+\alpha\begin{bmatrix}r_{t}+\gamma\max_{a^{\prime}}Q(s_{t+1},a^{\prime})-Q(s_{t},a_{t})\end{bmatrix}
    $${% endraw %}



*   **SARSA**：一种在策略（on-policy）算法，其更新依赖于当前策略实际采取的下一个动作 $$$a\_{t+1}$$$，而不是所有可能动作中的最优动作。更新规则为：
    

    {% raw %}$$
    Q_{new}(s_{t},a_{t})\leftarrow Q(s_{t},a_{t})+\alpha\begin{bmatrix}r_{t}+\gamma Q(s_{t+1},a_{t+1})-Q(s_{t},a_{t})\end{bmatrix}
    $${% endraw %}



*   **深度Q网络 (Deep Q-Network, DQN)**：将深度神经网络引入Q-learning，用一个网络 $$$Q(s, a; \theta)$$$ 来近似Q函数。通过经验回放（experience replay）和目标网络（target network）等技术解决了训练不稳定的问题。

在LLM领域，由于动作空间（所有可能的Token序列）巨大，价值学习方法不作为主流训练框架。但其核心思想在某些任务中仍有体现，例如动态选择上下文示例。

## 预训练与对齐阶段的强化学习方法

### 预训练阶段的强化学习方法
目前，将RL应用于LLM预训练的研究尚处于早期阶段。
*   **Dong等人**将预训练中的“下一个Token预测”任务重构为RL任务，当模型正确预测时给予可验证奖励。但此方法资源消耗巨大。
*   **Ghosh和Levine**将无标签图像的预训练视为一个RL问题，并通过自举（bootstrapping）的方式引入RL。
*   **OctoThinker**提出了一种两阶段的“中训练”（mid-training）策略，使用高质量、任务相关的数据对预训练模型进行继续训练，旨在使其更适应后续的RL训练，从而提升模型的RL扩展性。

### 对齐阶段的经典算法
对齐阶段的目标是使LLM的行为符合人类的偏好和价值观。
*   **RLHF**: 由Ouyang等人开创，已成为LLM对齐的 foundational paradigm。它包含三个步骤：1) 监督微调（SFT）；2) 训练奖励模型（RM）；3) 使用PPO等RL算法根据RM的反馈优化LLM策略。
*   **AI辅助对齐**:
    *   **Constitutional AI**: 由Bai等人提出，让AI系统根据一套预设的规则（宪法）自我监督和改进，无需人类标注有害输出。
    *   **RLAIF (Reinforcement Learning with AI Feedback)**: 利用一个更强大的LLM作为“裁判”来生成偏好标签，替代人类标注者，降低了数据成本。
*   **直接偏好优化 (Direct Preference Optimization, DPO)**: Rafailov等人提出的一种绕过显式奖励建模和RL训练的对齐方法。DPO通过一个简单的损失函数，直接根据偏好数据对LLM进行微调，其效果被证明等价于RLHF。这一方法大大简化了对齐过程，并催生了一系列变体，如$$DPOP$$、$$$\beta$-DPO$$。
*   **其他新兴对齐方法**:
    *   **KTO (Kahneman-Tversky Optimization)**: 仅使用二元反馈（好/坏）而非成对偏好数据进行优化。
    *   **ORPO (Odds Ratio Policy Optimization)**: 将SFT与偏好优化结合在一个统一的损失函数中，无需参考模型即可提升指令遵循能力。

### 新兴的奖励模型设计方法
奖励模型（Reward Model, RM）的质量直接决定了RL对齐的效果。近期研究致力于提升RM的性能和泛化能力。
*   **基于推理的奖励模型**:
    *   **RRM (Reward Reasoning Model)**: 通过思维链（Chain-of-Thought, CoT）推理，让RM在判断复杂响应时能够“思考”，并利用RL框架进行训练。
    *   其他研究将奖励建模本身视为一个推理任务，通过可验证的奖励和两阶段训练来提升RM的性能。
*   **统一与生成式奖励模型**:
    *   **UNIFIEDREWARD-THINK**: 首个基于多模态CoT推理的统一奖励模型。
    *   **GRM (Pointwise Generative Reward Modeling)**: 能够评估单个、成对或多个响应，克服了传统RM在输入灵活性上的限制。
*   **泛化与自动化奖励设计**:
    *   研究探索让RM像LLM一样遵循动态提供的自然语言原则来做出判断，以提升其泛化能力。
    *   **AUTORULE**: 从偏好反馈中自动提取规则，并利用验证器将规则满足度作为辅助奖励。

## 推理阶段的强化学习方法
自2025年以来，随着GPT-o1和DeepSeek R1等模型的发布，研究焦点逐渐转向在推理阶段使用RL，特别是RLVR技术，以突破LLM的推理能力极限。

### RLVR在提升LLM推理能力方面的实验发现
RLVR在数学和编程等任务中取得了显著成功，但也引发了关于其作用机制的学术争议。
*   **争议核心：RLVR是“发现”还是“放大”？**
    *   **支持“发现”**: Llama-colt的研究发现，在给予足够训练时间的情况下，RL能够发现基础模型中完全不存在的全新解题路径。
    *   **支持“放大”**: Yue等人的研究表明，尽管RLVR能提高找到正确答案的采样效率（即在小样本量k下的pass@k更高），但随着k的增大，基础模型的性能反而会超越RLVR模型。这表明RLVR生成的推理路径都已存在于基础模型的采样分布中，RLVR更像是一个高效的“采样器”，而非能力的“创造者”。
*   **策略多样性与熵崩溃 (Entropy Collapse)**
    *   研究发现，长时间的RL训练会导致策略熵急剧下降，造成“多样性崩溃”，模型会忘记已掌握的知识，也难以解决新问题。
    *   Cui等人建立了策略性能 $$$R$$$ 与熵 $$$\mathcal{H}$$$ 之间的关系式：$$$R=-a\exp(\mathcal{H})+b$$$，揭示了性能提升是以熵消耗为代价的，存在理论上限。
    *   为解决此问题，研究提出通过熵管理来维持探索，例如只对引发逻辑分支的“高熵Token”（如“因此”、“假设”）进行策略更新。
*   **对思维过程的反思**
    *   有研究质疑显式思维链（CoT）的必要性，发现对于某些任务，绕过思考过程反而能取得更好结果。
    *   Samineni等人批判性地指出，当前RL设置下的推理过程更像是“过滤后的迭代式监督微调”，响应长度的增加是训练设置的副作用，而非推理能力的真正提升。

### 面向LLM的强化学习算法新进展
为应对长链推理中的挑战，一系列针对LLM的RL算法被提出，大多围绕GRPO进行改进。
*   **DAPO (Decoupled Clip and Dynamic Sampling Policy Optimization)**: 对GRPO的改进，包含四项关键技术：
    1.  **Clip-Higher**: 放松PPO中对策略更新的对称截断，允许策略更自由地探索高奖励区域。
    2.  **Dynamic Sampling**: 对奖励极端（过高或过低）的提示进行重采样，避免样本浪费。
    3.  **Token-Level Policy Gradient Loss**: 按序列长度加权损失，抑制冗长低质输出。
    4.  **Overlong Reward Shaping**: 惩罚或截断过长的输出，稳定训练过程。
*   **Open-Reasoner-Zero**: 采用极简训练策略，无需预训练微调，以极高的效率提升了基础模型的推理能力。
*   **其他前沿算法**:
    *   **Kimina-Prover**: 将RL应用于定理证明，构建结构化的形式推理模式。
    *   **SRPO**: 采用两阶段RL中心训练策略，提升跨任务推理能力。
    *   **TANGO**: 通过RL联合训练生成器和过程级验证器，使其协同进化。
    *   **REINFORCE++**: 实现了类似PPO的稳定性和效率，但无需价值网络。
    *   **SuperRL**: 通过自适应机制检测奖励稀疏性，并切换到混合监督执行器以稳定学习。
    *   **KDRL**: 将知识蒸馏（Knowledge Distillation, KD）与RL（GRPO）结合。
    *   **R2-Reasoner**: 使用一个强化学习训练的“路由器”（router）将复杂问题分解，并分配给不同的LLM协同解决。
    *   **ToTRL**: 将思想树（Tree-of-Thoughts, ToT）的探索过程与RL相结合，通过强化学习引导模型在多路径逻辑问题中进行更高效的探索。