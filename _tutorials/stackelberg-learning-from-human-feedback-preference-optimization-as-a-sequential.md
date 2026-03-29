---
layout: default
title: "Stackelberg Learning from Human Feedback: Preference Optimization as a Sequential Game"
---

## ETH新作SLHF：用“领导者-跟随者”博弈重塑对齐，推理性能零样本提升

<img src="/images/2512.16626v1/A__title.jpg" alt="" style="width:85%; max-width:600px; margin:auto; display:block;">

在大模型对齐（Alignment）的战场上，**人类反馈强化学习**（**Reinforcement Learning from Human Feedback, RLHF**）无疑是当前的霸主。然而，你是否想过，RLHF 赖以生存的“奖励模型”可能从根本上就是一种过度简化？

> ArXiv URL：http://arxiv.org/abs/2512.16626v1

现实中的人类偏好往往充满了“石头剪刀布”式的循环（A 优于 B，B 优于 C，但 C 却优于 A），这种非传递性（Intransitive）的偏好结构，是传统的标量奖励函数无法捕捉的。为了解决这一难题，来自苏黎世联邦理工学院（ETH Zürich）的研究团队提出了一种全新的框架——**Stackelberg人类反馈学习**（**Stackelberg Learning from Human Feedback, SLHF**）。

该研究并未沿用传统的奖励建模思路，而是将对齐问题重构为一场**序贯博弈**（**Sequential Game**）。在这个游戏中，模型不再是单纯的“得分机器”，而是分化为两个角色：“领导者”和“跟随者”。这种设计不仅解决了复杂偏好建模的问题，更带来了一个令人惊喜的副作用：模型在推理阶段竟然具备了“自我修正”的能力，无需额外微调即可提升性能。

### 告别标量奖励的局限

目前的 RLHF 流程通常假设人类偏好可以通过一个标量奖励函数（Reward Model）来表示，这通常基于 Bradley-Terry 模型。然而，这种假设在面对复杂的人类偏好时显得捉襟见肘。

当偏好出现循环（如群体意见不一致导致的 Condorcet 悖论）时，并不存在一个绝对的“最优解”。之前的尝试如 **Nash人类反馈学习**（**NLHF**）试图通过纳什均衡来解决这个问题，但它构建的是一个“同时行动”的游戏。这就好比两个人在黑暗中出拳，双方都要猜测对方的动作，导致训练过程极不稳定，且最终策略往往是随机的混合策略。

SLHF 则另辟蹊径，引入了 Stackelberg 博弈模型。这是一个**序贯移动**的游戏：

1.  **领导者（Leader, $\pi$）**：率先做出承诺，采取一个行动（生成文本）。

2.  **跟随者（Follower, $\omega$）**：观察领导者的行动，并在此基础上做出反应（生成更好的文本）。

这种不对称性改变了游戏规则。跟随者面对的是一个已知的、固定的动作，只需解决一个简单的“优化问题”；而领导者则必须学会“预判”，它需要选择那些即使面对最强对手（跟随者）也能立于不败之地的行动。

### SLHF 的数学架构

SLHF 将对齐问题形式化为以下优化目标：




{% raw %}$$ \max_{\pi\in\Pi}\min_{\omega\in\Omega}\mathbb{E}_{x\sim\rho}\!\Big[\mathbb{E}_{y\sim\pi(\cdot\mid x)}\!\big[\mathbb{E}_{y^{\prime}\!\sim\omega(\cdot\mid x,y)}\!\big[p(y\succ y^{\prime}\mid x)\big]+\tau^{F}\mathrm{KL}_{x,y}(\omega\,\ \mid \,\omega^{\textnormal{ref}})\big]-\tau^{L}\mathrm{KL}_{x}(\pi\,\ \mid \,\pi^{\textnormal{ref}})\!\Big] $${% endraw %}



在这个公式中：

*   $\pi$ 是领导者策略，$\omega$ 是跟随者策略。

*   跟随者 $\omega$ 试图最小化领导者的胜率（即最大化 $y' \succ y$ 的概率），同时受到 KL 散度的约束。

*   领导者 $\pi$ 则试图最大化自己在面对最佳跟随者时的胜率。

研究团队证明，在标准的正则化假设下，SLHF 拥有唯一的 **Stackelberg均衡**（**Stackelberg Equilibrium**）。与纳什均衡不同，Stackelberg 均衡通常允许确定性的策略，这使得模型输出更加稳定可靠。

为了求解这个均衡，论文提出了 **StackelbergGDA** 算法。这是一种双时间尺度的梯度下降-上升算法（Two-timescale Gradient Descent Ascent）。关键点在于，跟随者的学习率 $\eta^F$ 被设定为大于领导者的学习率 $\eta^L$。这意味着跟随者能够快速适应领导者的变化，从而为领导者提供准确的反馈信号。

### 推理时的自我进化

SLHF 最具吸引力的特性在于其天然支持**推理时修正**（**Inference-time Refinement**）。

在传统的 RLHF 中，训练好的 Policy 就是最终产品。但在 SLHF 中，我们得到了两个互补的策略：

1.  领导者 $\pi$ 能够生成高质量的初始响应。

2.  跟随者 $\omega$ 经过训练，专门用于在给定领导者输出的情况下，生成一个“更好”的响应。

这意味着在推理阶段，我们可以直接利用跟随者来改进输出，而无需任何额外的训练或外部奖励模型。通过迭代采样，跟随者可以充当一个“润色器”或“纠错者”，不断提升回复的质量。这种能力在偏好发生变化（例如从群体偏好转向个人偏好）时尤为宝贵。

### 实现与实验

为了在大模型上高效实现 SLHF，研究者设计了一种巧妙的 Prompt 模板，使得领导者和跟随者可以共享同一个模型参数（通过 LoRA 等技术），只是输入提示不同。

![Prompt templates used to train a single-model for both Leader and Follower completions.](https://arxiv.org/html/2502.02985/x1.png)

实验结果表明：

*   **胜率显著**：在多个数据集上，SLHF 的跟随者策略始终优于 RLHF 和 NLHF 基线。

*   **扩展性强**：该方法在 0.5B 到 8B 参数量的模型上均表现出良好的扩展性。

*   **泛化能力**：最令人惊讶的是，SLHF 训练出的跟随者具有极强的泛化性。它可以用来修正其他完全不同模型生成的输出，实现跨模型的性能提升。

### 总结

SLHF 为大模型对齐提供了一个全新的视角。它不再执着于寻找一个完美的标量奖励函数，而是承认偏好的复杂性，并通过博弈论的方法加以解决。

通过引入“领导者-跟随者”的序贯结构，SLHF 不仅解决了非传递性偏好的建模难题，更重要的是，它将“自我修正”内化为了模型的一种原生能力。这种在推理阶段零样本提升性能的潜力，或许预示着下一代对齐技术的新方向。