---
layout: default
title: "A Comedy of Estimators: On KL Regularization in RL Training of LLMs"
---

## RL微调避坑指南：揭秘KL正则化的“梯度陷阱”与性能真相

<img src="/images/2512.21852v1/A__title.jpg" alt="" style="width:85%; max-width:450px; margin:auto; display:block;">

随着 DeepSeek-R1 等模型的爆火，**强化学习**（**Reinforcement Learning, RL**）在提升大模型推理能力方面的潜力再次成为焦点。现在的开发者们都在忙着复现 GRPO、PPO，试图让自己的模型在数学和代码任务上“顿悟”。

> ArXiv URL：http://arxiv.org/abs/2512.21852v1

但在你按下训练开始键之前，有没有想过一个看似微不足道的细节——**KL 散度（KL Divergence）**——可能正是决定你训练成败的关键？

为什么有的 RL 训练稳步上升，而有的却突然崩盘？为什么明明公式写的是优化 KL 正则化目标，代码实现出来的梯度却完全是另一回事？

今天要解读的这篇来自 CIFAR、Mila 和 McGill 大学等机构的论文 **《A Comedy of Estimators: On KL Regularization in RL Training of LLMs》**，就为你揭开这个被长期忽视的“隐形陷阱”。这不仅是一场关于估计器的“喜剧”，更是一份关于如何避免训练崩溃的严肃指南。

### 核心问题：我们真的算对 KL 了吗？

在 LLM 的 RL 训练（特别是 **RLVR**，即 **Reinforcement Learning with Verifiable Rewards**）中，为了防止模型在追求高奖励的过程中“遗忘”原本的语言能力，或者产生乱码（Language Drift），我们通常会在目标函数中加入一个正则项：**反向 KL 散度**（Reverse KL Divergence）。

标准的目标函数通常长这样：




{% raw %}$$ \max_{\theta} \mathbb{E} [ \text{Reward} - \beta \cdot \text{KL}(\pi_{\theta}  \mid  \mid  \pi_{\text{ref}}) ] $${% endraw %}



其中 $\pi\_{\theta}$ 是当前策略，$\pi\_{\text{ref}}$ 是参考策略（SFT 模型）。

问题在于，精确计算整个序列空间上的 KL 散度是不可能的（intractable）。因此，实际操作中我们必须使用**估计器（Estimator）**。

目前的开源库和论文中，大家都在用各种各样的“土办法”来近似这个 KL 值。最要命的是，大家把这个近似值放的位置也不一样：

1.  **放在奖励（Reward）里**：像 PPO 的某些实现，把 KL 惩罚当成一种负奖励。

2.  **放在损失（Loss）里**：像 GRPO（Group Region Policy Optimization）的流行实现，直接把 KL 项加到 Loss 函数中。

这篇论文的核心发现令人震惊：**这些不同的“写法”，会导致截然不同的梯度行为。很多流行的写法（比如直接加在 Loss 里），其实计算出的梯度是有偏的（Biased），这直接导致了训练的不稳定或性能下降。**

### 两位主角：K1 与 K3 估计器

论文主要研究了两种最常见的 KL 估计器：

1.  **K1 估计器（Naïve Estimator）**：

    这是最直观的写法，直接计算 log 概率之差：

    


    {% raw %}$$ K1 = \log \frac{\pi_{\theta}(y)}{\pi_{\text{ref}}(y)} = \log \pi_{\theta}(y) - \log \pi_{\text{ref}}(y) $${% endraw %}



2.  **K3 估计器（Schulman Estimator）**：

    由 John Schulman (OpenAI) 提出，旨在降低方差：

    


    {% raw %}$$ K3 = \frac{\pi_{\text{ref}}(y)}{\pi_{\theta}(y)} - 1 - \log \frac{\pi_{\text{ref}}(y)}{\pi_{\theta}(y)} $${% endraw %}



这两种估计器本身在统计上都是无偏的。**但是**，当你把它们放入 RL 的优化目标（Reward 或 Loss）并求导时，事情就变得复杂了。

### 梯度的“罗生门”：放在哪里很重要

论文通过数学推导和实验，总结出了一个关键的“避坑矩阵”：


| 估计器类型 | 放在 Reward 中 | 放在 Loss 中 | 结果分析 |
| :--- | :--- | :--- | :--- |
| **K1 (Naïve)** | **无偏梯度 (Unbiased)** ✅ | **有偏梯度 (Biased)** ❌ | **放在 Reward 中是最佳选择**；放在 Loss 中会导致严重的训练不稳定。 |
| **K3 (Schulman)** | **有偏梯度 (Biased)** ❌ | **有偏梯度 (Biased)** ❌ | 虽然都是有偏的，但 K3 放在 Loss 中（GRPO 的常见做法）通常能保持训练稳定，但性能不如无偏版本。 |

**关键结论 1：K1 放在 Reward 里才是“正解”**

论文证明，只有将 K1 估计器作为惩罚项加入到 Reward 中（即 $R' = R - \beta \cdot K1$），并且使用标准的策略梯度（Policy Gradient）更新时，我们得到的才是针对原目标函数的**无偏梯度估计**。

**关键结论 2：直接把 KL 加到 Loss 里通常是错的**

很多代码库为了方便，直接写 $$loss = policy_loss + beta * kl_loss$$。论文指出，这种做法（无论是用 K1 还是 K3）计算出的梯度，在数学上并不等于原目标函数（Eq. 1）的梯度。这是一种“梯度偏差”。

### 实验验证：偏差带来的代价

为了验证理论，研究团队使用 Qwen2.5-7B 和 Llama-3.1-8B 在 MATH 数据集上进行了大量的 RL 微调实验。

#### 1. 训练稳定性：错误的配置会导致崩溃

如下图所示，当试图将 K1 估计器直接放入 Loss 函数时（红色和橙色线），随着 KL 系数 $\beta$ 的增加，训练迅速崩溃，Pass@1 准确率跌至谷底。而将 K1 放入 Reward 中（蓝色线）则非常稳定。

![Figure 2: Training Instabilities when using K1 in loss.](images/page_7_Figure_3.jpg)

#### 2. 性能对比：无偏梯度带来更好的泛化

虽然 K3 放在 Loss 中（GRPO 的做法，下图中左侧）也能稳定训练，但在同等条件下，**使用无偏梯度的配置（K1 in Reward，下图中右侧）能达到更高的峰值性能**。

更重要的是，无偏梯度在**域外任务（Out-of-Distribution）**上的表现也更好。这意味着模型不仅仅是学会了做数学题，而是真正保留了通用的推理能力，没有被错误的梯度带偏。

![Figure 4: Pass@1 performance on MATH test set with K3-in-loss vs K1-in-reward](images/page_8_Figure_3.jpg)

#### 3. 异步训练中的表现

在工业界常用的异步训练（Asynchronous RL）设置下，数据的 Off-policy 程度更高。实验表明，**K1-in-Reward** 和 **K3-in-Loss** 是唯二能保持稳定的配置，而 K1-in-Reward 依然保持了微弱的优势。

![Figure 7: Comparison of different KL configurations in asynchronous RL setting](images/page_10_Figure_1.jpg)

### 总结与建议

这篇论文像是一部侦探小说，破解了 RL 训练中那些莫名其妙的“崩溃悬案”。对于正在尝试 RLHF 或 RLVR 的开发者，以下是几条黄金法则：

1.  **检查你的代码库**：不要盲目信任开源实现。去看看 KL 惩罚到底是在哪里计算的。

2.  **首选 K1-in-Reward**：如果你追求最佳的性能和数学上的正确性，请将 KL 散度（$\log \pi - \log \pi\_{ref}$）作为一项负奖励加到 Reward Model 的输出上，然后走标准的 PPO/GRPO 流程。这是目前发现的唯一能产生无偏梯度的配置。

3.  **警惕 K1-in-Loss**：千万不要直接把简单的 log 差值加到 Loss 函数里去优化，这几乎必然导致训练不稳定。

4.  **GRPO 的 K3-in-Loss 尚可**：如果你正在使用 GRPO 且代码默认将 KL 放入 Loss，请确保使用的是 K3（Schulman）估计器。它虽然有梯度偏差，但至少是稳定的。不过，如果有能力改写，尝试将其移至 Reward 端可能会带来意外之喜。

**简单来说：想让模型更聪明？先把 KL 散度算对！**