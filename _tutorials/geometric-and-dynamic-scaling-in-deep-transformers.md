---
layout: default
title: "Geometric and Dynamic Scaling in Deep Transformers"
---

## Transformer 越深越“傻”？几何视角揭秘百层大模型坍塌之谜

<img src="/images/2601.01014v1/A__title.jpg" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

在深度学习的殿堂里，我们一直信奉“更深即更强”。然而，当你试图将 Transformer 堆叠到 100 层甚至更深时，一个诡异的现象出现了：模型不仅没有变聪明，反而开始“坍塌”——特征变得越来越雷同，表达能力急剧下降。

> ArXiv URL：http://arxiv.org/abs/2601.01014v1

这仅仅是因为梯度消失吗？还是我们的优化器不够好？

来自纽约大学和石溪大学的研究团队给出了一个颠覆性的视角：**这不是优化问题，而是几何问题。** 传统的残差连接（Residual Connection）正在把你的模型推向“歧途”。今天，我们就来深度拆解这篇论文《Geometric and Dynamic Scaling in Deep Transformers》，看看他们提出的 **MGT（流形几何 Transformer）** 如何通过给神经网络装上“导航仪”和“橡皮擦”，打破深度缩放的诅咒。

### 核心痛点：深度 Transformer 的“几何迷失”

为什么深层网络会坍塌？论文指出，现有的 Transformer 架构存在两个致命的几何缺陷：

1.  **盲目累加（Write-Only Memory）**：

    标准的残差更新公式 $\mathbf{x}\_{l+1} = \mathbf{x}\_{l} + \mathcal{F}(\mathbf{x}\_{l})$ 假设所有的特征累加都是有益的。但这就像只准写不准擦的黑板，随着层数增加，信息不断堆积，噪声也随之累积，最终导致“秩坍塌”（Rank Collapse），即所有特征向量都指向同一个方向，失去了区分度。

2.  **脱轨风险（Manifold Drift）**：

    根据流形假设，有效的数据特征应该分布在一个低维流形（Manifold）上。但是，神经网络的更新向量 $\mathcal{F}(\mathbf{x}\_{l})$ 往往是一个高维欧几里得空间中的无约束向量。简单来说，模型每走一步，都可能一脚踩空，掉出这个“有效语义流形”，导致特征退化。

### 解决方案：MGT 的两大护法

为了解决这个问题，作者提出了 **MGT（Manifold-Geometric Transformer）**，它引入了两个核心机制，分别解决了“往哪走”和“走多远”的问题。

#### 1. 往哪走？—— 流形约束超连接 (mHC)

如果把模型更新比作登山，**mHC (Manifold-Constrained Hyper-Connections)** 就是那个时刻修正路线的向导。

传统的更新向量 $\mathbf{v}\_{raw}$ 是盲目的。mHC 的作用是将这个向量“投影”到当前数据流形的切空间（Tangent Space）上。




{% raw %}$$ \mathbf{v}_{mHC}=\mathbf{v}_{raw}\odot\sigma(\text{LN}(\mathbf{W}_{gate}\mathbf{x}_{l})) $${% endraw %}



虽然计算精确的切空间成本太高，但作者巧妙地使用了一种**软子空间近似（Soft Subspace Approximation）**。通过一个门控机制，mHC 抑制了那些偏离当前语义轨迹的“噪声方向”，确保每一步更新都走在正确的“语义道路”上，防止模型脱轨。

#### 2. 走多远？—— 深度增量学习 (DDL)

确定了方向，还需要控制步伐。这就是 **DDL (Deep Delta Learning)** 的用武之地。它赋予了模型“擦除”记忆的能力。

传统的残差连接只能做加法。而 DDL 引入了一个动态门控 $\beta$，允许模型执行类似 Householder 变换的操作：




{% raw %}$$ \mathbf{x}_{l+1}=\mathbf{x}_{l}+\mathbf{\beta}\odot(\mathbf{v}_{mHC}-\alpha\cdot\text{Proj}_{\mathbf{x}}(\mathbf{x}_{l})) $${% endraw %}



这里的关键在于 $\beta$ 可以是负数！这意味着模型不仅可以积累信息（加法），还可以**主动擦除**冗余或过时的信息（减法/反射）。

*   **几何有效性**：mHC 保证方向正确。

*   **动态遍历**：DDL 保证可以前进也可以后退（擦除）。

两者结合，构成了 MGT 的核心逻辑：**在正确的几何流形上，灵活地进行读写操作。**

![Architecture of the Manifold-Geometric Transformer (MGT) Block](https://arxiv.org/html/2501.00895v1/extracted/6106660/figures/arch_diagram.png)

*图1：MGT 模块架构图。清晰地展示了特征生成、通过 mHC 进行几何矫正（蓝/紫线），以及通过 DDL 进行动态擦除（橙线）的过程。*

### 实验设计：挑战 100+ 层极限

这篇论文不仅仅是理论推导，还设计了一套非常硬核的“压力测试”方案，旨在证伪“几何约束是深层扩展的关键”这一假设。

*   **秩演化分析（Rank Evolution）**：直接测量随着层数增加，特征矩阵的有效秩（Effective Rank）是否还能保持住。这是检验“坍塌”最直观的指标。

*   **协同效应验证**：通过消融实验（Ablation Study），证明 mHC 和 DDL 缺一不可。mHC 负责空间正则化，DDL 负责动态优选，两者是乘数效应而非加法效应。

*   **深度缩放测试**：直接对比 MGT 和标准 Transformer 在超深层设置下的表现。如果假设成立，MGT 应该在深度增加时表现出更强的鲁棒性，而普通 Transformer 则会性能饱和甚至下降。

### 总结与展望

MGT 的提出，本质上是对残差连接的一次“几何学修正”。它告诉我们，在构建超深网络时，不能只是一味地堆叠层数，更要考虑数据在流形上的几何演化。

**为什么这很重要？**

随着大模型对上下文长度和推理能力要求的提高，未来的模型势必会更深。MGT 提供了一种无需复杂优化技巧，仅通过架构改进就能维持深层信号完整性的思路。

如果你的模型在加深层数后效果不升反降，或许是时候检查一下：它是不是在几何空间里“迷路”了？