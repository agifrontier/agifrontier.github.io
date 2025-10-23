---
layout: default
title: "Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning"
---

# Every Attention Matters: An Efficient Hybrid Architecture for Long-Context Reasoning

- **ArXiv URL**: http://arxiv.org/abs/2510.19338v1

- **作者**: Chen Liang; Feng Zhu; Yibo Cao; Peng Jiao; Jingyu Hu; Mingyang Zhang; Yixuan Sun; Yankun Ren; Jun Zhou; Yao Zhao; 等26人

---

# TL;DR
本文提出了一种名为 Ring-linear 的高效混合注意力架构，通过巧妙地结合线性注意力和 Softmax 注意力，在大幅降低长上下文推理成本的同时，保持了强大的复杂推理能力。

# 关键定义
*   **混合线性注意力 (Hybrid Linear Attention)**: 本文提出的核心架构，它将模型中的 Transformer 层分为多个层组 (Layer Group)。在每个层组内，大部分层使用高效的线性注意力，仅保留一个层使用标准的 Softmax 注意力（如 GQA）。这种设计旨在平衡线性注意力的效率和 Softmax 注意力的表达能力。
*   **线性注意力 (Linear Attention)**: 本文采用 Lightning Attention 作为具体实现。与传统 Softmax 注意力二次方复杂度 $$O(n^2d)$$ 不同，其计算复杂度与序列长度 $$n$$ 成线性关系 $$O(nd^2)$$，且其状态记忆（等效于 KV 缓存）大小为常数 $$O(d^2)$$，不随序列长度增长而增长，极大地提升了长序列处理的效率。
*   **层组 (Layer Group)**: 混合架构的基本构建单元，由 $$M$$ 个线性注意力块和一个分组查询注意力 (Grouped Query Attention, GQA) 块构成。通过调整 $$M$$ 的值，可以灵活地改变模型中线性注意力与 Softmax 注意力的比例。
*   **训练-推理不对齐 (Training-Inference Disparity)**: 在强化学习（RL）中，由于训练框架（如 Megatron）和推理框架（如 vLLM）对同一操作（如 RMSNorm）的实现存在细微差异，导致模型在训练和推理（生成）时产生不同的输出。这种差异会逐层累积，在长序列生成和 MoE 架构中尤为严重，最终破坏 RL 训练的稳定性。

# 相关工作
当前，大语言模型的能力通过增加解码 Tokens 数量 (Test-Time Scaling) 得到了显著提升，同时，智能体 (Agent) 系统和代码生成等应用对长上下文处理能力提出了迫切需求。

然而，主流模型普遍采用的注意力机制（如 MHA, GQA）是主要瓶颈。其计算复杂度随序列长度呈二次方增长，而 I/O 开销（KV 缓存）随输出长度线性增长。这些限制严重阻碍了模型向更长上下文扩展。

为了解决这一问题，线性注意力应运而生，它将计算复杂度降至线性，并将状态记忆空间复杂度降为常数。但纯线性注意力模型在实际应用中存在性能短板，其效率优势也仅在超长序列（>8K）下才变得明显，在主流预训练长度（4K-8K）下，结合 MoE 架构后效率增益有限。

本文旨在解决上述挑战，通过设计一种名为 Ring-linear 的混合架构，以平衡效率与性能，并对其训练和推理过程进行系统性优化，从而在长上下文场景下实现低成本和高性能。

# 本文方法
本文的核心是一种名为 Ring-linear 的混合架构，它结合了高度稀疏的 MoE 架构和混合线性注意力机制。本文推出了基于该架构的两个模型：Ring-mini-linear-2.0 (16B 参数) 和 Ring-flash-linear-2.0 (104B 参数)。

### 基础架构
Ring-linear 模型由 $$N$$ 个层组构成，每个层组包含 $$M$$ 个线性注意力块和一个 GQA（Softmax 注意力）块。该架构深度整合了激活率仅为 1/32 的稀疏 MoE 设计，并应用了多项先进技术，如无辅助损失的路由策略、多 Token 预测 (Multi-Token Prediction, MTP)、QK-Norm 和部分旋转位置编码 (Partial-RoPE) 等。

<img src="/images/2510.19338v1/x2.jpg" alt="Ring-linear 架构图" style="width:85%; max-width:450px; margin:auto; display:block;">

两个开源模型的具体配置如下表所示：


| 模型 | 总参数 (B) | 激活参数 (B) | 层数 | 头数 | 头维度 | 中间层维度 | 词表大小 | MoE 专家数 | M (线性层数) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Ring-mini-linear-2.0 | 16 | 1.6 (0.957) | 40 | 40 | 128 | 14336 | 151851 | 32 | 4 |
| Ring-flash-linear-2.0 | 104 | 7.4 (6.1) | 56 | 64 | 128 | 24576 | 151851 | 32 | 7 |

*注：括号中的激活参数为不含 Embedding 层的参数量。*

### 混合线性注意力

#### 创新点
本文没有采用纯线性注意力，而是设计了一种混合架构。通过将模型层划分为多个组，每组包含 $$M$$ 个线性注意层和 1 个 Softmax 注意力层，实现了效率和性能的平衡。

<img src="/images/2510.19338v1/x3.jpg" alt="混合线性注意力架构与线性注意力模块设计" style="width:85%; max-width:600px; margin:auto; display:block;">

实验中的扩展定律（Scaling Law）研究表明（见下图），混合线性架构的性能始终优于纯 Softmax 注意力架构。并且，对于更高的计算预算，采用更大的层组（即更大的 $$M$$ 值）效果更佳。最终，Ring-flash-linear-2.0 设 $$M=7$$，Ring-mini-linear-2.0 设 $$M=4$$。

<img src="/images/2510.19338v1/x4.jpg" alt="不同 M 值下的扩展定律曲线" style="width:85%; max-width:450px; margin:auto; display:block;">

#### 线性注意力的实现
本文采用 Lightning Attention 作为线性注意力的具体实现，其核心思想是利用递归形式进行计算。输出 $$o_t$$ 可以通过更新一个大小为 $$d x d$$ 的状态矩阵 $$kv_t$$ 来获得，该矩阵在整个生成过程中大小保持不变，从而实现了常数级的空间复杂度。
其递归形式如下：


{% raw %}$$
\begin{split}\textbf{k}\textbf{v}_{0}&=0\in\mathbb{R}^{d\times d},\\ \textbf{k}\textbf{v}_{t}&=\lambda\textbf{k}\textbf{v}_{t-1}+\textbf{k}_{t}^{\text{T}}\textbf{v}_{t},\\ \textbf{o}_{t}&=\textbf{q}_{t}(\textbf{k}\textbf{v}_{t}),\\ \end{split}
$${% endraw %}



#### 优点
1.  **解码成本低**：由于线性注意力的 KV 缓存/状态记忆是常数大小，整个模型的 KV 缓存大小不随序列长度显著增长。与 GQA 等方法相比，Ring-linear 在长序列解码时具有巨大的内存访问优势，从而提升了推理速度。

    <img src="/images/2510.19338v1/x5.jpg" alt="不同序列长度下各种注意力机制的 KV 缓存/状态内存访问大小对比" style="width:85%; max-width:600px; margin:auto; display:block;">

2.  **性能与效率兼得**：混合架构保留了 Softmax 注意力，弥补了纯线性注意力在某些任务（如检索）上的性能短板，同时通过大量使用线性注意力层，大幅降低了计算和 I/O 开销。

### 计算优化
为了充分发挥混合架构的优势，本文从训练和推理两个方面进行了深度优化，核心在于 GPU 算子（Kernel）的融合与定制。

<img src="/images/2510.19338v1/x6.jpg" alt="计算优化架构图" style="width:80%; max-width:300px; margin:auto; display:block;">

#### GPU 算子优化
通过将多个独立操作（如门控机制、路由器计算、QK Norm 等）融合成一个单一的 GPU 算子，显著减少了 GPU 内存的读写次数和计算延迟。这不仅降低了训练时的显存消耗，还允许使用更大的微批次大小 (micro-batch size)，从而提升了训练吞吐量。

#### FP8 训练优化
针对 FP8 混合精度训练，本文设计了创新的优化策略。
*   **融合量化**：将 BF16 到 FP8 的量化操作与前续的激活函数（如 SiLU）或归一化层算子相融合。这避免了中间结果的读写开销，对于 H800这类高性能 GPU 而言，减少了因量化带来的性能瓶颈。
*   **细粒度重计算**：在反向传播的重计算阶段，通过优化，使前向传递仅计算和输出后续计算所需的张量（如转置后的量化 $$x$$），避免了冗余计算。

<img src="/images/2510.19338v1/x7.jpg" alt="FP8 训练优化前后对比" style="width:85%; max-width:600px; margin:auto; display:block;">

### 模型训练策略

#### 继续预训练
为了节约成本，模型从已有的 Ling-base-2.0 模型初始化参数，然后进行两阶段的继续预训练：
1.  **能力恢复阶段**：使用 4K 上下文长度和与基础模型相同的语料库，训练 600B 至 1T 的 Tokens，以恢复模型的基础能力。
2.  **中程训练阶段**：逐步将上下文窗口从 4K 扩展到 128K，并增加高质量推理数据的比例，为后续微调做准备。

#### 后训练
包括监督微调（SFT）和强化学习（RL）。
*   **SFT**: 数据集侧重于数学、代码、科学等高难度推理任务，并涵盖通用知识、智能体任务等，以确保模型的全面能力。
*   **RL**: 本文发现并解决了RL训练崩溃的一个关键原因：**训练与推理的不对齐**。由于训练和推理框架对 RMSNorm、RoPE 等标准组件的实现有细微差异，导致输出不一致，特别是在 MoE 架构和长链推理 (Long-CoT) 场景下问题被放大。通过系统性地对齐训练和推理引擎，本文实现了长期稳定的 RL 训练。

# 实验结论
### 效率验证
*   **训练效率**: 经过优化的 FP8 训练框架显著提升了吞吐量。与基线相比，Ring-mini-linear-2.0 的训练吞吐量提升了 77%，Ring-flash-linear-2.0 提升了 57%。
*   **推理效率**:
    *   在**预填充 (Prefill)** 阶段，当上下文长度超过 8K 时，Ring-linear 的吞吐量开始超越纯 Softmax 注意力的 Ring-2.0 模型，在 128K 上下文时，吞吐量是基线密集模型的 8 倍以上。
    *   在**解码 (Decode)** 阶段，当生成长度超过 4K 时，Ring-linear 开始展现优势。在 64K 上下文时，其吞吐量是基线模型的 10 倍以上。
    *   本文还开发了首个支持树状掩码 (tree mask) 的线性注意力算子，为混合线性模型实现投机解码 (speculative decoding) 加速小批量推理提供了可能。

<img src="/images/2510.19338v1/x8.jpg" alt="Ring-mini-linear-2.0 推理吞吐量对比" style="width:85%; max-width:450px; margin:auto; display:block;">
<img src="/images/2510.19338v1/x9.jpg" alt="Ring-mini-linear-2.0 推理吞吐量对比(续)" style="width:85%; max-width:450px; margin:auto; display:block;">

<img src="/images/2510.19338v1/x10.jpg" alt="Ring-flash-linear-2.0 推理吞吐量对比" style="width:85%; max-width:450px; margin:auto; display:block;">
<img src="/images/2510.19338v1/x11.jpg" alt="Ring-flash-linear-2.0 推理吞吐量对比(续)" style="width:85%; max-width:450px; margin:auto; display:block;">

### 模型性能
*   **能力恢复**: 在继续预训练后，Ring-linear-base 模型在编码、数学、知识、NLU 等多项能力上恢复了原基础模型 98% 以上的性能，仅在推理和专业知识等少数任务上因知识遗忘问题有轻微下降。

<img src="/images/2510.19338v1/x12.jpg" alt="Ring-linear-base-2.0 各项能力恢复情况" style="width:85%; max-width:450px; margin:auto; display:block;">
<img src="/images/2510.19338v1/x13.jpg" alt="Ring-flash-linear-base-2.0 各项能力恢复情况" style="width:85%; max-width:450px; margin:auto; display:block;">

### 最终结论
本文成功设计并实现了一个名为 Ring-linear 的高效混合注意力架构。该架构通过结合线性和 Softmax 注意力，并辅以从底层算子到训练框架的全方位优化，实现了模型性能和效率的极佳平衡。实验结果表明，该系列模型在大幅降低长上下文推理成本（相比密集模型降低至 1/10）的同时，仍在多个复杂的推理基准测试中保持了顶尖（SOTA）性能，为构建经济高效的长上下文大语言模型提供了有效的途径。