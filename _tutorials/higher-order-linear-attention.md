---
layout: default
title: "Higher-order Linear Attention"
---

# Higher-order Linear Attention

- **ArXiv URL**: http://arxiv.org/abs/2510.27258v1

- **作者**: Zhen Qin; Quanquan Gu; Yifan Zhang

- **发布机构**: Princeton University; University of California

---

# TL;DR
本文提出了一种名为高阶线性注意力 (Higher-order Linear Attention, HLA) 的新型注意力机制，它通过紧凑的前缀充分统计量 (prefix sufficient statistics) 实现了高阶交互，同时保持了线性时间复杂度和流式计算能力，从而在不牺牲表达能力的情况下解决了标准注意力机制的二次方复杂度瓶颈。

# 关键定义
*   **高阶线性注意力 (Higher-order Linear Attention, HLA)**: 一种泛化线性注意力的机制。它通过引入高阶交互（如二阶张量积）来增强模型的表达能力，同时通过将计算分解为多个低阶矩（例如，键向量外积的和），使其能够在每个时间步以线性时间复杂度和常数大小的状态进行流式计算。
*   **前缀摘要 (Prefix Summaries)**: 一组在序列处理过程中可以流式更新的统计量，每个时间步的更新成本不依赖于序列长度。对于二阶HLA，核心摘要包括键的二阶矩 $\mathbf{S}\_t^K$, 查询-值累加器 $\mathbf{C}\_t^{QV}$ 和查询质量 $\mathbf{m}\_t^Q$ 等。
*   **关联扫描 (Associative Scans)**: 一种并行计算技术，用于高效训练HLA模型。通过为HLA的状态更新定义一个满足结合律的二元操作（如幺半群或半直积），可以在数据块上并行执行扫描计算，其结果与串行循环完全相同，从而解决了循环神经网络训练效率低下的问题。
*   **非对称高阶线性注意力 (Asymmetric Higher-order Linear Attention, AHLA)**: HLA的一个变体，它计算非对称的级联乘积 $\mathbf{AAV}$（其中 $\mathbf{A}=\mathbf{Q}\mathbf{K}^{\top}$），与标准的对称形式 $\mathbf{AA}^{\top}\mathbf{V}$ 互补。AHLA同样支持严格的因果流式计算，并具有不同的计算成本和状态构成。

# 相关工作
现代大语言模型（LLMs）的基础是Transformer架构及其核心组件——缩放点积注意力 (scaled dot-product attention)。然而，其计算和内存复杂度随序列长度 $n$ 呈 $O(n^2)$ 增长，这严重制约了模型在长上下文场景下的应用。

为了解决这一瓶颈，研究领域涌现了多种高效的替代方案，包括线性注意力 (Linear Attention)、现代循环神经网络 (RNNs)、状态空间模型 (State Space Models, SSMs) 等。这些方法通常能实现线性时间复杂度和在推理时 $O(1)$ 的状态更新。然而，它们大多局限于一阶或基于核函数的近似，这可能限制了模型的表达能力。

本文旨在解决的核心问题是：如何设计一种既具备注意力机制那样的数据依赖和高阶交互能力，又能像现代循环架构一样实现高效流式计算和并行训练的机制。

# 本文方法
本文的核心是提出高阶线性注意力（HLA），它通过紧凑的前缀摘要实现了高阶交互的流式计算。

### 二阶HLA
作为基础，本文从二阶张量注意力出发：


{% raw %}$$
\mathbf{T}_{2} := (\mathbf{Q}\mathbf{K}^{\top})(\mathbf{Q}\mathbf{K}^{\top})^{\top} = \mathbf{Q}(\mathbf{K}^{\top}\mathbf{K})\mathbf{Q}^{\top} \in \mathbb{R}^{n \times n}
$${% endraw %}


其关键在于依赖于键的二阶矩 $\mathbf{K}^{\top}\mathbf{K}$。这启发了通过维护前缀摘要（prefix summaries）来进行流式计算。在时间步 $t$，维护以下摘要：
*   **键的二阶矩**: $\mathbf{S}\_{t}^{K} \coloneqq \sum\_{i\leq t}\mathbf{k}\_{i}\mathbf{k}\_{i}^{\top} \in \mathbb{R}^{d\times d}$
*   **查询-值累加器**: $\mathbf{C}\_{t}^{QV} \coloneqq \sum\_{i\leq t}\mathbf{q}\_{i}\mathbf{v}\_{i}^{\top} \in \mathbb{R}^{d\times d\_v}$
*   **查询质量**: $\mathbf{m}\_{t}^{Q} \coloneqq \sum\_{i\leq t}\mathbf{q}\_{i} \in \mathbb{R}^{d}$

这些摘要的更新成本为 $O(d^2 + d d\_v)$，与序列长度无关。

基于这些摘要，二阶HLA的输出（默认为非归一化形式）在时间步 $t$ 定义为：


{% raw %}$$
\mathbf{o}_{t} \coloneqq \mathbf{q}_{t}^{\top}\mathbf{S}_{t}^{K}\mathbf{C}_{t}^{QV}
$${% endraw %}


也可以进行归一化：


{% raw %}$$
\mathbf{o}_{t} = \frac{\mathbf{q}_{t}^{\top}\mathbf{S}_{t}^{K}\mathbf{C}_{t}^{QV}}{\mathbf{q}_{t}^{\top}\mathbf{S}_{t}^{K}\mathbf{m}_{t}^{Q}+\varepsilon}
$${% endraw %}


这里的 $\mathbf{S}\_t^K$ 充当了一个数据依赖的、可学习的度量矩阵，丰富了模型的表达能力。当设 $\mathbf{S}\_t^K = \mathbf{I}$ 时，该形式能够退化为一种线性注意力。

### 创新点1：通过扩展摘要实现因果遮蔽
标准的注意力机制需要在计算中应用因果遮蔽，以确保在自回归任务中，当前时间步的输出只依赖于过去的信息。在HLA中直接应用遮蔽会破坏计算的分解结构。

为了解决这个问题，本文引入了两个额外的扩展前缀摘要：


{% raw %}$$
\mathbf{G}_{t} \coloneqq \sum_{i\leq t}\left(\mathbf{k}_{i}\mathbf{k}_{i}^{\top}\right)\mathbf{C}_{i-1}^{QV} \in \mathbb{R}^{d\times d_v}
$${% endraw %}




{% raw %}$$
\mathbf{h}_{t} \coloneqq \sum_{i\leq t}\left(\mathbf{k}_{i}\mathbf{k}_{i}^{\top}\right)\mathbf{m}_{i-1}^{Q} \in \mathbb{R}^{d}
$${% endraw %}


通过这些修正项，严格因果的二阶HLA输出可以被精确地计算出来，而无需物化任何 $n \times n$ 的矩阵。例如，非归一化的因果输出为：


{% raw %}$$
\mathbf{o}_{t} = \mathbf{q}_{t}^{\top}(\mathbf{S}_{t}^{K}\mathbf{C}_{t}^{QV} - \mathbf{G}_{t})
$${% endraw %}


所有摘要（包括 $\mathbf{G}\_t$ 和 $\mathbf{h}\_t$）都支持常数时间的在线更新，保持了流式计算的效率。
*   **更新规则**:
    *   $\mathbf{G}\_{t} = \mathbf{G}\_{t-1}+\mathbf{k}\_{t}(\mathbf{k}\_{t}^{\top}\mathbf{C}\_{t-1}^{QV})$
    *   $\mathbf{h}\_{t} = \mathbf{h}\_{t-1}+\mathbf{k}\_{t}(\mathbf{k}\_{t}^{\top}\mathbf{m}\_{t-1}^{Q})$

### 创新点2：通过关联扫描实现并行训练
纯粹的循环模型在GPU上训练效率低下。为了实现高效的并行训练，本文为HLA的状态更新定义了一个关联操作符 $$⊕$$，并使用关联扫描（如Blelloch scan）来计算前缀和。

*   **无遮蔽情况**: 状态 $\mathcal{S}=(\mathbf{S},\mathbf{C},\mathbf{m})$ 的合并是简单的加法，构成一个幺半群 (monoid)。
    

    {% raw %}$$
    (\mathbf{S}_{A},\mathbf{C}_{A},\mathbf{m}_{A}) \oplus (\mathbf{S}_{B},\mathbf{C}_{B},\mathbf{m}_{B}) = (\mathbf{S}_{A}{+}\mathbf{S}_{B},\,\mathbf{C}_{A}{+}\mathbf{C}_{B},\,\mathbf{m}_{A}{+}\mathbf{m}_{B})
    $${% endraw %}


*   **有遮蔽情况**: 状态 $\mathcal{S}=(\mathbf{S},\mathbf{C},\mathbf{m},\mathbf{G},\mathbf{h})$ 的合并更为复杂，构成一个半直积 (semidirect product) 结构，因为需要考虑跨片段的交互项。
    

    {% raw %}$$
    \begin{aligned}
    (\mathbf{S}_{A}, \mathbf{C}_{A}, \mathbf{m}_{A}, \mathbf{G}_{A}, \mathbf{h}_{A}) &\oplus (\mathbf{S}_{B}, \mathbf{C}_{B}, \mathbf{m}_{B}, \mathbf{G}_{B}, \mathbf{h}_{B}) = \\
    \big(\mathbf{S}_{A}{+}\mathbf{S}_{B},\; \mathbf{C}_{A}{+}\mathbf{C}_{B},\; &\mathbf{m}_{A}{+}\mathbf{m}_{B},\; \mathbf{G}_{A}{+}\mathbf{G}_{B}+\mathbf{S}_{B}\mathbf{C}_{A},\; \mathbf{h}_{A}{+}\mathbf{h}_{B}+\mathbf{S}_{B}\mathbf{m}_{A}\big)
    \end{aligned}
    $${% endraw %}


该方法可以对序列分块，在块内和块间并行执行扫描，得到的激活值与串行循环完全相同，从而实现了高效且精确的并行训练。该框架同样可以扩展到带有指数衰减 $\gamma$ 的情况。

![Masked (Second Order) HLA with Within-Chunk Scan](https://raw.githubusercontent.com/wylAImoreira/img-bed/main/202405231718919.png)

### 非对称高阶线性注意力 (AHLA)
本文还提出了一种互补的变体，称为AHLA。它计算的是左级联积 $\mathbf{Q}(\mathbf{K}^\top\mathbf{Q})(\mathbf{K}^\top\mathbf{V})$，而不是HLA中的对称形式。AHLA同样支持流式计算和因果遮蔽，但使用了不同的前缀摘要，例如：
*   $\mathbf{P}\_{t}^{KV} \coloneqq \sum\_{j\leq t}\mathbf{k}\_{j}\mathbf{v}\_{j}^{\top}$
*   $\mathbf{E}\_{t} \coloneqq \sum\_{i\leq t}\mathbf{k}\_{i}\big(\mathbf{q}\_{i}^{\top}\mathbf{P}\_{i}^{KV}\big)$

其流式输出为 $\mathbf{o}\_{t}^{\textsc{AHLA}} = \mathbf{q}\_{t}^{\top}\mathbf{E}\_{t}$。AHLA的计算成本为 $O(d d\_v)$，在某些情况下比HLA更高效。

# 实验结论
本文主要聚焦于算法结构和理论推导，并未提供具体的实验结果或与其他模型的性能比较。

**总结**
本文给出了一个完整的、可扩展的注意力机制框架——高阶线性注意力（HLA）。其主要贡献和优势如下：
*   **表达能力强**: 通过引入二阶甚至更高阶的交互，HLA具备了比标准线性注意力更强的数据依赖混合能力。
*   **计算高效**: HLA在推理时具有线性的时间复杂度（二阶为 $O(d^2 + d d\_v)$）和 $O(1)$ 的状态更新成本，非常适合长上下文场景。
*   **严格因果**: 通过创新的扩展摘要，HLA能够在不牺牲流式计算效率的前提下，精确实现自回归任务所需的严格因果遮蔽。
*   **并行训练**: 借助关联扫描技术，HLA的训练可以被高效并行化，且其结果与串行计算完全一致，避免了近似反向传播带来的问题。

总而言之，HLA作为一个可直接替换标准注意力的构建模块，巧妙地融合了注意力机制的数据依赖加权特性与现代循环架构的高效率，为构建可扩展的长上下文语言模型提供了一个有力的、有原则的工具。