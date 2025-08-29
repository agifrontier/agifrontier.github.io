---
layout: default
title: "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
---

# Mamba: Linear-Time Sequence Modeling with Selective State Spaces

- **ArXiv URL**: http://arxiv.org/abs/2312.00752v2

- **作者**: Tri Dao; Albert Gu

- **发布机构**: Carnegie Mellon University; Princeton University

---

# TL;DR
本文提出了一种名为Mamba的新型序列建模架构，它通过引入选择性状态空间模型（Selective SSM），在保持线性时间复杂度的同时，实现了与Transformer相媲美的性能，尤其在处理长序列任务上表现出色。

# 关键定义
*   **结构化状态空间模型 (Structured State Space Models, SSM/S4)**：一类受经典状态空间模型启发的序列模型，可被视为循环神经网络(RNN)和卷积神经网络(CNN)的结合。它通过参数 $(\Delta, \mathbf{A}, \mathbf{B}, \mathbf{C})$ 将一维输入序列 $x(t)$ 映射到输出 $y(t)$。SSM可以高效地以循环或卷积两种模式进行计算。

*   **线性时不变性 (Linear Time Invariance, LTI)**：这是先前SSM的一个核心特性，意味着其动态参数（如 $\mathbf{A}, \mathbf{B}$）在所有时间步上都是固定的。该特性使得模型可以通过高效的卷积进行并行训练，但也限制了其根据输入内容动态调整行为的能力，使其难以处理需要内容感知（content-aware）的任务。

*   **选择性状态空间模型 (Selective State Space Models, S6)**：本文提出的核心概念，是对传统SSM的改进。其关键在于使SSM的某些参数，特别是步长参数 $\Delta$ 以及矩阵 $\mathbf{B}$ 和 $\mathbf{C}$，成为输入数据的函数。这打破了LTI特性，使模型能够根据当前输入 token “选择性地”传播或遗忘信息。

*   **选择机制 (Selection Mechanism)**：指让SSM参数依赖于输入数据的具体机制。该机制赋予模型根据上下文动态调整其内部状态更新和输出的能力，从而能够过滤无关信息、关注相关信息，解决了LTI模型的根本局限性。

*   **硬件感知的并行扫描算法 (Hardware-aware Parallel Scan)**：为解决选择性SSM因打破LTI而无法使用高效卷积计算的问题，本文设计的一种新算法。该算法利用GPU的内存层级结构（HBM和SRAM），通过内核融合(kernel fusion)、并行扫描(parallel scan)和重计算(recomputation)等技术，在循环模式下高效计算模型，避免了在主存中物化巨大的中间状态，从而实现了线性时间复杂度和高吞吐量。

# 相关工作
当前，作为大模型（Foundation Models）支柱的序列模型几乎完全由Transformer架构主导。Transformer的核心自注意力（self-attention）机制能够有效捕捉上下文中的密集信息，但其计算和内存复杂度随序列长度二次方增长 ($O(L^2)$)，这使其在处理长序列时效率低下，并限制了上下文窗口的大小。

为了克服这一瓶颈，研究界提出了许多亚二次方时间复杂度的架构，例如线性注意力、门控卷积以及早期的结构化状态空间模型（SSM）。然而，这些模型在语言等信息密集型模态上的表现始终未能与Transformer相媲美。

本文指出，这些高效模型的一个关键弱点是它们**缺乏基于内容进行推理的能力**。由于它们的线性时不变性（LTI）特性，模型无法根据输入内容选择性地关注或忽略信息。本文旨在解决这一具体问题：**创建一个既能拥有RNN/SSM的线性时间效率，又能达到Transformer强大性能的序列模型**。

# 本文方法
本文方法的核心是设计一个能够进行内容感知的选择性状态空间模型（Selective SSM），并为其开发一种高效的计算实现，最终将其整合到一个简洁而强大的Mamba架构中。

<img src="/images/2312.00752v2/x1.jpg" alt="Mamba概览" style="width:90%; max-width:700px; margin:auto; display:block;">
> 图1：上图展示了Mamba的核心思想。传统的结构化SSM（左）通过一个高维隐状态 $h$
将输入 $x$ 的每个通道独立映射到输出 $y$。为了计算效率，它们要求参数 $(\Delta, \mathbf{A}, \mathbf{B}, \mathbf{C})$ 在时间上是固定的（时不变），从而可以使用卷积来避免物化巨大的隐状态。Mamba的S6模型（右）引入了选择机制，使这些参数依赖于输入，从而打破了时不变性。为了解决由此带来的效率问题，本文设计了一种硬件感知的扫描算法，该算法只在GPU更快的内存层级（如SRAM）中物化扩展后的状态，从而保持了计算的高效性。

### 动机：选择作为一种压缩手段

序列建模的一个根本问题是如何将长上下文压缩成一个更小、更有效的状态。Transformer之所以有效但低效，是因为它几乎不压缩上下文（保留完整的KV缓存）。而传统的RNN效率高，但效果差，因为其固定大小的状态是信息瓶颈。

本文认为，构建强大序列模型的关键在于**选择性 (selectivity)**：即模型需具备根据上下文关注或过滤输入的能力。两个合成任务凸显了传统LTI模型的失败：
1.  **选择性复制任务 (Selective Copying)**：要求模型记住特定（有色）的 token 并忽略其他（白色）的 token。LTI模型由于其固定的动态性，无法区分哪些是需要记忆的。
2.  **归纳头任务 (Induction Heads)**：要求模型根据特定上下文提示给出答案。这需要模型理解上下文关系，而LTI模型的卷积核是静态的，无法处理这种动态变化的依赖关系。

<img src="/images/2312.00752v2/x2.jpg" alt="合成任务示意图" style="width:90%; max-width:700px; margin:auto; display:block;">
> 图2：选择性任务揭示了LTI模型的局限性。(左) 标准的复制任务中，输入-输出间距固定，LTI模型可轻松解决。(右上) 在选择性复制任务中，间距是随机的，需要时变模型才能选择性地记忆或忽略输入。(右下) 归纳头任务需要根据上下文进行关联回忆，这是大语言模型的关键能力。

这些任务表明，一个高效且强大的模型必须能够智能地选择哪些信息进入其状态。

### Mamba的核心：选择性状态空间 (Selective SSM)

为了赋予SSM选择能力，本文让其参数依赖于输入。传统的SSM（S4）和本文的选择性SSM（S6）的对比如下：


| SSM (S4) - 时不变                                       |
| :-------------------------------------------------------- |
| 1: $x:\mathtt{(B,L,D)}$                                     |
| 2: $\mathbf{A}:\mathtt{(D,N)}\leftarrow\mathsf{Parameter}$        |
| 3: $\mathbf{B}:\mathtt{(D,N)}\leftarrow\mathsf{Parameter}$        |
| 4: $\mathbf{C}:\mathtt{(D,N)}\leftarrow\mathsf{Parameter}$        |
| 5: $\Delta:\mathtt{(D)}\leftarrow\tau\_{\Delta}(\mathsf{Parameter})$ |
| 6: $\overline{\mathbf{A}},\overline{\mathbf{B}} \leftarrow\mathsf{discretize}(\Delta,\mathbf{A},\mathbf{B})$ |
| 7: $y\leftarrow\mathsf{SSM}(\overline{\mathbf{A}},\overline{\mathbf{B}},\mathbf{C})(x)$ $\triangleright$ 循环或卷积 |


| SSM + Selection (S6) - 时变                               |
| :-------------------------------------------------------- |
| 1: $x:\mathtt{(B,L,D)}$                                     |
| 2: $\mathbf{A}:\mathtt{(D,N)}\leftarrow\mathsf{Parameter}$        |
| 3: $\mathbf{B}:{\color[rgb]{0.72,0,0}\mathtt{(B,L,N)}}\leftarrow{\color[rgb]{0.72,0,0}s\_{B}(x)}$ |
| 4: $\mathbf{C}:{\color[rgb]{0.72,0,0}\mathtt{(B,L,N)}}\leftarrow{\color[rgb]{0.72,0,0}s\_{C}(x)}$ |
| 5: $\Delta:{\color[rgb]{0.72,0,0}\mathtt{(B,L,D)}}\leftarrow\tau\_{\Delta}(\mathsf{Parameter}{\color[rgb]{0.72,0,0}+s\_{\Delta}(x)})$ |
| 6: $\overline{\mathbf{A}},\overline{\mathbf{B}} \leftarrow\mathsf{discretize}(\Delta,\mathbf{A},\mathbf{B})$ |
| 7: $y\leftarrow\mathsf{SSM}(\overline{\mathbf{A}},\overline{\mathbf{B}},\mathbf{C})(x)$ $\triangleright$ 只能用循环（扫描） |

#### 创新点
核心创新在于让参数 $\Delta$ (步长)、$\mathbf{B}$ (输入矩阵) 和 $\mathbf{C}$ (输出矩阵) 成为输入 $x$ 的函数。


{% raw %}$$
\mathbf{B}_t = s_B(x_t), \quad \mathbf{C}_t = s_C(x_t), \quad \Delta_t = \tau_{\Delta}(\text{param} + s_{\Delta}(x_t))
$${% endraw %}


这使得模型从时不变（LTI）变为时变，从而具备选择性。
*   **$\Delta$ 的作用**：$\Delta$ 参数推广了RNN中的门控机制。一个大的 $\Delta$ 会重置隐状态 $h$ 并聚焦于当前输入 $x\_t$，而一个小的 $\Delta$ 则会保持原有状态并忽略当前输入。这允许模型动态决定是“记住”还是“跳过”每个 token。
*   **$\mathbf{B}$ 和 $\mathbf{C}$ 的作用**：选择性的 $\mathbf{B}$ 和 $\mathbf{C}$ 允许模型更精细地控制输入 $x\_t$ 如何影响状态 $h\_t$，以及状态 $h\_t$ 如何影响输出 $y\_t$。这可以被解释为基于内容（输入）和上下文（隐状态）来调节动态。

### 挑战与解决方案：硬件感知的并行扫描算法

选择性机制打破了LTI，使得模型无法再使用高效的卷积算法。如果采用传统的循环计算，需要物化一个巨大的隐状态 $h$（维度为 $B \times L \times D \times N$），这在内存和速度上都是不可接受的。

为了解决这个难题，本文设计了一种名为 **选择性扫描 (Selective Scan)** 的硬件感知算法，它结合了三种经典技术：
1.  **内核融合 (Kernel Fusion)**：避免在GPU的高带宽内存（HBM，较慢）中反复读写中间结果。算法将SSM参数和输入从HBM加载到SRAM（片上内存，极快）中，在SRAM内完成离散化和循环计算，最后仅将最终输出写回HBM。这极大地减少了内存I/O开销。
2.  **并行扫描 (Parallel Scan)**：虽然循环是顺序的，但本文采用了并行扫描算法，将其转化为一个可在GPU上高效并行执行的形式，解决了顺序计算的瓶颈。
3.  **重计算 (Recomputation)**：为了在反向传播时节省内存，算法不保存中间的隐状态。而是在需要时（反向传播期间）重新从输入计算它们。这使得Mamba的内存占用与经过优化的Transformer（如使用FlashAttention）相当。

这个算法使得选择性SSM在保持强大表达能力的同时，实现了线性时间复杂度 $O(BLD)$ 和极高的训练/推理速度。

### Mamba的简洁架构
本文将选择性SSM集成到一个简洁、同质化的**Mamba块**中，并用其构建整个网络。

<img src="/images/2312.00752v2/x3.jpg" alt="Mamba架构图" style="width:90%; max-width:700px; margin:auto; display:block;">
> 图3：Mamba块的设计。它将先前SSM架构中的H3块与Transformer中常见的MLP块进行了简化和合并。输入经过线性层扩展维度后，一路通过SiLU激活函数，另一路通过核心的S6层（选择性SSM）。两路结果逐元素相乘（门控），再通过线性层投影回原维度。整个Mamba模型由这种同质化的块堆叠而成。

这种设计比Transformer中交错的注意力块和MLP块更加简单和统一。Mamba块取代了注意力层和前馈网络（FFN）层，形成了一个单一、重复的结构单元。

# 实验结论

Mamba在多种模态和任务上都取得了卓越的成果，验证了其作为通用序列模型骨干的潜力。

*   **合成任务**：Mamba轻松解决了“选择性复制”和“归纳头”任务，并且能够将解决方案外推到超过1百万长度的序列，证明了其选择机制的有效性。

*   **真实世界数据**：
    *   **语言建模**：Mamba是**第一个达到Transformer级别性能的线性时间序列模型**。在预训练困惑度和下游任务（如常识推理）评估中，Mamba-3B全面超越了同等规模的Transformer，并与两倍于其大小的Transformer模型（如Pythia-7B）性能相当。
    *   **音频和基因组学**：Mamba在音频波形和DNA序列建模上均优于包括Hyena和Transformer在内的先前SOTA模型。更重要的是，它的性能随着上下文长度的增加而持续提升，甚至在百万长度的序列上也是如此，这证明了其处理超长序列的能力。

*   **效率**：
    *   **推理速度**：Mamba的推理吞吐量是同等规模Transformer的**5倍**。这是因为它具有RNN的特性，生成新 token 只需要 $O(1)$ 的时间，而Transformer需要 $O(L)$。
    *   **训练效率**：训练过程中的计算和内存开销随序列长度线性增长，这使得训练极长序列成为可能。

*   **模型消融**：实验表明，选择机制（特别是对 $\Delta$, $\mathbf{B}$, $\mathbf{C}$ 的选择性）是Mamba在语言等信息密集型数据上取得成功的关键。

**总结**：Mamba通过选择性状态空间模型，成功地将RNN的效率与Transformer的性能结合起来。它在性能、效率和长上下文处理能力方面展现出巨大优势，为大模型提供了一个极具吸引力的替代方案，有潜力成为下一代序列建模的基础架构。