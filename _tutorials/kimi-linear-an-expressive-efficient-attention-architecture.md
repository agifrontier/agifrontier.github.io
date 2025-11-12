---
layout: default
title: "Kimi Linear: An Expressive, Efficient Attention Architecture"
---

## Kimi Linear: An Expressive, Efficient Attention Architecture

- **ArXiv URL**: http://arxiv.org/abs/2510.26692v2

随着大语言模型（LLM）日益成为强大的智能体，如何让它们在处理长任务时更高效，成了一个核心难题。

传统注意力机制，就像一个记忆力超群但效率不高的人。他能记住所有事情（上下文），但每次回忆（计算）时，都要把所有记忆翻一遍，这在上下文很长时，速度慢得让人无法忍受。这便是 **二次方时间复杂度** 和 **线性增长的键值缓存（KV cache）** 带来的瓶颈。

为了解决这个问题，研究者们提出了**线性注意力 (Linear Attention)**。它试图用一种更聪明的方式来压缩记忆，计算速度快，但代价是表达能力和准确性有所下降，尤其在短文本上也不如传统方法。

于是，一种折中的**混合架构 (Hybrid Architecture)** 诞生了。它把大部分计算交给高效的线性注意力，少数关键部分仍由强大的全注意力负责。但这就像一个委员会，虽然高效，却始终没能干出超越那个全能天才（全注意力模型）的业绩。

本文介绍的 **Kimi Linear** 架构，则旨在打破这一僵局。它不仅要实现高效率，更要在质量上全面超越全注意力。

<img src="/images/2510.26692v2/page_0_Figure_8.jpg" alt="Kimi Linear 性能与加速效果" style="width:90%; max-width:700px; margin:auto; display:block;">

### Kimi Delta Attention (KDA)：更精细的记忆管理者

Kimi Linear 的核心是一个名为 **Kimi Delta Attention (KDA)** 的全新线性注意力模块。KDA 是在 **门控增量网络 (Gated DeltaNet, GDN)** 基础上的重要改进。

要理解 KDA 的创新，可以打个比方。

传统的注意力机制，其记忆状态更新可以看作一个不断累加的过程：


{% raw %}$$ \mathbf{S}_t = \mathbf{S}_{t-1} + \boldsymbol{k}_t \boldsymbol{v}_t^\top $${% endraw %}


这就像把新知识不断堆到旧知识上。但记忆需要遗忘，否则信息就会冗余爆炸。

后来引入了遗忘机制，比如 GDN，它在更新记忆时会乘上一个遗忘系数 $ \alpha\_t $：


{% raw %}$$ \mathbf{S}_t = \alpha_t (\dots) \mathbf{S}_{t-1} + (\dots) $${% endraw %}


这个 $ \alpha\_t $ 是一个标量，对于一个注意力头内的所有信息，它只能“一刀切”地决定遗忘多少。这就像一个粗放的内存管理器，要么保留整个程序，要么关闭整个程序。

而 **KDA 的核心创新在于“细粒度门控 (fine-grained gating)”**。它将标量遗忘门 $ \alpha\_t $ 升级为一个对角矩阵 $$Diag(αt)$$：


{% raw %}$$ \mathbf{S}_t = \left(\mathbf{I} - \beta_t \mathbf{k}_t \mathbf{k}_t^\top\right) \text{Diag}\left(\alpha_t\right) \mathbf{S}_{t-1} + \beta_t \mathbf{k}_t \mathbf{v}_t^\top $${% endraw %}


这意味着记忆状态的**每一个特征通道 (channel) 都有自己独立的遗忘速率**。

这就像一个精细的内存管理器，它可以独立关闭浏览器里某个不用的标签页，而不是关闭整个浏览器。这种精细化控制，让模型能更有效地利用其有限的循环神经网络（RNN）式记忆，**准确遗忘不重要的信息，同时牢牢记住关键内容**。

<img src="/images/2510.26692v2/page_3_Figure_4.jpg" alt="KDA 示意图" style="width:90%; max-width:700px; margin:auto; display:block;">

### 硬件高效的并行算法

为了让这种精细的门控机制能在 GPU 上高效运行，KDA 采用了一种专门设计的 **块并行算法 (chunkwise-parallel algorithm)**。

它基于一种特殊的 **对角加低秩 (Diagonal-Plus-Low-Rank, DPLR)** 矩阵形式，相比通用的 DPLR 公式，KDA 的设计**大大减少了计算量**。

简单来说，它在计算一个数据块（chunk）时，可以将一系列复杂的矩阵变换压缩成一个紧凑的表示，同时保持数值稳定性。这使得 KDA 的算子效率比通用的 DPLR 实现**提升了约 100%**。

<img src="/images/2510.26692v2/page_4_Figure_7.jpg" alt="KDA 与 DPLR 速度对比" style="width:85%; max-width:450px; margin:auto; display:block;">

### Kimi Linear 整体架构

Kimi Linear 模型采用了混合架构，每 4 个注意力层中，就有 **3 个是 KDA 层**，**1 个是全注意力层**。这种 3:1 的均匀混合结构，既通过 KDA 层大幅降低了 KV 缓存（最高可达75%），又通过全注意力层保留了全局信息流动的能力。

<img src="/images/2510.26692v2/page_5_Figure_1.jpg" alt="Kimi Linear 模型架构图" style="width:85%; max-width:450px; margin:auto; display:block;">

此外，模型还使用了**无位置编码 (NoPE)** 的设计。这不仅简化了长上下文训练（无需调整 RoPE 基频等），还能在推理时将全注意力层转换为高效的**多查询注意力 (MQA)**。

### 实验效果

一系列严格的对比实验证明了 Kimi Linear 的优越性。

#### 合成任务与消融研究

在回文、多查询关联回忆（MQAR）等考验长程记忆的合成任务中，KDA 的收敛速度和最终精度都显著优于 GDN 和其他线性注意力方法，证实了细粒度门控的优势。

<img src="/images/2510.26692v2/page_8_Figure_5.jpg" alt="合成任务上的性能对比" style="width:85%; max-width:450px; margin:auto; display:block;">

#### 公平对决：1.4 万亿 Token 训练

本文进行了大规模的公平对比。Kimi Linear、全注意力基线（MLA）和混合 GDN 基线（GDN-H）使用完全相同的模型参数、训练流程和 1.4 万亿 Token 的数据进行预训练。

结果显示，Kimi Linear 在通用知识、数学推理、代码和中文等几乎所有评测项上，**都一致性地超越了两个基线模型**。


| 模型 | MMLU-Pro | GPQA-Diamond | HumanEval-Python | MBPP-Python |
| :--- | :--- | :--- | :--- | :--- |
| MLA | 64.91 | 39.5 | 45.1 | 63.8 |
| GDN-H | 65.17 | 39.4 | 44.5 | 62.4 |
| Kimi Linear | **66.45** | **40.4** | **45.7** | **65.3** |

这证明 Kimi Linear 不仅高效，在模型质量上也更胜一筹。

#### 扩展性与效率

扩展法则实验表明，在同等计算资源下，Kimi Linear 的性能损失比全注意力模型更低，这意味着它具有**更高的计算效率**。

在效率方面，Kimi Linear 的优势极为显著。
*   **预填充 (Prefilling) 阶段**：在 100 万 Token 的上下文长度下，预填充速度是全注意力的 **2.9 倍**。
*   **解码 (Decoding) 阶段**：在 100 万 Token 的上下文长度下，解码速度是全注意力的 **6 倍**，这主要得益于其极小的固定大小状态，避免了巨大的 KV 缓存。

<img src="/images/2510.26692v2/page_12_Figure_2.jpg" alt="不同上下文长度下的解码速度" style="width:85%; max-width:600px; margin:auto; display:block;">

这意味着在处理超长文本或进行多轮对话时，Kimi Linear 能提供更快的响应速度和更高的吞吐量。

### 方法的本质优势

#### KDA 作为可学习的位置编码

传统的 Transformer 需要专门的位置编码（如 RoPE）来感知序列顺序。而 KDA 中的门控增量法则，其本身就包含了一种**数据依赖、可学习的位置编码**功能。与 RoPE 固定的旋转频率不同，KDA 的细粒度门控可以为不同特征维度动态调整“位置感”，提供了更大的灵活性。

#### 与稀疏注意力的对比

当前提升效率的路线主要有两条：线性注意力和稀疏注意力。
*   **稀疏注意力**：从完整的 KV 缓存中挑选关键信息，检索能力强，但仍需存储全部缓存，效率受限。
*   **线性注意力**：将历史信息压缩成一个固定大小的状态，理论上表达能力更强，更符合“压缩即智能”的理念。

KDA 通过细粒度门控缓解了传统线性注意力检索能力弱的短板，同时保持了其在效率上的巨大优势。

### 总结

本文提出的 Kimi Linear 是一种创新的混合注意力架构。其核心模块 KDA 通过**细粒度的门控机制**和**硬件高效的并行算法**，实现了表达能力和计算效率的统一。

在一系列严格的公平对比中，Kimi Linear 首次证明了混合架构**可以在各种场景下（短上下文、长上下文、强化学习）全面超越强大的全注意力基线**。

它不仅模型性能更优，还将 KV 缓存使用量减少了高达 75%，并在百万级长文本解码中实现了高达 6 倍的吞吐量提升。这些结果表明，Kimi Linear 可以作为现有全注意力架构的**直接替代品 (drop-in replacement)**，为下一代智能体模型提供了更优越、更高效的基础。