---
layout: default
title: "Efficient Memory Management for Large Language Model Serving with PagedAttention"
---

# Efficient Memory Management for Large Language Model Serving with PagedAttention

- **ArXiv URL**: http://arxiv.org/abs/2309.06180v1

- **作者**: Siyuan Zhuang; Ying Sheng; Woosuk Kwon; Haotong Zhang; Joseph E. Gonzalez; Cody Hao Yu; Lianmin Zheng; Zhuohan Li; Ion Stoica

- **发布机构**: Independent Researcher; Stanford University; UC San Diego; University of California, Berkeley

---

# TL;DR
本文提出 PagedAttention，一种受操作系统虚拟内存和分页机制启发的注意力算法，通过将键值缓存（KV Cache）划分为非连续的内存块进行管理，解决了大型语言模型（LLM）服务中因内存碎片化和无法共享而导致的效率低下问题，从而在 vLLM 系统中实现了 2-4 倍的吞吐量提升。

# 关键定义
本文的核心是围绕一种全新的内存管理方法展开的，关键定义如下：

*   **PagedAttention**: 一种创新的注意力算法。它允许模型在计算注意力时，处理存储在**非连续**物理内存空间中的键（Key）和值（Value）向量。这是与传统注意力算法最本质的区别，后者要求KV缓存必须存储在连续的内存块中。

*   **KV 块 (KV Block)**: PagedAttention 管理内存的基本单位。每个序列的 KV 缓存被分割成固定大小的块，每个块包含固定数量 Token 的键和值向量。这类似于操作系统中的“页”（Page）。

*   **块表 (Block Table)**: 用于实现逻辑内存到物理内存映射的数据结构。每个序列都拥有一个块表，它记录了该序列的逻辑 KV 块到 GPU 物理内存中实际 KV 块的映射关系。这类似于操作系统中的“页表”（Page Table）。

# 相关工作
当前的大型语言模型（LLM）服务严重受限于 GPU 内存，尤其是用于存储上下文信息的键值缓存（KV Cache）。尽管现有系统如 Orca 采用了迭代级调度等细粒度批处理技术来提升效率，但它们在内存管理方面仍存在严重瓶瓶颈。

现有系统的核心问题在于，它们要求每个请求的 KV 缓存必须存储在**连续的内存空间**中。由于 LLM 生成的输出长度事先未知，系统不得不为每个请求预先分配其可能达到的最大长度的内存块。这种策略导致了以下两大问题：

1.  **严重的内存浪费**：
    *   **内部碎片化 (Internal Fragmentation)**：当实际生成的序列长度远小于预分配的最大长度时，大量内存被闲置浪费。
    - **外部碎片化 (External Fragmentation)**：由于不同请求的预分配块大小不一，内存中会产生许多无法被利用的小碎片空间。
    *   根据本文的测试（如下图），在现有系统中，高达 60% 至 80% 的 KV 缓存内存被浪费，实际利用率极低。

    <img src="/images/2309.06180v1/page_1_Figure_0.jpg" alt="不同 LLM 服务系统中的平均内存浪费百分比" style="width:85%; max-width:600px; margin:auto; display:block;">
    <center>图 2. 不同 LLM 服务系统中的平均内存浪费百分比。</center>

2.  **无法实现内存共享**：在并行采样（为同一提示生成多个输出）或束搜索（Beam Search）等场景下，不同生成序列之间存在大量可共享的 KV 缓存（如提示部分）。但由于连续内存分配的限制，每个序列的 KV 缓存被隔离在独立的内存区域，无法实现共享，造成了冗余存储。

本文旨在解决上述由连续内存分配策略导致的内存碎片化和共享难题，以提高内存利用率，从而增大服务批次大小（Batch Size）并提升系统总吞吐量。

# 本文方法
为了解决上述内存管理挑战，本文提出了 PagedAttention 算法，并基于此构建了 vLLM 服务引擎。其核心思想借鉴了操作系统中经典的虚拟内存和分页技术。

<img src="/images/2309.06180v1/page_4_Figure_0.jpg" alt="vLLM 系统概览" style="width:85%; max-width:600px; margin:auto; display:block;">
<center>图 4. vLLM 系统概览。</center>

### PagedAttention 算法

PagedAttention 是 vLLM 的核心。与传统注意力机制不同，它将每个序列的 KV 缓存分割成固定大小的 **KV 块（KV Blocks）**。这些块在物理内存中无需连续存放。在执行注意力计算时，PagedAttention 核函数（Kernel）通过查询一个名为 **块表（Block Table）** 的映射结构，来定位并获取所需的 KV 块。这个过程类似于 CPU 通过页表访问物理内存。

<img src="/images/2309.06180v1/page_4_Figure_9.jpg" alt="PagedAttention 算法图示" style="width:85%; max-width:600px; margin:auto; display:block;">
<center>图 5. PagedAttention 算法图示，注意力键值向量存储在非连续的内存块中。</center>

这种设计将序列的逻辑内存布局与物理内存布局解耦，带来了极大的灵活性。

### vLLM 内存管理机制

vLLM 的 KV 缓存管理器利用 PagedAttention 实现了类似操作系统虚拟内存的管理方式：
*   **逻辑块与物理块**：每个请求的 KV 缓存被视为一连串的**逻辑块**。vLLM 的内存管理器维护一个全局的**物理块**池。
*   **按需分配**：系统不再为请求预留最大长度的内存。而是在生成过程中，当一个逻辑块被填满时，才为其分配一个新的物理块，并更新块表。这极大地减少了内部碎片，将内存浪费限制在最后一个未被完全填满的块内。
*   **消除外部碎片**：由于所有物理块大小相同，内存池的管理变得非常简单，完全消除了外部碎片。

<img src="/images/2309.06180v1/page_5_Figure_0.jpg" alt="vLLM 中的块表转换过程" style="width:85%; max-width:600px; margin:auto; display:block;">
<center>图 6. vLLM 中的块表转换过程。</center>

如上图所示，一个包含7个 Token 的提示最初只分配了两个物理块（物理块7和1）。随着新 Token 的生成，当物理块1被填满后，系统才会分配一个新的物理块（物理块3），并更新其块表。

### 优点：灵活的内存共享

PagedAttention 的设计天然支持高效的内存共享，这是其相比现有系统的另一大优势。

*   **并行采样 (Parallel Sampling)**：当一个请求需要生成多个不同的输出序列时，这些序列共享相同的输入提示（Prompt）。vLLM 只会为该提示存储一份物理 KV 缓存，并让所有输出序列的块表都指向这组共享的物理块。当某个序列的生成内容开始偏离时，vLLM 采用**写时复制 (Copy-on-Write)** 机制，仅为需要修改的那个块复制一份新的物理块，从而以极低的成本实现分叉。

    <img src="/images/2309.06180v1/page_6_Figure_0.jpg" alt="并行采样示例" style="width:85%; max-width:600px; margin:auto; display:block;">
    <center>图 8. 并行采样示例。</center>

*   **束搜索 (Beam Search)**：束搜索中的内存共享模式更为复杂和动态。vLLM 通过为每个物理块维护一个**引用计数**来轻松应对。当多个候选束共享一个物理块时，其引用计数增加；当某个束被丢弃时，其引用的物理块的计数减少。当引用计数归零时，该物理块被释放。这种方式避免了传统实现中大量繁琐且低效的内存拷贝操作。

    <img src="/images/2309.06180v1/page_6_Figure_6.jpg" alt="束搜索示例" style="width:90%; max-width:700px; margin:auto; display:block;">
    <center>图 9. 束搜索示例。</center>

*   **共享前缀 (Shared Prefix)**：对于多个请求共用的系统提示（System Prompt），vLLM 可以预先计算并缓存其 KV 缓存。后续请求可以直接将它们的逻辑块映射到这些已缓存的物理块上，从而跳过对共享前缀的重复计算。

### 调度与分布式执行

*   **调度与抢占**：当 GPU 内存耗尽时，vLLM 需要抢占某些正在运行的请求。它采用一种“要么全有，要么全无”的策略，即对一个序列组（如一个束搜索请求中的所有候选束）进行整体换出。被抢占的序列可以通过两种方式恢复：
    1.  **交换 (Swapping)**：将其 KV 缓存从 GPU 内存交换到 CPU 内存。
    2.  **重计算 (Recomputation)**：直接丢弃其 KV 缓存，在调度回来时重新计算。

*   **分布式执行**：对于需要跨多个 GPU 部署的大模型，vLLM 使用一个中心化的调度器来管理全局的块表。调度器在每个推理步骤开始前，将包含输入 Token 和块表信息的控制消息广播给所有 GPU 工作节点。各节点根据收到的块表信息独立执行计算，无需在内存管理上进行同步，从而高效地支持了张量模型并行。

# 实验结论
本文在一系列实验中将 vLLM与两个业界领先的基准系统——FasterTransformer（高度优化的推理引擎）和 Orca（为吞吐量优化的服务系统）进行了对比。实验使用了 OPT 和 LLaMA 等多种尺寸的模型，并基于 ShareGPT（长对话）和 Alpaca（短指令）数据集构造了模拟负载。


| 模型尺寸 | 13B | 66B | 175B |
| :--- | :--- | :--- | :--- |
| GPUs | A100 | 4×A100 | 8×A100-80GB |
| 总 GPU 内存 | 40 GB | 160 GB | 640 GB |
| 参数大小 | 26 GB | 132 GB | 346 GB |
| 用于 KV 缓存的内存 | 12 GB | 21 GB | 264 GB |
| 最大 KV 缓存槽数 | 15.7K | 9.7K | 60.1K |

**关键实验结果总结如下：**

1.  **吞吐量显著提升**：vLLM 的吞吐量相比 FasterTransformer และ Orca 提升了 2-4 倍。在相同的延迟水平下，vLLM 能够承受比 Orca（Oracle，理论上限）高 1.7-2.7 倍的请求速率，比 FasterTransformer 高出最多 22 倍。

    <img src="/images/2309.06180v1/page_9_Figure_2.jpg" alt="在 ShareGPT 和 Alpaca 数据集上的单序列生成性能对比" style="width:85%; max-width:600px; margin:auto; display:block;">
    <center>图 12. 在 ShareGPT 和 Alpaca 数据集上的单序列生成性能对比</center>

2.  **长序列负载优势更明显**：在处理具有较长输入和输出的 ShareGPT 数据集时，vLLM 的性能优势比在处理短序列的 Alpaca 数据集时更为显著。这是因为长序列使得传统方法的内存浪费问题更加突出，而 PagedAttention 的高效管理能力在此场景下得到了充分发挥。

    <img src="/images/2309.06180v1/page_8_Figure_7.jpg" alt="ShareGPT 和 Alpaca 数据集的输入输出长度分布" style="width:85%; max-width:600px; margin:auto; display:block;">
    <center>图 11. ShareGPT 和 Alpaca 数据集的输入输出长度分布。</center>

3.  **对复杂解码算法效果更佳**：PagedAttention 的内存共享机制使其在处理并行采样和束搜索等复杂解码任务时，相比传统方法有更大的性能增益。

**最终结论：**
实验结果有力地证明，通过引入 PagedAttention 机制，vLLM 解决了 LLM 服务中长期存在的 KV 缓存内存管理难题。其近乎零浪费的内存利用率和灵活的共享能力，使得系统能够支持更大的批处理规模，从而显著提升了服务吞吐量，降低了单次请求的成本。这一方法为构建更高效、更经济的 LLM 服务提供了坚实的基础。