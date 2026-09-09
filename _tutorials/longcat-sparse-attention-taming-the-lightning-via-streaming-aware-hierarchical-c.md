---
layout: default
title: "LongCat Sparse Attention：破解DSA瓶颈，百万上下文长文本训练与推理兼得"
description: "LongCat：其中，DeepSeek 提出的 DeepSeek Sparse Attention（DSA）凭借独立的轻量化检索器 Lightning Indexer，实现了精细的 Token 级动态检索，不仅在 DeepSeek-V3.2 与 GLM-5 等前沿开源模型中落地。"
arxiv_id: "2608.01662"
paper_published: "2026-08-03"
published_at: "2026-09-09T13:15:08.606362+08:00"
topics:
  - "推理"
  - "模型训练"
tags:
  - "推理"
  - "模型训练"
  - "AI论文解读"
related_tutorials:
  - "robostral-navigate"
  - "post-training-on-office-work-improves-software-engineering-a-behavioral-account-"
  - "frontis-ma1-training-an-ai4ai-model-towards-recursive-self-improvement-in-machin"
  - "echoverse-deep-evolving-environments-for-training-computer-use-agents-at-scale"
---

<p class="paper-original-title" lang="en">LongCat Sparse Attention: Taming the Lightning via Streaming-aware Hierarchical Cross-Layer Indexing</p>

在超长上下文大模型逐渐走向生产环境的今天，注意力机制的计算开销正在经历一场深刻的“瓶颈转移”。传统全注意力机制受制于二次复杂度 $\mathcal{O}(L^2)$，促使业内转向各类稀疏注意力方案。其中，DeepSeek 提出的 DeepSeek Sparse Attention（DSA）凭借独立的轻量化检索器 Lightning Indexer，实现了精细的 Token 级动态检索，不仅在 DeepSeek-V3.2 与 GLM-5 等前沿开源模型中落地，更在长文本建模上做到了近乎无损的精度。

> ArXiv URL：https://arxiv.org/abs/2608.01662

然而，当上下文长度进一步向 128K、甚至 1000K（1M Tokens）演进时，DSA 的系统级底层短板逐渐暴露。美团 LongCat 团队在深入剖析硬件执行特征后发现，DSA 的核心痛点不再是注意力本身的矩阵运算，而是动态稀疏索引引入的两大硬件系统级瓶颈：**输出非连续导致的显存带宽雪崩**，以及**检索器自身潜藏的 $\mathcal{O}(L^2)$ 计算与排序开销**。

为此，研究团队提出了软硬件协同设计的稀疏注意力架构——**LongCat Sparse Attention（LSA）**。该方案通过流感知索引（Streaming-Aware Indexing）、跨层复用索引（Cross-Layer Indexing）以及分层索引（Hierarchical Indexing）三大互补技术，不仅将长文本推理延迟大幅压低，更原生支持了 100 万 Token 上下文的端到端预训练，成功支撑起 1.6T 参数规模的 LongCat-2.0 基础模型。

### 剖析 DSA：稀疏注意力背后的两大硬件瓶颈

要理解 LSA 的创新，必须先看清楚 DSA 是如何在长序列下“跑慢”的。

DSA 在每一层包含两个核心算子：**Lightning Indexer（LI）** 和 **核心稀疏注意力（Sparse Flash Attention, SFA）**。LI 负责对历史前缀的所有 Token 计算相关性得分，并筛选出 Top-$K$ 个关键 Token；SFA 则只针对这 $K$ 个选出的候选 Token 进行标准的多头/多潜变量注意力计算。

直观上看，由于 $K$ 是一个远小于序列长度 $L$ 的固定常数（例如在 128K 长度下取 $K=2048$，稀疏度达 98.4%），SFA 的计算复杂度被成功约束在 $\mathcal{O}(LK)$。但经过深度的底层硬件 Profiling，研究人员揭示了两个严重的系统级瓶颈：

#### 瓶颈一：离散内存访问导致 HBM 带宽利用率跌至 4.5%

为了保证模型效果，DSA 采用完全动态的 Token 级细粒度选择。这意味着在推理的解码阶段，核心稀疏算子必须从高带宽显存（HBM）中非连续地抓取（Gather）散落分布的单个 KV 向量。

在现代 AI 加速芯片架构中，硬件并发依赖于显存请求的合并访问（Memory Coalescing）。一个硬件计算核心在理想合并状态下能够维持约 50 个在途缓存行（每行 512 字节），对应约 25.6 KB 的显存并发窗口。然而在 DSA 中，每次动态抓取的 BF16 格式 MLA 潜变量 KV 仅有 1152 字节，跨越 3 个缓存行。这就带来两个灾难性后果：首先，单次抓取仅利用了约 6% 的内存级并发能力；其次，这 3 个缓存行内部的数据打包效率仅有 75% 左右。两者叠加，导致硬件的实际有效 HBM 带宽利用率急剧暴跌至理论峰值的 **4.5%**（相当于吞吐缩减到仅剩二十二分之一）。

更糟糕的是反向传播阶段。训练过程中，梯度的写回需要基于相同的不连续 Token 索引执行 `scatter_add` 操作。由于不同计算核心处理的 Token 存在重合，随机离散的写入地址会引发高频的内存写入冲突（Write Conflicts），强行将高度并行的显存写入序列化，严重拖慢训练吞吐。

#### 瓶颈二：线性增长的检索器耗时反噬长文本性能

在自回归解码阶段，SFA 算子由于只计算固定预算 $K$ 个 Token，单步耗时几乎恒定；但 Lightning Indexer 却必须扫描全量前缀长度 $L$，带来严格的 $\mathcal{O}(L)$ 复杂度。

实测数据显示，在 Batch Size 为 4、BF16 精度、$K=2048$ 的配置下，当上下文从 4K 增长到 1024K 时，SFA 的层耗时基本稳定在 0.10 毫秒左右，而 Lightning Indexer 的单层耗时从 0.034 毫秒飙升至 0.930 毫秒，激增 27 倍。在 1024K 上下文时，**检索器耗时已经占到整个注意力层总延迟的 90%**。换言之，原本旨在节省算力的检索模块，在百万序列尺度下反倒成为了最沉重的计算负担；若在训练或 Prefill 阶段，检索器的全局计算复杂度仍然是实打实的 $\mathcal{O}(L^2)$。

### LSA 的核心解法：三位一体的协同优化框架

针对上述内存与计算双重痛点，LongCat Sparse Attention 没有推倒重来，而是构建了一套硬件友好且兼顾表现力的三级分治框架。

```

                    ┌────────────────────────────────────────────────────────┐

                    │               LongCat Sparse Attention (LSA)           │

                    └───────────────────────────┬────────────────────────────┘

                                                │

         ┌──────────────────────────────────────┼──────────────────────────────────────┐

         ▼                                      ▼                                      ▼

【Streaming-Aware Indexing】            【Cross-Layer Indexing】              【Hierarchical Indexing】

   · 显存访问模式重塑                     · 深度方向计算摊销                     · 粗粒度到细粒度分级筛选

   · 50% 预算切分为 Sink + 滑动窗口       · N 层分组，仅 Owner 层计算索引        · Block-level Page 粗筛

   · 50% 预算用于动态 Token 检索          · 跨层联合蒸馏对齐各层表征             · 候选 Page 内做局部精细评分

   · 恢复连续读写，释放 HBM 吞吐          · 索引计算开销直接降至 1/N             · 检索复杂度降至次线性 (推理免训)

```

#### 1. 流感知索引（Streaming-Aware Indexing, SI）：用结构化局部性换取硬件吞吐

DSA 之所以显存带宽低下，根本原因在于其赋予每个查询完全无序的索引挑选权。但从注意力机制的内在规律来看，这种全动态策略并不符合模型实际的激活习惯。

研究团队分析了密集注意力模型在处理长文本时的实际权重分布，验证了一个被称为“流式模式”（Streaming Pattern）的现象：无论序列多长，注意力汇聚点（Attention Sink，即开头的极少数 Token）以及紧邻查询位置的局部滑动窗口（Sliding Window Area, SWA），占据了绝大部分的注意力权重。实验统计显示，在超过 5K 长度的上下文场景中，这两部分固定的局部区域平均捕获了模型全层超过 **83%** 的注意力权重。

基于这一洞察，LSA 将总预算 $K$ 明确拆解为三部分：




{% raw %}$$ \mathcal{S}_{t} = \mathcal{S}_{\text{sink}} \cup \mathcal{S}_{\text{swa}} \cup \mathcal{S}_{\text{sparse}} $${% endraw %}



在典型配置中，固定前缀汇聚预算设为 $K_{\text{sink}}=16$，局部滑动窗口预算设为 $K_{\text{swa}}=1024$，动态精细检索预算 $K_{\text{sparse}}=1008$（总预算 $K=2048$）。

这一设计带来了显著的硬件效益与训练收益：

1. **显存连续访问与合并读写**：约 50% 的 KV Cache 变成了地址完全固定的连续内存块，底层算子可以以大块缓存行直接并发加载，从根本上缓解了 DSA 的内存访问不连续问题。

2. **检索候选空间实质性压缩**：Lightning Indexer 不再需要评估全量序列，只用在排除 Sink 和 SWA 后的中间区域执行打分，计算量天然下降。

3. **更健壮的蒸馏机制**：尽管在推理时 Indexer 仅检索中间区域，但在稀疏训练阶段，LSA 依然将 Sink 和 SWA 纳入蒸馏的监督信号中。由于这些高权重区域提供了极其丰富的结构化信息，模型在保留局部上下文的同时，对中间离散关键 Token 的辨别力反而更加敏锐。

#### 2. 跨层索引（Cross-Layer Indexing, CLI）：跨层相关性与跨层蒸馏

如果说 SI 优化的是单层内的显存访存行为，那么 CLI 则直击检索器的层间冗余计算。

在深度神经网络中，相邻层之间的注意力表征具备强连续性。研究人员在测试中证实，相邻两层各自独立挑选的 Top-$K$ 集合，其 Token 重合率平均达到 57.4%。更关键的是，如果直接把上一层的索引集合借给下一层使用，能够覆盖目标层高达 **93.2%** 的注意力权重。

<img src="/images/2608.01662/fig_cli_overlap.webp" alt="层间注意力重要 Token 的重叠率及复用注意力质量" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

然而，如果在 DSA 中强行生搬硬套相邻层的索引，模型下游性能会出现不可忽视的退化，这是因为原始 Indexer 的打分参数是严格围绕本层特征进行梯度优化的。

CLI 的创新之处在于将 Transformer 按深度划分为大小为 $N$ 的层组（Group）。每组中仅由第一层作为“主导层”（Owner Layer）运行 Lightning Indexer，后续的 $N-1$ 个“复用层”（Reuse Layers）完全跳过索引计算，直接共享主导层的索引集合。为了让主导层的 Indexer 具备跨层统摄能力，LSA 重构了训练阶段的蒸馏损失函数：




{% raw %}$$ \mathcal{L}_{\text{CLI}} = \sum_{i=0}^{N-1} \mathcal{L}_{I}^{(l+i)} $${% endraw %}



在主导层训练时，其 Indexer 被施加联合蒸馏损失，强制要求其打分结果同时逼近当前组内所有 $N$ 个层各自的全注意力权重分布。通过这种训练期约束，单一检索器被塑造成能够提取整个层组“最大公约数”关键信息的全局筛选器。在常规设定 $N=2$ 时，模型的索引算力与显存交互开销直接腰斩，且几乎没有造成精度损失。

#### 3. 分层索引（Hierarchical Indexing, HI）：推理期免微调的粗到细裁剪

尽管 SI 与 CLI 已经在训练和推理阶段大幅剥离了计算负担，但在长达 1M 的极端序列推理中，中间候选区依然存在数十万 Token 需要检索。

为此，LSA 进一步设计了分层索引机制（HI）。HI 采用由粗到细的两阶段过滤策略，且完全属于**无需训练、即插即用（Training-free）**的推理加速方案：

* **粗粒度页面过滤**：将连续序列以固定大小 $P$ 切分为若干块（Pages）。首先利用轻量级的块级表征（Block-level Representation）计算查询与各 Page 的相关度，快速筛选出得分最高的 Top-$M$ 个候选 Page。

* **细粒度 Token 检索**：仅在选出的 $M$ 个 Page 内部执行原始的 Token 级 Indexer 评分与 Top-$K$ 截断。

通过这种分层结构，单查询的检索复杂度从原有的 $\mathcal{O}(L)$ 大幅降阶为 $\mathcal{O}(L/P + MP)$。由于候选空间在宏观上先经过了一道聚类性质的页面剪枝，底层矩阵计算规模锐减，在超长文本推理下能够提供可观的二次提速，且对长程下游任务的命中率扰动微乎其微。

### 实验与落地验证：从 69B 到 1.6T 模型的全面覆盖

为了验证 LSA 架构在不同规模和任务下的鲁棒性，研究团队依托 LongCat 系列模型开展了多梯度的系统性实验，覆盖了从中等参数的 LongCat-Flash-Lite（69B-A3B，总参数 690 亿、激活参数 30 亿）到大尺寸的 LongCat-Flash（560B-A27B），并最终全面应用于万亿级旗舰模型 LongCat-2.0（1.6T-A48B）。

#### 1. 全任务表现无损对齐全注意力

在基准评测中，引入 LSA 的模型不仅在常规常识推理、代码生成（如 HumanEval、MBPP）、数学推理（GSM8K、MATH）等通用任务上追平了全注意力基线（Full MLA），在包括 Needle In A Haystack（大海捞针）、L-Eval、InfiniteBench 在内的极端长文本检索与理解基准上同样展现出高度的一致性。


| 架构方案 | 上下文支持 | 关键评测质量（通用/长文本） | 显存带宽有效利用率 | 索引计算开销 |
| :--- | :--- | :--- | :--- | :--- |
| **标准全注意力 (MLA)** | 受限 ($\mathcal{O}(L^2)$ 爆炸) | 基准基线 (100%) | 良好 (连续访问) | 无需检索器 |
| **原始 DSA 方案** | 支持长文本 | 接近无损对齐 | 极低 ($\sim 4.5\%$) | 极高 (长序列占 90% 耗时) |
| **LongCat Sparse (LSA)** | **原生支持 100 万 Token** | **全面追平全注意力** | **显著提升 (恢复块读写)** | **骤降 (摊销至 $1/N$ 并分层)** |

在 560B 大规模 MoE 模型上，实验进一步证明，经过长文本继续预训练后，基于 LSA 构建的模型在保持推理经济性的同时，没有表现出任何灾难性遗忘或逻辑连贯性崩塌。

#### 2. 开源成果：LongCat-Flash-Lite-Sparse

为了让稀疏注意力的软硬件工程实践能被更广泛的社区使用，团队开源了 LongCat-Flash-Lite-Sparse（69B-A3B）。该模型原生训练支持 1M Token 上下文，在融合了多轮长文本微调语料后，不仅在极长文档解析与代码库级别 Agent 任务中表现出色，在端到端推理吞吐上相较于原版稠密模型实现了翻倍级别的性能飞跃。

### 总结与展望

DSA 揭示了独立检索器驱动稀疏注意力的巨大潜力，而 LongCat Sparse Attention（LSA）则代表着该技术路线从“算法可行”迈向“硬件高效”的关键跃升。

LSA 的三项核心机制针对不同层面的系统约束：流感知索引（SI）通过沉淀 50% 的确定性注意力窗口，换取了底层硬件渴求的连续内存访问吞吐；跨层联合蒸馏索引（CLI）抓住了层间语义冗余，实现了多层对单一检索结果的安全摊销；分层索引（HI）则在推理侧构建了轻巧的粗细分级过滤管道。

这套软硬件协同方案成功支撑了 1.6 万亿参数规模长文本模型的低成本预训练与超长文本低延迟推理。它给大模型工程界提供了一个明确信号：在向百万乃至千万级上下文极限演进的道路上，仅仅降低算法的理论 FLOPs 远远不够，只有尊重底层内存架构、将访存局部性与注意力内在拓扑深度结合，稀疏化架构才能真正释放出长文本大模型的生产力价值。
