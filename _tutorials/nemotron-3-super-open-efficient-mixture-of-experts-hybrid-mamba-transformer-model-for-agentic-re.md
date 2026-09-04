---
layout: default
title: "Nemotron 3 Super开源：提速7.5倍的120B混合架构MoE解读"
description: "大模型推理成本高昂，长文本处理更是内存“吞金兽”。如何在保证千亿参数级别强大推理能力的同时，将推理吞吐量提升数倍？NVIDIA最近开源的Nemotron3Super交出了一份惊艳的答卷。这款总参数量达120B（激活参数仅12B）的模型，不仅原生支持高达100万的上下文窗口，更在复杂推理场景下实现了越级碾压。"
arxiv_id: "2604.12374"
topics:
  - "模型优化"
  - "基础模型"
tags:
  - "LatentMoE"
  - "MTP layers"
  - "Mamba-Attention"
  - "Mixture-of-Experts"
  - "NVFP4"
  - "Nemotron 3 Super"
related_tutorials:
  - "every-attention-matters-an-efficient-hybrid-architecture-for-long-context-reason"
  - "rethinking-supervised-fine-tuning-emphasizing-key-answer-tokens-for-improved-llm"
  - "kimi-k2-open-agentic-intelligence"
  - "introspective-diffusion-language-models"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">Nemotron 3 Super: Open, Efficient Mixture-of-Experts Hybrid Mamba-Transformer Model for Agentic Reasoning</p>

大模型推理成本高昂，长文本处理更是内存“吞金兽”。如何在保证千亿参数级别强大推理能力的同时，将推理吞吐量提升数倍？

> **ArXiv URL**：http://arxiv.org/abs/2604.12374v1

NVIDIA最近开源的Nemotron 3 Super交出了一份惊艳的答卷。这款总参数量达120B（激活参数仅12B）的模型，不仅原生支持高达100万的上下文窗口，更在复杂推理场景下实现了越级碾压。

在8k输入与64k输出的实测场景下，该模型实现了比同级别模型高出7.5倍的吞吐量。该研究是如何打破性能与效率的平衡魔咒的？答案藏在其独特的架构创新与训练哲学中。

### 核心架构创新

本文的核心亮点在于三种前沿架构的深度融合。研究团队并未盲目堆砌参数，而是从硬件底层的痛点出发重构了网络。

#### LatentMoE机制

`**混合专家网络**（**MoE**）`虽然能在不增加计算量的情况下扩大参数，但传统的MoE在实际部署时，往往会遭遇内存带宽和节点间通信的严重瓶颈。

传统的MoE就像是把公司里最原始、最厚重的文件（完整维度特征）原封不动地搬运给各个业务专家。这不仅搬运成本极高，专家处理起来也颇为费力。

该研究引入了创新的LatentMoE架构。如图所示，输入Token $x$ 首先通过一个可学习的降维矩阵，被压缩到一个低维的“潜在空间”。

<img src="/images/2604.12374v1/latent_moe.webp" alt="Refer to caption" style="width:80%; max-width:300px; margin:auto; display:block;">

这个过程好比先将厚重的文件浓缩成一页纸的“核心摘要”，然后再分发给数量更多、分工更细的专家进行处理。

专家们在低维空间内完成计算后，再通过升维矩阵将结果还原回原始维度 $d$。这种设计大幅降低了通信负载与权重读取开销。

通过节省下来的开销，模型能在相同的计算预算下，激活更多的专家。这种“降维打击”在参数和算力双重维度上实现了极致的精度收益。为了保住模型下限，路由门控等关键计算仍保留在全维度进行。

#### 混合Mamba-Attention模式

在处理长文本时，Transformer架构的KV Cache呈二次方增长，这是制约大并发吞吐量的核心痛点。

本文采用了混合交织的架构设计。模型的大部分层使用了Mamba-2块。Mamba像是一条高效的流水线，在生成时维持恒定大小的状态，彻底消除了内存的爆炸式增长。

而Attention层则被作为“全局锚点”稀疏地穿插其中，负责在关键节点回顾全局信息，确保长程依赖不丢失。两者完美互补。

#### 多Token预测

传统的自回归生成就像是走一步看一步，而Nemotron 3 Super引入了`**多Token预测**（**Multi-Token Prediction, MTP**）`机制。

模型在训练时被要求同时预测未来多个位置的Token。这不仅促使模型学习到更长远的结构依赖，还解锁了原生的推测解码能力。

区别于需要外部草稿模型的方案，本文设计了一个共享权重的预测头。在推理时，这个“内部草稿员”可以极其轻量地生成长篇候选序列。

由于采用了共享权重，预测头在生成多步内容时不会出现严重的分布偏移现象，最终交由主模型一次性验证即可。在SPEED-Bench测试中，该设计展现了极高的接受率。

<img src="/images/2604.12374v1/mtp_trtllm_perf.webp" alt="Refer to caption" style="width:85%; max-width:450px; margin:auto; display:block;">

### 极致的预训练工程

该模型的预训练规模达到了惊人的25万亿Token，分两个阶段进行。前80%注重广度，后20%聚焦高质量数据。

最引人瞩目的是，它是首个在NVFP4极低精度下完成预训练的模型。这证明了在硬件友好的低精度格式下，依然能实现稳定且高精度的模型收敛。

数据层面，研究团队投入了巨大精力构建合成数据集。例如在代码领域，利用大模型从基础基准中提取抽象概念，再反向生成百万级的问答对，并经过严格的抽象语法树校验。

该研究还探讨了`**权重空间合并**（**Weight-Space Merging**）`技术。通过对滑动窗口内的检查点进行平均，可以模拟学习率衰减的效果，从而节省了约16%的算力开销。

### 面向Agent的后训练

Nemotron 3 Super的后训练阶段极具特色，重点强化了智能体的长程推理与工具调用能力。

<img src="/images/2604.12374v1/x11.webp" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">

#### 双阶段监督微调

在`**监督微调**（**SFT**）`阶段，该研究发现单阶段训练会导致长输入短输出的任务性能严重退化。

因此，本文设计了精巧的双阶段损失函数。第一阶段采用Token级别的全局平均损失：


{% raw %}$$ \mathcal{L}_{\text{tok}}=\frac{\sum\limits_{c\in\mathcal{B}}\sum\limits_{t\in\mathcal{O}_{c}}\ell_{t}}{\sum\limits_{c\in\mathcal{B}}|\mathcal{O}_{c}|} $${% endraw %}


这诱导模型在长篇推理过程中不遗漏任何细节。而第二阶段则切换为样本级别的归一化损失，防止超长输出的样本主导模型梯度更新方向，完美恢复了常规指令遵循的能力。

#### 三阶段强化学习

强化学习阶段被拆分为三个核心步骤，全面提升模型的自主决策能力。

首先是多环境的`**基于可验证奖励的强化学习**（**RLVR**）`，并行覆盖了数学、代码、安全等21个环境。研究发现，跨环境联合训练能有效防止灾难性遗忘。

值得一提的是，该阶段引入了“低资源推理”模式。通过将奖励与输出Token数挂钩，模型学会了在保证正确率的前提下，尽量缩短思考过程。

其次是针对软件工程的端到端强化学习（SWE-RL）。模型会在真实的Apptainer容器环境中，化身OpenHands智能体，自主修复代码漏洞并跑通测试用例。

最后，团队训练了一个遵循特定原则的生成式奖励模型，完成了高质量的RLHF，确保模型在复杂Agent交互中依然符合人类的价值观。

### 实验评估与性能霸榜

在严苛的基准测试中，Nemotron 3 Super的基础模型在各类学科与代码测试上，显著优于同等规模的GLM-4.5-Air-Base等先进开源模型。

而在最能体现商业价值的推理吞吐量上，借助TRT-LLM和vLLM等推理框架，其在8k输入/64k输出场景下的表现令人惊叹。

实测数据显示，其吞吐量分别达到了GPT-OSS-120B的2.2倍和Qwen3.5-122B的7.5倍，彻底改变了千亿参数模型的部署经济学。


### 工程启示

Nemotron 3 Super的全面开源，不仅提供了权重，更公开了详尽的训练配方与数据集。这为工业界带来了深刻的启示：

首先，LatentMoE证明了在设计稀疏架构时，不能仅看理论的FLOPs计算量，必须将内存带宽和通信开销作为核心优化目标。

其次，Mamba与Attention的混合架构结合原生的推测解码，正成为解决长文本Agent超低延迟落地的标准答案。

最后，大规模的多环境RLVR与真实的沙盒环境闭环反馈，已经成为突破模型自主决策能力的必由之路。纯粹依赖人类标注的时代正在远去。
