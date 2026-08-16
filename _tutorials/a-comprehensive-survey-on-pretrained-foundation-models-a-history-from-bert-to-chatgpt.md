---
layout: default
title: "千亿参数的基石：预训练基础模型(PFMs)跨模态演进全解析"
description: "深度学习的演进史上，很少有技术能像如今的大模型这样，在短短几年内彻底重塑整个行业的底层逻辑。过去，研究人员习惯于针对单一任务从零开始训练模型；而现在，整个AI界都在拥抱一种全新的范式：“预训练+微调”。"
arxiv_id: "2302.09419"
topics:
  - "基础模型与理论"
  - "多模态与视觉"
tags:
  - "Artificial General Intelligence"
  - "Autoregressive Language Models"
  - "BERT"
  - "ChatGPT"
  - "GPT"
  - "Model Efficiency"
related_tutorials:
  - "rlhf-a-comprehensive-survey-for-cultural-multimodal-and-low-latency-alignment-me"
  - "vision-mamba-efficient-visual-representation-learning-with-bidirectional-state-s"
  - "toward-general-purpose-robots-via-foundation-models-a-survey-and-meta-analysis"
  - "personal-llm-agents-insights-and-survey-about-the-capability-efficiency-and-security"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">A Comprehensive Survey on Pretrained Foundation Models: A History from BERT to ChatGPT</p>

深度学习的演进史上，很少有技术能像如今的大模型这样，在短短几年内彻底重塑整个行业的底层逻辑。

> **ArXiv URL**：http://arxiv.org/abs/2302.09419v3

过去，研究人员习惯于针对单一任务从零开始训练模型；而现在，整个AI界都在拥抱一种全新的范式：“预训练+微调”。在这场变革的核心，正是本文要深入探讨的主角——`**预训练基础模型**（**Pretrained Foundation Models, PFMs**）`。

从最早引发轰动的BERT，到如今展现出惊人通用推理能力的ChatGPT，PFMs已经跨越了文本的边界，全面渗透到计算机视觉、图学习等多个数据模态中。该研究系统性地梳理了PFMs的发展脉络与核心机制。本文将带您深入剖析这些千亿参数巨兽背后的技术真相。

<img src="/images/2302.09419v3/page_4_Figure_0.jpg" alt="PFMs Evolution" style="width:90%; max-width:700px; margin:auto; display:block;">

### 架构基石：Transformer与学习机制

PFMs之所以能够拥有处理海量数据的能力，其底层的网络架构起到了决定性作用。在自然语言处理与计算机视觉领域，Transformer已经成为绝对的主流。

Transformer摒弃了传统的循环与卷积结构，完全依赖注意力机制。通过自注意力，模型能够在同一序列的不同位置之间建立联系。这种设计不仅解决了处理长序列时的长距离依赖问题，更重要的是，它天然适合大规模并行计算。正是这种极高的并行度，使得模型参数量从BERT的几亿，飙升至GPT-3的1750亿，甚至PaLM的5400亿。

#### “通识教育”与“专业进修”

为了理解PFMs的学习机制，我们可以引入一个延续性的概念：**“通识教育”与“专业进修”**。

预训练阶段，就如同让模型接受无所不包的“通识教育”。在这个阶段，模型不需要应对特定的考试（下游任务），而是通过海量的无标签数据，学习数据中蕴含的通用规律与内在结构。在这里，自监督学习成为了最核心的手段。

通过精心设计的预训练任务，模型需要最小化预测值与真实数据之间的差异。一个典型的损失函数可以表示为：


{% raw %}$$ \mathop{\rm arg\,min}_{\boldsymbol{\Theta}} \frac{1}{n} \sum_{i=1}^{n} \mathcal{L}(f(\boldsymbol{x}_{i}; \boldsymbol{\theta}), \boldsymbol{y}_{i}) + \lambda \Omega(\boldsymbol{\theta}) $${% endraw %}



完成了“通识教育”后，模型就具备了极强的基础表征能力。接下来只需极少量的有标签数据，通过微调（“专业进修”），就能在分类、生成等具体任务上大放异彩。

此外，强化学习也被引入到PFMs的训练中。智能体通过与环境的交互，以最大化长期累积奖励为目标，这为后续的指令对齐打下了数学基础：


{% raw %}$$ \max_{\theta} \mathbb{E}_{s_t}[R_t | s_t, a_t = \pi_\theta(s_t)] $${% endraw %}



### 语言模型：从填空题到逻辑推理

自然语言处理是PFMs最先爆发的阵地。现有的预训练语言模型主要分为自回归模型、上下文模型和排列语言模型。

BERT代表了上下文模型的巅峰，它就像是一个做“完形填空”的高手。通过掩码语言模型机制，BERT在输入序列中随机遮盖一些词，然后让模型根据双向上下文去预测这些词。这种双向机制极大地增强了模型对句法和语义的深层理解。

然而，像ChatGPT这样的生成式大语言模型，则走的是自回归路线（如GPT系列）。它们从左到右逐字预测下一个词。为了让这些模型不仅能“接话”，还能听懂人类的复杂指令，研究者引入了指令对齐方法。

#### 强化学习与思维链

指令对齐的核心目标是让模型遵循人类意图，生成有用且无害的内容。除了使用高质量人工标注数据进行监督微调外，基于人类反馈的强化学习成为了ChatGPT成功的关键。

更令人惊叹的是，这些模型展现出了`**思维链**（**Chain-of-Thought, CoT**）`推理能力。通过在提示词中引导模型展示中间推理步骤，例如先计算“农民剩下70只鸡”，再计算“剩下10头猪”，最后得出总数，大语言模型在算术和符号推理任务上的性能得到了极其显著的提升。

### 视觉模型：重建与对比的博弈

随着PFMs在文本领域的巨大成功，研究人员迅速将目光转向了`**计算机视觉**（**Computer Vision, CV**）`。

在视觉领域，获取高质量标注图片的成本极高。因此，基于自监督学习的预训练成为了破局的关键。本文总结了多种视觉预训练范式，其中最具代表性的是“基于重建”与“基于对比”的方法。

<img src="/images/2302.09419v3/page_21_Figure_0.jpg" alt="Visual Reconstruction" style="width:90%; max-width:700px; margin:auto; display:block;">

#### 掩码图像建模

受到NLP中掩码预训练的启发，视觉领域也引入了掩码图像建模。例如BEiT和MAE模型，它们将图像切割成一个个图块（Patch）。在训练时，模型会随机掩码掉大比例（如75%）的图块，然后要求网络仅凭剩下的少量图块，重建出原本的完整图像。

这种极具挑战性的“通识教育”任务，迫使视觉模型放弃学习浅层的像素级快捷方式，转而深度理解图像的全局语义和空间结构。为了解决计算效率问题，研究者还提出了局部掩码重建等改进方案，让模型仅关注局部窗口，大幅提升了训练效率。

#### 对比学习机制

另一种主流方法是对比学习。它的核心逻辑非常直观：将同一张图片的不同增强版本（如裁剪、变色）视为“正样本对”，将不同的图片视为“负样本对”。

模型在特征空间中，需要不断拉近正样本对的距离，同时推开负样本对。为了防止模型走捷径（即输出恒定特征），研究者设计了动量编码器、记忆库，甚至像SimSiam这样不需要负样本的孪生网络结构。这种机制让视觉模型学会了对图像变换保持不变的强健特征。

### 图学习：拓扑结构的特征补全

现实世界中，还有大量的数据是以图的形式存在的，例如社交网络、分子结构等。`**图学习**（**Graph Learning, GL**）`领域的参数量也在快速膨胀，对大规模预训练数据的需求日益急迫。

由于图数据包含了独特的节点和边信息，且没有固定的网格或序列结构，直接套用文本或图像的方法并不现实。

<img src="/images/2302.09419v3/page_27_Figure_0.jpg" alt="Graph Pretraining" style="width:90%; max-width:700px; margin:auto; display:block;">

该研究指出，基于`**图信息补全**（**Graph Information Completion, GIC**）`的预训练成为了主流。这同样是对“通识教育”理念的延续。

研究者会随机掩码图中的部分属性或拓扑结构，然后利用未被掩码的图数据去恢复它们。例如，AttributeMask任务会随机遮盖节点属性，要求模型进行重建；而EdgeMask任务则试图通过遮盖边，来强迫图神经网络学习局部的连通性信息。这种方式使得模型不仅能理解单个节点，更能深刻洞察节点之间的相互关系。

### 挑战与工程启示

尽管PFMs在多模态上都取得了压倒性的优势，但该研究也明确指出了该领域面临的巨大挑战：

首先是**模型效率与压缩**。预训练模型对算力的需求是极其恐怖的。如何在保证高层语义和语法信息不丢失的前提下，实现无损压缩？像ALBERT通过参数共享来大幅减少模型体积的思路，为未来的端侧部署提供了重要的工程启示。

其次是**隐私与安全**。大模型在预训练时吞噬了海量互联网数据，这其中不可避免地包含了敏感信息。如何确保大模型在生成内容时不泄露隐私数据，甚至如何抵御针对模型的恶意攻击，是学术界和工业界亟待解决的关键命题。

从BERT到ChatGPT，预训练基础模型已经证明了“一次预训练，处处可微调”的巨大潜力。向着多模态统一融合与通用人工智能的目标，PFMs的演进之路才刚刚开始。
