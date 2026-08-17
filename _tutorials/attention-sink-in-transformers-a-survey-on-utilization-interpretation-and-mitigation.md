---
layout: default
title: "大模型为何总盯“废话”？全景解析Transformer注意力沉没(AS)的4大利用范式"
description: "当我们在跟大语言模型对话，或者让视觉模型识别图像时，它们的注意力真的都放在你精心准备的提示词或核心物体上吗？事实并非如此。研究人员发现了一个极其反直觉的现象：无论模型多么先进，它们总是会将海量的“注意力”倾注在毫无语义信息的“废话”上。"
arxiv_id: "2604.10098"
topics:
  - "基础模型"
tags:
  - "Attention Mechanism"
  - "Attention Sink"
  - "Fundamental Utilization"
  - "Hallucinations"
  - "Inference Dynamics"
  - "Interpretability"
related_tutorials:
  - "a-survey-of-weight-space-learning-understanding-representation-and-generation"
  - "transformers-are-ssms-generalized-models-and-efficient-algorithms-through-struct"
  - "large-language-model-brained-gui-agents-a-survey"
  - "a-survey-on-data-selection-for-language-models"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">Attention Sink in Transformers: A Survey on Utilization, Interpretation, and Mitigation</p>

当我们在跟大语言模型对话，或者让视觉模型识别图像时，它们的注意力真的都放在你精心准备的提示词或核心物体上吗？事实并非如此。

> **ArXiv URL**：http://arxiv.org/abs/2604.10098v1

研究人员发现了一个极其反直觉的现象：无论模型多么先进，它们总是会将海量的“注意力”倾注在毫无语义信息的“废话”上。例如，文本模型会死死盯住句首的占位符，视觉模型会紧紧盯住图片边缘的纯色背景。这种在底层架构中普遍存在的注意力高度集中于低信息量Token的现象，被称为**注意力沉没**（**Attention Sink, AS**）。

本文将基于最新的系统性综述，深入剖析AS现象的底层逻辑。我们将不局限于探究它为何发生，更将重点揭示顶尖的工程师们如何“变废为宝”，巧妙利用这一现象来突破大模型的上下文长度瓶颈与推理速度。

### 核心机制剖析

要理解AS，必须回到Transformer的底层原语：**多头自注意力**（**Multi-Head Self-Attention, MHSA**）机制。

对于输入序列，Transformer通过计算查询矩阵 $\mathbf{Q}$ 与键矩阵 $\mathbf{K}$ 的点积来评估Token之间的相关性。最关键的一步在于，这些相关性得分必须通过 $softmax$ 函数进行归一化：


{% raw %}$$ \text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V})=\text{Softmax}\left(\frac{\mathbf{Q}\mathbf{K}^{T}}{\sqrt{d_{k}}}\right)\mathbf{V} $${% endraw %}



这里隐藏着一个硬性约束：无论当前Token是否真的需要关注其他上下文，$softmax$ 的输出总和必须严格等于1。

为了方便理解，我们可以将注意力机制看作一个封闭的“水压系统”，系统的总水压恒定为1。当模型处理到某些层时，如果它发现当前并不需要提取任何有用的上下文信息，它该把这部分多余的“水压”排向哪里？

模型非常聪明地演化出了一种策略：它会自发寻找一个固定的、没有实际语义危害的Token（例如初始位置的特殊字符），将其作为“泄压阀”。大量无处安放的注意力被倾泻到这个Token上，从而保持了整体数值表示的稳定性。这就是AS产生的根源。

<img src="/images/2604.10098v1/x3.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">
Figure 4: 标准Transformer架构及典型AS现象示意图，沉没Token表现出异常高的注意力得分。

在数学定义上，如果一个Token的累计注意力得分远超平均值，且其自身承载的信息量极低，它就被判定为沉没Token。

### 跨域模型表征

AS并非某种特定模型的缺陷，而是Transformer家族的“遗传特征”。但在不同的模型架构中，这个“泄压阀”的具象表现存在显著差异。

**大语言模型**（**Large Language Models, LLMs**）
在自回归生成中，由于因果掩码的限制，模型只能关注历史信息。研究发现，LLMs在深层网络中，会将跨越所有注意力头的巨大比重分配给最初的几个Token（如 `[BOS]`）。这些初始Token就像是序列中的锚点，吸收了系统运行中产生的多余注意力。

**视觉大模型**（**Vision Transformers, ViTs**）
在纯视觉任务中，AS现象同样存在，但其空间分布呈现出另一种规律。ViTs中的沉没Token通常出现在图像边缘的背景Patch上。

<img src="/images/2604.10098v1/image.jpg" alt="Refer to caption" style="width:80%; max-width:300px; margin:auto; display:block;">
Figure 11: ViT中的异常值与AS分析总结。可以观察到注意力高度集中在无语义的背景图像块上。

这些背景Patch不仅吸引了不成比例的注意力概率，而且其激活值（Value magnitudes）异常低。它们的作用不再是传递视觉特征，而是作为一种隐式的偏置项，帮助模型稳定整体的注意力分布。

**多模态大模型**（**Multi-Modal Large Language Models, MLLMs**）
当视觉与文本融合时，AS表现为多模态聚集。注意力不仅会流向文本端的初始Token，还会在视觉端形成特定的“视觉注意力沉没”。这些视觉沉没点往往是无关紧要的背景区域，它们会无差别地吸收注意力，甚至可能导致模型产生幻觉，因为它们分散了模型对核心语义目标的关注。

### 核心利用范式

既然AS是由于架构限制而必然产生的“泄压阀”，强行拆除它往往会导致模型性能崩溃。因此，当前的工程前沿已经从“消除它”转向了“利用它”。综述将其归纳为几种极具启发性的核心范式。

#### 沉没Token保留

这是目前在长文本推理和高效部署中最成功的策略，其核心思想极其淳朴：既然模型需要这个“泄压阀”来维持稳定，那我们在做任何上下文压缩时，**永远保留它**。

在**键值缓存**（**KV Cache**）压缩任务中，如果我们采用滑动窗口机制丢弃旧Token，模型往往会瞬间崩溃。但以 `StreamingLLM` 为代表的技术发现，只要在缓存中永久保留前几个沉没Token的KV特征：


{% raw %}$$ \hat{\mathcal{C}}_{t}=\{(k_{i},v_{i}):i\in\mathcal{I}^{\text{sink}}\cup\mathcal{I}^{\text{window}}\} $${% endraw %}


模型就能在极小的内存占用下，无限期地进行流畅生成。

<img src="/images/2604.10098v1/x9.jpg" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">
Figure 13: StreamingLLM通过保留AS以及最近的Token，实现了在超长文本上的高效稳定计算。

不仅在内存压缩中如此，在**量化保护**（**Quantization-Aware Protection**）中，由于沉没Token往往伴随着极端的激活异常值，对它们进行低比特量化会导致灾难性的精度损失。现代量化方案会将这些少数的AS Token保留在较高的精度（如16位），而对其余Token进行激进的2比特量化，从而在不掉点的情况下大幅压缩模型体积。

#### 注意力重分配

保留策略是一种被动妥协，而**注意力重分配**（**Attention Redistribution**）则试图主动改造系统。

它的逻辑是：既然沉没Token占用了过多的注意力额度，我们可以通过干预手段，把这些额度“抢”回来，重新分配给真正有语义价值的Token。

在显式重分配中，算法会定位到沉没Token的索引集合 $\mathcal{S}$，并在计算出注意力矩阵后强制衰减它们的分数，将释放出的概率质量平均补充到目标Token集合 $\mathcal{T}_{i}$ 中：


{% raw %}$$ \tilde{A}_{ij}= A_{ij}+\beta\cdot\frac{1}{|\mathcal{T}_{i}|}\sum_{s\in\mathcal{S}}A_{is},\quad j\in\mathcal{T}_{i} $${% endraw %}



这种主动疏导“水压”的方法在多模态模型中表现极佳。通过将冗余的视觉注意力重定向到核心对象上，不仅提升了视觉定位能力，还显著降低了多模态模型“胡说八道”（幻觉）的概率。

#### 可学习前缀引入

相较于让模型自己在输入数据中随机寻找“泄压阀”，不如我们主动为它修建一个标准的“蓄水池”。

**可学习前缀Token**（**Learnable Prefix Tokens**）技术在模型输入端硬编码了少量带有可训练参数的特殊Token $\mathbf{P}$，将其与原始数据 $\mathbf{X}$ 拼接：


{% raw %}$$ \mathbf{S}=[\mathbf{P};\mathbf{X}]\in\mathbb{R}^{(K+N)\times D} $${% endraw %}



在训练过程中，模型会自发地学会将冗余的全局注意力路由到这些参数化的前缀上。著名的 `Vision Transformers Need Registers` 研究就是采用了这种方法。为ViT添加“寄存器”Token后，那些原本分布在图像背景上的丑陋的注意力异常斑点消失了，模型的特征图变得异常干净，不仅密集预测任务的性能大幅提升，模型的可解释性也得到了极大改善。

### 工程实践启示

注意力沉没现象的发现与利用，给大模型工程界带来了极其深刻的启示。

目前的底层优化通常假设AS的位置是静态且固定的（例如永远在句首）。但在复杂的视觉模型或混合专家系统中，AS往往在非初始位置动态涌现。这就对底层的算子开发提出了新要求：未来的高性能推理框架（如 `vLLM` 或 `TensorRT-LLM`），亟需集成轻量级的动态AS检测机制。

同时，我们也要认识到，AS的本质是 $softmax$ 操作带来的副产物。在下一代Transformer架构甚至是非Transformer架构（如线性注意力和状态空间模型）中，如何从根本上优化注意力分配逻辑，避免算力的无谓浪费，将是通往更高效通用人工智能的关键一步。但在当前阶段，精细化地管理和利用好这些“废话”Token，依然是压榨大模型性能的最有效手段。
