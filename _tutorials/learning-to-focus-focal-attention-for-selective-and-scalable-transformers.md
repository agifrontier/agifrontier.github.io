---
layout: default
title: "Learning to Focus: Focal Attention for Selective and Scalable Transformers"
---

## Learning to Focus: Focal Attention for Selective and Scalable Transformers

- **ArXiv URL**: http://arxiv.org/abs/2511.06818v1

# TL;DR
本文提出一种名为**Focal Attention**的注意力机制，它通过控制softmax函数的温度参数来锐化注意力分布，使模型能更集中于相关Token并抑制无关Token，从而在更少的参数或训练数据下，显著提升模型性能，尤其在长上下文任务中表现优越。

# 相关工作
当前，基于Transformer架构的大语言模型是AI领域的主流，其核心是注意力机制。标准的注意力机制使用softmax函数计算Token的权重，但在长上下文中，这种方式常常产生“嘈杂”的概率分布，即便是无关的Token也会被分配一定的注意力权重。这种噪声会损害模型每一层的特征选择效率，成为性能瓶颈，尤其在处理包含数千甚至数十万Token的长序列时问题更加突出。原始Transformer通过一个固定的缩放因子（维度的平方根倒数）来稳定训练，但这对于所有层和所有上下文可能并非最优。

本文旨在解决标准注意力机制中存在的**注意力分布噪声问题**，从而提升模型在复杂任务，特别是长上下文场景下的特征选择能力和整体性能。

# 本文方法
本文方法的核心是对Transformer中的标准注意力机制进行一个简单而有效的修改，提出了**Focal Attention**。其本质是在计算注意力权重时，显式地引入一个温度（temperature）参数来调整softmax函数的输出分布。

标准注意力机制的计算公式为：


{% raw %}$$ Attention(X) = \text{softmax}(\frac{QK^T}{\sqrt{d}})V $${% endraw %}


其中 $$softmax$$ 的输入 logits 被一个固定的因子 $$sqrt(d)$$ 缩放。

Focal Attention 修改了这个缩放因子，提出了两种实现方式：

### 恒定温度（Constant Temperature）
这是最简单的一种形式。引入一个全局的、作为超参数设定的温度缩放因子 $$t$$。公式变为：


{% raw %}$$ Attention(X) = \text{softmax}(\frac{QK^T}{t\sqrt{d}})V $${% endraw %}


通过设置一个较小的值（如 $$t < 1$$），可以增大logits之间的差异，使得softmax输出的概率分布更加“尖锐”（sharper）。这能让模型更聚焦于少数最相关的Token，并忽略大量不相关的背景信息，从而实现“降噪”和更精准的特征提取。

<img src="/images/2511.06818v1/intro_base.jpg" alt="标准注意力与Focal Attention对比" style="width:85%; max-width:600px; margin:auto; display:block;">
<img src="/images/2511.06818v1/intro_focal.jpg" alt="标准注意力与Focal Attention对比" style="width:85%; max-width:600px; margin:auto; display:block;">
*图1：左图为基线模型的注意力分布，存在噪声。右图为Focal Attention的分布，更集中于相关Token。*

### 可学习温度（Learned Temperature）
为了让模型能根据不同层、不同上下文动态调整注意力的锐度，本文还提出了一种可学习的温度方案。温度 $$τ$$ 不再是固定的超参数，而是从当前注意力模块的输入隐状态 $$X$$ 中学习得到：


{% raw %}$$ \tau = \text{clip}(\text{mean}(Xw_{\tau}), \tau_{min}, \tau_{max}) $${% endraw %}


其中，$w\_{\tau}$ 是一个可学习的参数向量。然后，注意力计算变为：


{% raw %}$$ Attention(X) = \text{softmax}(\frac{QK^T}{\tau})V $${% endraw %}


这种方式允许模型自主学习在不同层级采用不同的注意力策略，例如在底层使用较“柔和”的注意力分布以捕捉更广泛的信息，而在高层使用更“尖锐”的分布以做出更确信的决策。

### 创新点
- **核心创新**：将通常用于模型输出层以控制生成多样性的**温度缩放技术**，创新性地应用到了Transformer模型**内部的每一层注意力机制**中，以解决特征选择中的噪声问题。
- **优点**：该方法非常**简单**，只涉及对注意力计算公式的一个微小调整，计算开销极低。但它能有效**锐化注意力分布**，提升模型信噪比，从而在模型尺寸、训练数据和上下文长度等维度上展现出更优的**伸缩性（scaling properties）**。

# 实验结论
实验围绕LLaMA架构的模型，对Focal Attention的有效性进行了全面验证。

### 伸缩性（Scaling Properties）
- **模型尺寸**：Focal Attention展现出更优的模型尺寸伸缩性。在达到与基线模型相同的性能时，Focal Attention最多可节省**42%的参数**。
- **训练数据**：Focal Attention需要更少的训练数据。实验表明，使用Focal Attention的模型仅需**210B Token**的训练数据，即可达到基线模型使用**315B Token**训练后的性能水平，节省了约**33%**的训练数据。
- **上下文长度**：随着上下文长度的增加（从2048到8192），Focal Attention相较于基线模型的性能优势也随之增大，证明了其在处理长序列上的优越性。

<img src="/images/2511.06818v1/scale_params_loss.jpg" alt="模型尺寸扩展性" style="width:85%; max-width:600px; margin:auto; display:block;">
<img src="/images/2511.06818v1/scale_params_tasks.jpg" alt="模型尺寸扩展性" style="width:85%; max-width:600px; margin:auto; display:block;">
*图2：Focal Attention在不同模型尺寸下的验证损失和任务平均准确率均优于基线模型。*

<img src="/images/2511.06818v1/scale_tokens_loss.jpg" alt="训练数据扩展性" style="width:85%; max-width:600px; margin:auto; display:block;">
<img src="/images/2511.06818v1/scale_tokens_tasks.jpg" alt="训练数据扩展性" style="width:85%; max-width:600px; margin:auto; display:block;">
*图3：随着训练Token数量增加，Focal Attention的性能优势愈发明显。*

### 下游任务与长上下文能力
- **通用任务**：在一个2.7B参数的模型上，与基线模型相比，Focal Attention在常识推理基准测试中平均取得了**2.2个点的绝对提升**。

<br>


| 模型 (2.7B @ 315B tokens) | ARC-e | ARC-c | BoolQ | HellaS. | LAMBADA | PIQA | WinoG. | **平均** |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Transformer (基线) | 79.5 | 50.1 | 82.2 | 80.0 | 79.6 | 82.2 | 75.3 | 75.6 |
| Focal Attention (恒定温度) | **81.7** | **52.4** | **84.5** | **81.4** | 79.2 | **82.3** | **76.0** | **78.2** |
| Focal Attention (可学习温度) | 81.3 | 51.5 | 83.2 | 81.2 | **80.3** | 82.2 | 75.7 | 77.9 |

<br>

- **长上下文任务**：这是Focal Attention表现最出色的领域。在HELMET长上下文评测框架中，Focal Attention在**上下文学习（In-Context Learning, ICL）**、**检索增强生成（RAG）**、**长文档问答**等任务上，取得了**17%到82%的显著相对性能提升**。即使在超出训练长度的上下文（如从32K扩展到64K）进行评估时，其性能优势依然稳固。

<img src="/images/2511.06818v1/icl.jpg" alt="长上下文任务性能" style="width:85%; max-width:600px; margin:auto; display:block;">
<img src="/images/2511.06818v1/rag.jpg" alt="长上下文任务性能" style="width:85%; max-width:600px; margin:auto; display:block;">
*图4：在上下文学习（左）和RAG（右）等长上下文任务中，Focal Attention显著优于基线模型。*

- **表现平平的场景**："可学习温度"版本在部分任务上表现不如更简单的"恒定温度"版本。在长上下文评测中，"段落重排（Passage Reranking）"任务上的结果好坏参半。

### 消融研究
- 最佳温度值 $$t$$ 在0.4附近。所有测试的 $$t < 1$$ 的值都比 $$t = 1$$（基线）表现更好。
- 从头开始使用Focal Attention进行训练，比在预训练好的模型上继续训练以适应Focal Attention效果更好。

### 总结
Focal Attention是一种简单、高效且易于实现的注意力机制改进。它通过锐化注意力分布，显著提升了Transformer模型的性能和伸缩效率，在长上下文任务中尤为有效，为构建更强大、更高效的大语言模型提供了一条极具前景的路径。