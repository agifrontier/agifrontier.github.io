---
layout: default
title: "Learning to Focus: Focal Attention for Selective and Scalable Transformers"
---

## Learning to Focus: Focal Attention for Selective and Scalable Transformers

- **ArXiv URL**: http://arxiv.org/abs/2511.06818v1

# TL;DR
本文提出了一种名为“焦点注意力”（Focal Attention）的方法，通过在注意力机制中引入一个可控的温度（temperature）参数来锐化softmax分布，从而使模型能更专注于相关信息、抑制无关噪声，显著提升了模型在长上下文任务上的性能和整体扩展性。

# 背景
Transformer架构是现代大型语言模型的核心，其关键在于注意力机制。标准的注意力机制使用Softmax函数来计算token之间的相关性权重。

然而，标准的Softmax在处理长序列时，其输出的概率分布往往是“嘈杂的”，即会为许多不相关的token分配微小但非零的注意力权重。这种“注意力噪声”会逐层累积，损害模型有效筛选特征的能力，尤其是在上下文非常长、无关信息爆炸的情况下，这一问题尤为严重。

本文旨在解决注意力机制中的噪声问题，通过一种简单的方法增强模型从长序列中精确提取关键信息的能力。

# 方法
本文提出的核心方法是 **焦点注意力 (Focal Attention)**，它通过主动控制Softmax的温度来调整注意力分布的锐度。标准的注意力计算公式为：


{% raw %}$$ Attention(X)=softmax{(\frac{QK^{T}}{\sqrt{d}})}V $${% endraw %}


其中 $$softmax$$ 的输入 $$logits$$ 被一个固定的值 $$sqrt(d)$$（$$d$$是键向量的维度）缩放。

焦点注意力则在此基础上引入一个额外的温度参数 $$t$$。当 $$t < 1$$ 时，它会放大 $$logits$$ 之间的差异，使得Softmax输出的概率分布更加“尖锐”或“聚焦”。

### 创新点
本文的创新之处在于，将温度缩放这一技术系统性地应用于Transformer模型内部的**每一层注意力模块**中，并将其作为提升模型表示学习能力的核心手段，而非仅仅用于推理阶段的文本生成。这与仅在输出层使用温度采样的传统做法有本质区别。

其核心优点是**选择性（selectivity）**。通过锐化注意力分布，模型被迫将注意力集中在少数最相关的token上，有效过滤了上下文中的噪声，从而提高了信噪比。这种机制在长上下文场景下尤其强大，因为需要被忽略的无关信息数量巨大。

<img src="/images/2511.06818v1/intro_base.jpg" alt="标准注意力与焦点注意力的对比" style="width:85%; max-width:600px; margin:auto; display:block;">
<img src="/images/2511.06818v1/intro_focal.jpg" alt="标准注意力与焦点注意力的对比" style="width:85%; max-width:600px; margin:auto; display:block;">
*图1：左图为标准注意力的分布，较为分散；右图为焦点注意力的分布，显著减少了噪声，将注意力权重重新分配给了更相关的token。*

本文提出了两种实现焦点注意力的方式：

### 恒定温度的Transformer (Transformer with Constant Temperature)
这是最简单直接的实现。在整个模型的所有注意力层中，使用一个全局固定的温度超参数 $$t$$。注意力公式被修改为：


{% raw %}$$ Attention(X)=softmax{(\frac{QK^{T}}{t\sqrt{d}})}V $${% endraw %}


实验表明，即使是一个简单的、全局共享的 $$t < 1$$（如 $$t=0.4$$）也能带来显著的性能提升。这种方法几乎不增加任何计算开销。

### 可学习温度的Transformer (Transformer with Learned Temperature)
为了让模型更具适应性，本文提出了一种让模型自行学习每层注意力温度 $$τ$$ 的方法。具体来说，温度 $$τ$$ 是根据 attention 模块的输入隐状态 $$X$$ 动态计算得出的：


{% raw %}$$ \tau=clip(mean(Xw_{\tau}),\tau_{min},\tau_{max}) $${% endraw %}


其中 $$w_τ$$ 是一个与隐状态维度 $$d$$ 相同的可学习参数向量。然后，该 $$τ$$ مباشرة用于缩放 $$logits$$：


{% raw %}$$ Attention(X)=softmax{(\frac{QK^{T}}{\tau})}V $${% endraw %}


这种设计允许模型在不同层学习到不同的注意力锐度。例如，底层网络可能需要更“柔和”的注意力来 exploratory地融合信息，而高层网络则可能需要更“尖锐”的注意力来做出精确的决策。

# 效果
本文通过从头开始训练不同尺寸的模型（400M到9.5B），并在多个维度上进行了广泛的实验验证。

### 优秀的扩展性
焦点注意力在模型尺寸、训练数据量和上下文长度三个维度上都展现出比标准Transformer更优的扩展性。

*   **模型尺寸扩展**：焦点注意力能以更少的参数达到与基线模型相同的性能。实验表明，**使用焦点注意力的模型只需基线模型58%的参数（即减少42%）**，即可在下游任务上取得同等准确率。

    <img src="/images/2511.06818v1/scale_params_loss.jpg" alt="模型尺寸扩展性" style="width:85%; max-width:600px; margin:auto; display:block;">
    <img src="/images/2511.06818v1/scale_params_tasks.jpg" alt="模型尺寸扩展性" style="width:85%; max-width:600px; margin:auto; display:block;">
    *图2：随着模型参数量的增加，Focal Attention（红色和蓝色线）相比基线模型（灰色线）在验证损失和下游任务准确率上始终表现更优，且优势随模型增大而更明显。*

*   **训练数据扩展**：在相同的模型尺寸（2.7B）下，焦点注意力仅需**约210B tokens的训练数据**，即可达到基线模型训练了315B tokens后的性能，相当于**节省了33%的训练数据**。

    <img src="/images/2511.06818v1/scale_tokens_loss.jpg" alt="训练数据扩展性" style="width:85%; max-width:600px; margin:auto; display:block;">
    <img src="/images/2511.06818v1/scale_tokens_tasks.jpg" alt="训练数据扩展性" style="width:85%; max-width:600px; margin:auto; display:block;">
    *图3：随着训练token数量的增加，Focal Attention的性能优势持续扩大。*

*   **上下文长度扩展**：随着上下文长度从2048增加到8192，焦点注意力带来的性能提升也随之增大，表明它在处理更长序列时比标准Transformer更有效。

    <img src="/images/2511.06818v1/long_context_777m.jpg" alt="上下文长度扩展性" style="width:85%; max-width:600px; margin:auto; display:block;">
    *图4：在不同尺寸的模型上，随着上下文长度增加，Focal Attention（红色和蓝色）相较于基线（灰色）的验证损失优势越来越大。*

### 下游任务和长上下文能力
在2.7B规模、经过315B tokens训练的模型上，焦點注意力的表现如下：

*   **通用推理任务**：在包括ARC、HellaSwag、LAMBADA等在内的常识推理任务上，焦点注意力模型（恒定温度版）的平均分比基线模型**高出2.2个绝对百分点**。


| 模型 | ARC-e | ARC-c | BoolQ | HellaSwag | LAMBADA | PIQA | Winogrande | 平均分 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 基线 Transformer | 74.2 | 48.0 | 79.5 | 77.2 | 75.9 | 80.7 | 73.1 | 72.7 |
| 焦点注意力 (恒定温度) | **76.0** | **48.8** | **83.1** | **78.2** | 76.5 | 81.3 | **75.5** | **74.9** |
| 焦点注意力 (可学习温度) | 75.6 | 48.5 | 82.2 | 77.9 | **77.2** | 81.3 | 74.3 | 73.8 |

*   **长上下文能力**：这是焦点注意力表现最亮眼的领域。在包含多示例上下文学习（ICL）、长文档问答（LongQA）、检索增强生成（RAG）等任务的HELMET评测基准上，焦点注意力模型在长達32K的上下文长度上展现出巨大优势。与基线相比，它在不同任务上取得了**17%到82%的相对性能提升**。

    <img src="/images/2511.06818v1/icl.jpg" alt="长上下文任务综合表现" style="width:85%; max-width:600px; margin:auto; display:block;">
    <img src="/images/2511.06818v1/rag.jpg" alt="长上下文任务综合表现" style="width:85%; max-width:600px; margin:auto; display:block;">
    *图5：在多示例上下文学习（ICL）和检索增强生成（RAG）等长上下文任务中，Focal Attention（蓝色）在不同上下文长度上均显著优于基线模型（灰色）。*

### 消融研究
*   **最佳温度**：对于2.7B模型，恒定温度 $$t=0.4$$ 时效果最佳。这表明适度的锐化是有益的。
*   **从头训练 vs. 微调适应**：将一个预训练好的标准Transformer模型通过微调来适应焦点注意力，虽然能提升性能，但效果不如从头开始就使用焦点注意力进行训练。这说明焦点注意力影响了模型底层的学习范式，最好在训练初期就引入。