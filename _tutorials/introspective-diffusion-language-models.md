---
layout: default
title: "吞吐量飙升3倍！I-DLM破局并行解码，媲美同级自回归质量"
description: "大语言模型的世界一直被逐字生成的自回归机制统治。虽然扩散模型承诺了令人兴奋的并行生成前景，但它们在文本质量上始终无法与自回归模型抗衡。为什么扩散模型在图像领域大杀四方，在文本生成领域却总是“差一口气”？"
arxiv_id: "2604.11035"
topics:
  - "基础模型"
  - "模型训练"
tags:
  - "Autoregressive Models"
  - "Diffusion Language Models"
  - "Introspective Acceptance Rate"
  - "Introspective Consistency"
  - "Introspective Diffusion Language Model"
  - "Introspective Strided Decoding"
related_tutorials:
  - "why-low-precision-transformer-training-fails-an-analysis-on-flash-attention"
  - "seesaw-accelerating-training-by-balancing-learning-rate-and-batch-size-schedulin"
  - "small-llms-pruning-vs-training-from-scratch"
  - "gemini-1-5-unlocking-multimodal-understanding-across-millions-of-tokens-of-context"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">Introspective Diffusion Language Models</p>

大语言模型的世界一直被逐字生成的自回归机制统治。虽然扩散模型承诺了令人兴奋的并行生成前景，但它们在文本质量上始终无法与自回归模型抗衡。

> **ArXiv URL**：http://arxiv.org/abs/2604.11035v1

为什么扩散模型在图像领域大杀四方，在文本生成领域却总是“差一口气”？本文的研究团队直击痛点，指出传统扩散语言模型缺乏一种关键能力：`**内省一致性**（**Introspective Consistency**）`。

基于此，该研究推出了 `**内省扩散语言模型**（**Introspective Diffusion Language Model, I-DLM**）`。在15个基准测试中，它不仅追平了同规模自回归模型的质量，更带来了高达3倍的吞吐量提升。

### 核心症结：何为内省一致性？

自回归模型有一个隐秘的结构优势：它们在训练时不仅学习预测下一个词，还隐式地学会了复核之前生成的词。换句话说，它们对自己生成的内容保持着高度认同。

然而，传统的 `**扩散语言模型**（**Diffusion Language Models, DLMs**）` 通常采用多步双向去噪，这导致模型生成的词分布与其自身的下一步预测产生了分歧。

为了量化这一现象，该研究提出了 `**内省接受率**（**Introspective Acceptance Rate**）` 的概念。它用于衡量模型在内部是否接受其之前生成的标记。

这就好比一个团队在流水线上作业，自回归模式下的员工每接手一个零件，都会顺势检查前面的工序。而传统扩散模式下的员工虽然同时开工，但互相不看对方的半成品，最后拼装出来的产品往往难以严丝合缝。这种缺乏内省一致性的表现，严重制约了传统 DLM 在复杂推理任务上的表现。

### 破局之道：重塑训练范式

该研究没有像前人那样试图让扩散模型变得“更像”自回归，而是从自回归的本质出发，提取其核心原则并融入并行生成范式中。

I-DLM 通过一种极其高效的一致性训练配方，将预训练的自回归模型转化为具备内省能力的扩散模型。

这套配方包含三个关键要素：严格的因果注意力机制、`**逻辑偏移**（**Logit Shifting**）` 以及全掩码目标。具体而言，该研究设计了如下的核心损失函数来兼顾生成与验证：




{% raw %}$$ \mathcal{L}\_{\text{mask}}=-\frac{1}{|\mathcal{S}\_{t}|}\sum\_{\ell\in\mathcal{S}\_{t}}\log p\_{\theta}(x\_{0}^{\ell+1}\mid[x\_{t},x\_{0}]\_{\leq\ell}) $${% endraw %}






{% raw %}$$ \mathcal{L}=\mathcal{L}\_{\text{mask}}+\hat{s}\cdot\mathcal{L}\_{\text{clean}},\quad\hat{s}=\frac{\mathcal{L}\_{\text{mask}}}{\mathcal{L}\_{\text{clean}}} $${% endraw %}



因果掩码确保了生成时上下文在各个去噪步骤中的一致性。逻辑偏移则通过统一的隐藏状态，巧妙地桥接了验证与生成过程，并尊重了基础自回归模型固有的行为逻辑，保证了训练的鲁棒性。

全掩码目标通过密集的监督信号保证了训练效率。这种一步到位的训练无需复杂的蒸馏调度或掩码课程，为构建高质量 DLM 提供了一条稳定且高效的路径。

### 解码革新：内省跨步解码算法

在推理端，该研究引入了创新的 `**内省跨步解码**（**Introspective Strided Decoding, ISD**）` 算法。

传统 DLM 需要多次迭代去噪，计算开销巨大。而 ISD 能够在同一次前向传播中，既生成一批新词，又根据因果锚点分布验证之前生成的词。

<img src="/images/2604.11035v1/x6.jpg" alt="Figure 4" style="width:85%; max-width:600px; margin:auto; display:block;">

在 $[MASK]$ 位置，模型大胆提出新词；在内省位置，模型则严格复核已有的词。这种机制从数学上保证了输出分布与基础自回归模型一致，无需依赖外部的置信度试探或单独的验证轮次。

团队依然沿用前面的流水线比喻：ISD 算法相当于给并行作业的员工配备了一个同步校验器。每当产出一批零件，校验器瞬间完成质检和修正。它既保持了多线程并行推进的速度，又找回了顺次检查的严密性。

### 系统工程：无缝接入现代服务栈

纯算法的并行往往难以在实际部署中转化为真正的速度提升。现有的扩散模型推理堆栈大多是定制的，与现代大模型服务框架不兼容。

I-DLM 的另一大杀手锏在于其极强的工程落地能力。由于在架构上保留了严格的因果注意力结构，I-DLM 可以直接作为插件集成到诸如 SGLang 这样的成熟自回归服务系统中。

**继承自回归系统优化**
每个 ISD 步骤都映射到连续批处理框架的原生扩展模式中。由于 ISD 保证每步至少产出一个高质量标记，所有请求都能均匀推进，连续批处理可以无缝工作。这使得 I-DLM 在高并发下依然能保持吞吐量的线性缩放。

**静态批处理调度设计**
ISD 算法具有严格的依赖链：前向传播到验证，再到修剪和准备。为了避免 CPU-GPU 的同步开销，该研究设计了静态批处理解码循环，在连续的 ISD 步骤中重用批处理对象，极大地降低了系统延迟。

**算子融合与验证优化**
验证步骤被深度优化并融合为一个单一的 Triton 内核。它集成了在线 $softmax$ 和 Gumbel-max 修正。在约 78% 的常见接受路径下，内核只需进行一次流式传递即可返回，彻底跳过了耗时的修正步骤。

### 实证实战：质量与速度的双重飞跃

实验数据充分证明了这套设计的威力。该研究基于 Qwen3 训练了 8B 和 32B 版本的 I-DLM 模型，展现了惊人的性能基线。

<img src="/images/2604.11035v1/x1.jpg" alt="Figure 1" style="width:85%; max-width:600px; margin:auto; display:block;">

在衡量复杂数学推理能力的 AIME-24 基准上，I-DLM 达到了 69.6 的高分。在编程基准 LiveCodeBench-v6 上则达到了 45.7。这两个成绩分别超越了 16B 参数规模的 LLaDA-2.1-mini 模型达 26 分和 15 分之多。

不仅如此，由于消除了推理引擎的不兼容性，I-DLM 在服务效率上面对高并发请求展现出卓越的扩展性。它能够提供比以往 SOTA 扩散语言模型高出约 3 倍的实际吞吐量。

当与顶尖的 `**投机解码**（**Speculative Decoding**）` 方法 EAGLE3 竞争时，I-DLM 依然不落下风。尽管 EAGLE3 依赖额外的草稿模型，I-DLM 在单请求吞吐量上全面胜出，证明了其原生并行能力的巨大潜力。

### 结语启示

该研究通过敏锐地捕捉并修复“内省一致性”这一关键缺失，成功地将自回归模型的极高生成质量与扩散模型的并行速度完美融合。

I-DLM 打破了“扩散模型必须向自回归模型妥协”的思维定势，提供了一种统一生成与自我验证的优雅方案。当然，这种架构在极高并发场景下的显存管理仍有进一步压榨的空间。但不可否认，I-DLM 为大模型在算力受限时代的规模化部署，指出了一条兼顾质量与效率的崭新赛道。
