---
layout: default
title: "TransMem：不必重算长上下文，稀疏隐藏状态化身外挂记忆"
description: "为了让 TransMem 学习到通用的“信息提取与利用能力”，而非特定的事实知识，作者提出了一种优雅的训练策略： 证据条件自蒸馏（Evidence-Conditioned Self-Distillation, ECSD） 。上图 (b) 揭示了该训练范式的精髓。"
arxiv_id: "2607.29032"
paper_published: "2026-07-31"
published_at: "2026-09-11T13:15:08.164389+08:00"
topics:
  - "知识系统"
  - "模型优化"
tags:
  - "知识系统"
  - "模型优化"
  - "AI论文解读"
related_tutorials:
  - "sana-video-20-hybrid-linear-attention-with-attention-residuals-for-efficient-vid"
  - "robostral-navigate"
  - "frontis-ma1-training-an-ai4ai-model-towards-recursive-self-improvement-in-machin"
  - "echoverse-deep-evolving-environments-for-training-computer-use-agents-at-scale"
---

<p class="paper-original-title" lang="en">TransMem: Transforming Hidden States into Memory for Large Language Models</p>

面对几十万 Token 的超长上下文或多轮对话历史，大语言模型（LLM）往往暴露出两个令人头疼的硬伤：一边是随长度暴涨的显存与计算开销，另一边是众所周知的“迷失在中间”现象。很多时候，回答当前问题所需的关键证据其实早就被模型在前文计算过，但在因果自回归机制和注意力“近因偏置”（Recency Bias）的干扰下，那些深埋在前序文本里的特征很容易被后续的大量无关噪声稀释。

> ArXiv URL：https://arxiv.org/abs/2607.29032

目前业界的常规解法大体分为两类：要么靠外部显式检索（如各种 RAG 或 Agent 记忆框架），要么盲目扩充上下文窗口。但外部检索经常受限于分块截断与检索召回率，甚至面临反复迭代检索带来的高延迟；而简单粗暴地扩大上下文，计算成本呈二次方上升，依然解决不了注意力分布不均的问题。

针对这一痛点，一项名为 TransMem（Transforming Hidden States into Memory）的研究提供了一条全新路径：**与其在外部搭建复杂的文本数据库或反复全量重算长文本，不如直接将预填充阶段早已算好的极少量“稀疏隐藏状态”（Hidden States）变成动态可调用的参数化记忆**。该方法完全冻结 LLM 主干模型，仅在深层网络外挂轻量 Transformer 块与门控单元，配合独创的“特权证据自蒸馏”训练，不仅将 MemoryAgentBench 的准确率从 29.54% 提升至 40.00%，还在超长评测基准 LoCoMo 上取得了最高 29.25 的 $F_1$ 分数飞跃，且推理阶段的显存与计算增量完全不随上下文长度膨胀。

<img src="/images/2607.29032/introline.webp" alt="自回归 Transformer 中上下文表示的两个固有特性" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么隐藏状态是绝佳的“免费记忆”？

长上下文场景下，模型反复对全部历史 Token 做自注意力计算极其低效。但如果仔细探究自回归 Transformer 内部的表示演进，会发现两个天然存在的表征特性：

其一，隐藏状态天然聚合了前面的所有信息。因果注意力的三角下三角计算方式决定了，某一段文本尾部的隐藏状态，在深层已经对该片段及其前缀形成了高度浓缩的隐式摘要。如上图 (a) 所示，后序的隐状态本身就是前序序列的高阶压缩表征。

其二，深层隐状态具有显著的区域位置敏感性。如上图 (b) 所示，自注意力机制存在极强的近因偏置（除最开头的注意力下沉 Token 之外），越靠近当前生成位置的 Token，往往被分配越高的注意力权重。这意味着，即便回答问题的关键证据出现在文档前半段，处于末尾生成点的特征也会被更靠后的无关上下文“冲淡”。

这两个特质说明了一个关键事实：分散在不同历史位置的关键隐藏状态，恰好构成了整篇长文本互补的、多视角的语义片段。模型此前之所以在长文本中迷失，不是因为它没有算过这些信息，而是因为缺乏一种机制，在生成当前 Token 时把那些散落在各处的历史隐状态重新激活并注入到当前的计算流中。

### TransMem 架构：轻量介入与动态门控

基于这一发现，TransMem 没有去改造笨重的主干模型，而是设计了一个轻量级的即插即用参数化推理模块。整个系统的核心思想可以概括为：**主干保持不动，提取稀疏锚点，隐空间特征微调，动态门控注入**。

<img src="/images/2607.29032/main.webp" alt="TransMem 框架总览：推理机制与证据条件自蒸馏" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

如上图 (a) 所示，在推理阶段，主干模型照常执行前向传播。TransMem 并不抓取成千上万个历史 Token 的全部隐状态，而是策略性地挑选一小组极稀疏的历史隐状态集合 $H_M = \operatorname{Concat}(H_M^1, \dots, H_M^S)$。这些历史隐状态随后被送入一个轻量的 TransMem 模块中，该模块主体仅由极少数 Transformer 块 $f_m^K$ 和一个投影矩阵 $W_\Delta^K$ 构成，用于提炼跨片段的记忆残差向量 $\Delta h$：




{% raw %}$$ h_m = f^K_m(H_M, h_{1:t}^A), \quad \Delta h = W^K_\Delta h_m $${% endraw %}



为了防止外挂记忆过度干扰主干原有的生成能力，TransMem 引入了一个动态门控网络 $f_g^K$。门控网络根据提炼出的记忆特征，为当前 Token 计算一个非负缩放因子 $g_m$：




{% raw %}$$ g_m = f^K_g(h_m) = \alpha \cdot \sigma\left(\frac{W_g h_m}{\tau}\right) $${% endraw %}



最终，带有记忆干预的隐状态 $\hat{h}^K_t$ 仅通过简单的残差相加完成融合：$\hat{h}^K_t = h^K_t + g_m \odot \Delta h$，随后继续送入主干模型的下一层 Transformer 块中进行后续计算。由于该计算完全发生在隐状态层面，无需把历史文本拉回显存重新拼进上下文窗口，因此引入的延迟和计算量极低。

### 核心机制：让学生向“只看证据”的特权教师看齐

参数化记忆网络最容易犯的错误是“死记硬背”——把训练数据里的具体事实硬编码进参数里，导致换个任务或换篇文档就彻底失效。为了让 TransMem 学习到通用的“信息提取与利用能力”，而非特定的事实知识，作者提出了一种优雅的训练策略：**证据条件自蒸馏（Evidence-Conditioned Self-Distillation, ECSD）**。

上图 (b) 揭示了该训练范式的精髓。研究团队构建了一对共享同一冻结主干权重的“教师”与“学生”：

- **教师模型（Teacher）**处于信息特权地位：它只接收纯净的关键证据 $E$ 和问题 $Q$，没有任何冗长无关的长篇上下文干扰。由于不存在噪声，教师模型的预测概率分布 $p_T(\cdot \mid A_{<i}, E, Q)$ 几乎代表了该主干在理想状态下的最佳证据推理上限。

- **学生模型（Student）**则是配备了可学习 TransMem 模块的同一主干模型：它接收包含大量干扰信息的完整长上下文 $C$。它的任务是通过自蒸馏损失函数，尽量让自己的预测分布逼近教师：




{% raw %}$$ \mathcal{L}_{\mathrm{SD}} = \frac{1}{\lvert A \rvert}\sum_{i=1}^{\lvert A \rvert}\mathrm{KL}\!\left(p_T(\cdot\mid A_{<i}, E, Q)\,\middle\|\,p_S(\cdot\mid A_{<i}, C)\right) $${% endraw %}



这种自蒸馏设定的精妙之处在于，教师迫使学生学会一件事：**如何在满屏无关干扰的长上下文中，通过 TransMem 提取的历史隐状态，把丢失的黄金证据“补救”回来，并主动压制无关噪声**。

此外，研究团队还采用了全回路联合训练（In-loop training），将冻结的主干网络置于反向传播图内，联合优化注入在多个网络层的记忆模块。这样能确保高层记忆模块在更新时，直接感知到低层模块已经生效的隐状态修正，彻底避免了层间目标脱节的匹配误差。

### 实验评测：跨架构、跨长度的全面增益

为了验证 TransMem 是否真正学到了可迁移的长文本推理能力，研究人员在仅使用 HotpotQA 训练集训练 TransMem 的前提下，直接将其迁移到两组极具挑战性的外部长文本基准上进行 Zero-shot 泛化评估：

1. **LoCoMo**：平均长度约 16K Token 的多轮长交互基准，涵盖单跳（SH）、多跳（MH）、开放域（OD）和时序（TP）推理挑战。

2. **MemoryAgentBench**：很多场景上下文长度直接超过 256K Token，极度考验隐空间记忆的极限利用能力。

同时，实验测试了不同量级与架构的主干模型，包括 Qwen3-4B-Instruct、Llama3.1-8B-Instruct 以及 Qwen2.5-14B-Instruct。

实验表明，TransMem 带来了系统性的能力飞跃。在 LoCoMo 基准上，各主干模型搭载 TransMem 后均取得了至少 27% 的相对性能提升，$F_1$ 分数净增幅在 11.58 到 29.25 之间；在 HotpotQA 测试集上，Exact Match（EM）相对提升了约 14%，$F_1$ 增益达到 10.20 到 13.03。在超长上下文的 MemoryAgentBench 上，模型的平均准确率更是从基线的 29.54% 稳步提升至 40.00%。

一个颇具启发性的发现是模型跨架构的表现：尽管 Llama3.1-8B-Instruct 的基线能力不如 Qwen 系列亮眼，但在使用基于 Qwen3 架构设计的单层 TransMem 模块后，其 LoCoMo $F_1$ 分数达到了 51.64，HotpotQA 达到了 71.63，直接逼近甚至看齐了参数量更大的 Qwen2.5-14B-Instruct。这证明了轻量参数化记忆模块能够作为一种模块化插件，跨越参数规模弥补主干记忆能力的短板。

<img src="/images/2607.29032/ablation.webp" alt="TransMem 在计算效率与不同配置下的消融表现" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 计算开销：摆脱上下文长度的“常数级”代价

对于长文本外挂机制而言，如果计算开销随着上下文长度线性甚至二次方增长，工程落地将寸步难行。上图 (a) 给出了 TransMem 与同类参数化记忆方法（如 $\delta$-Mem、MLPMemory）在推理开销上的直观对比。

以 Qwen3-4B-Instruct 为例，当上下文长度从 10K Token 逐步提升到 100K Token 时：

- $\delta$-Mem 的额外计算开销呈现明显的线性上升趋势，每增加 10K Token 的隐记忆，就要引入近 100 GFLOPs 的额外计算；

- 而 TransMem 的额外计算复杂度**完全与输入上下文长度解耦**，其计算量仅取决于事先选定的记忆槽位数量 $\lvert H_M \rvert$。无论长文本是 10K 还是 100K，TransMem 引入的推理延迟与浮点运算量都保持在极低且恒定的水平线。

与此同时，消融实验揭示了网络注入深度与训练目标选择的关键取舍：

在**注入深度**方面，研究对比了将 TransMem 插入模型不同层（从后 2 层到后 8 层，以及中间层）的效果。实验表明，并非插入越多层越好，插入过早的中间层也收效甚微。在总共 36 层的 Qwen3-4B 上，仅在**最后 4 层（Last-4）**注入 TransMem 取得了最佳综合得分，尤其是在多跳推理（Multi-Hop）任务上表现最为突出。这印证了深层表征才承载了最充分的语义抽象，记忆融合发生在网络决策的后半程最为高效。

在**训练范式**方面，研究人员对比了常规有监督微调（SFT）、强化学习策略优化（GRPO）以及在策蒸馏（OPD）。结果表明，直接基于真实答案进行 SFT 表现明显滞后，即便在 SFT 基础上叠加强大的强化学习（GRPO 达到 53.18 $F_1$），也依然不及特权教师自蒸馏（ECSD 达到 54.11 $F_1$）。这说明对于记忆机制而言，Token 级别的特权证据指导远比粗粒度的答案奖励信号更有助于隐状态的去噪与校准。

<img src="/images/2607.29032/gate_layerwise_behavior.webp" alt="Qwen3-4B 最后四层门控因子的层级分布行为" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 窥探黑盒：门控网络到底学到了什么？

为了弄清 TransMem 内部的动态干预机制，作者可视化了 Qwen3-4B 模型中最后四层门控网络在测试集上的激活分布，如上图所示。从第 32 层到第 35 层，门控因子的取值呈现出极其清晰的**单调递增放大**趋势：

- 在第 32 层和 33 层，门控因子整体集中在接近 0 的区域，呈现出保守微调的状态，仅对极少数高确信度特征进行小幅修正；

- 随着层数进一步深入至 34 层和 35 层，门控因子的分布中心显著右移，均值大幅提高，展现出强烈的特征放大作用。

这种高度一致的“分工协作”表明，TransMem 并没有在所有插入层简单粗暴地注入相同强度的干预，而是自然演化出了一种层级调谐策略：前序层先进行保守校准，最靠近输出的深层再进行大尺度的记忆特征注入与整合。这种分层递进的机制，正是它在保持主干语言连贯性的同时，精准恢复关键长程证据的秘诀所在。

### 总结与启示

TransMem 为长上下文大模型推理与智能体记忆架构提供了一个颇具启发性的新视角。它证明了：

1. **不需要把长文本的一切都反复喂给自注意力机制**。因果自回归模型在预填充阶段计算出的深层隐状态，原本就是极佳的高阶压缩表征；

2. **记忆能力可以与主干推理能力解耦**。通过特权自蒸馏，我们可以在完全冻结主干的前提下，仅用轻量模块教会模型“如何去噪和提炼记忆”，且这种元能力具有极强的跨任务泛化性；

3. **参数化记忆完全可以做到常数级开销**。通过稀疏隐状态锚点与跨层门控注入，长文本的推理成本不必再随交互轮数失控攀升。

随着长长程 Agent（Long-horizon Agents）在软件工程、复杂科研与系统自动化中的广泛落地，这种既不增加主干负担、又能敏锐捕捉长程关键线索的参数化记忆范式，很可能会成为下一代轻量化 Agent 架构演进的重要参考支点。
