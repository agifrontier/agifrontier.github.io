---
layout: default
title: "HarnessBridge：重塑Agent交互，Token骤降90%还能涨点"
description: "随着上下文窗口越来越长，大模型在长周期任务中依然常常迷失。它们要么被历史交互中堆积如山的无效信息淹没，要么陷入死循环，疯狂消耗宝贵的API额度。过去，为了解决这个问题，开发者们通常会手写一套复杂的脚手架（Harness）代码。这些脚手架负责截断上下文、重试错误、解析输出。"
arxiv_id: "2606.12882"
topics:
  - "AI Agent"
  - "基础模型与理论"
tags:
  - "HarnessBridge"
  - "SWE-bench Verified"
  - "Terminal-Bench 2.0"
  - "action projection"
  - "agent-environment interface"
  - "bidirectional projection"
related_tutorials:
  - "learning-on-the-job-an-experience-driven-self-evolving-agent-for-long-horizon-ta"
  - "larger-datasets-can-be-repeated-more-a-theoretical-analysis-of-multi-epoch-scali"
  - "tthe-test-time-harness-evolution"
  - "multi-agent-ai-systems-outperform-human-teams-in-creativity"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">HarnessBridge: Learnable Bidirectional Controller for LLM Agent Harness</p>

随着上下文窗口越来越长，大模型在长周期任务中依然常常迷失。它们要么被历史交互中堆积如山的无效信息淹没，要么陷入死循环，疯狂消耗宝贵的API额度。

> **ArXiv URL**：http://arxiv.org/abs/2606.12882v1

<img src="/images/2606.12882v1/p01_harnessbridge_page_1.jpg" alt="论文中的核心图示" style="width:85%; max-width:600px; margin:auto; display:block;">

*论文原图：用于辅助理解核心方法或实验结果。*

过去，为了解决这个问题，开发者们通常会手写一套复杂的**脚手架**（**Harness**）代码。这些脚手架负责截断上下文、重试错误、解析输出。然而，面对日益复杂的交互，这种人工编写的硬编码规则越来越力不从心。

加州大学洛杉矶分校（UCLA）的最新研究提出了一种全新的思路：HarnessBridge。本文不再修补那些死板的代码规则，而是训练了一个轻量级的“可学习双向控制器”，直接接管Agent与环境的交互。它不仅让各大模型在评测中的成功率不降反升，更是让部分模型的Token消耗暴降超过90%。

### 传统硬编码脚手架的困境

在长周期任务（如代码修复、终端导航）中，Agent需要与环境进行几十甚至上百轮的交互。

**生成器**（**Generator**）能力的提升固然重要，但环境与Agent之间的“接口”同样决定了任务的成败。现有的Harness通常是静态的、手动设计的。

这种设计在两个方向上都暴露出致命缺陷。在“环境到Agent”方向，原始轨迹中充满了已过时的错误信息和冗余的输出，这极大地增加了Token成本，并分散了模型的注意力。在“Agent到环境”方向，模型可能会重复无效动作、进入死循环或输出不合规的指令，白白浪费环境执行步数。

### HarnessBridge：可学习的双向代理

为了打破这一瓶颈，该研究提出了HarnessBridge。这是一种端到端可学习的交互策略，它将Agent与环境之间的接口参数化为一个双向投影（Bidirectional Projection）。

如果把Agent比作一位远程办公的指挥官，环境是前线战场。那么HarnessBridge就是介于两者之间的“智能副官”。这位副官负责双向把关：对内，不让冗长无效的情报干扰指挥官；对外，果断拦下指挥官的离谱命令并给出建议。

具体而言，给定系统提示 $s$、任务指令 $q$ 和当前历史 $H_t$，HarnessBridge 学习到一个策略 $\pi_h$：




{% raw %}$$ \pi_h : (s, q, H_t, a_t) \mapsto (\widetilde{H}_t, a'_t) $${% endraw %}



其中 $\widetilde{H}_t$ 是暴露给Agent的压缩状态，$a'_t$ 是最终提交给环境的动作。在这个过程中，Agent的原始生成策略 $\pi_g$ 保持完全冻结，系统只优化这个接口策略 $\pi_h$。

### 对内把关：观察投影机制

对内过滤被称为**观察投影**（**Observation Projection**）。它的核心目标是决定原始交互历史应该如何向Agent展示。

设第 $t$ 轮的历史为一系列交互单元 $H_t = (h_1, . . . , h_t)$。观察投影会输出：




{% raw %}$$ \widetilde{H}_t = P_{obs}(s, q, H_t) = (U_t, \widetilde{h}_1, . . . , \widetilde{h}_t) $${% endraw %}



这里的核心机制在于曝光决策 $z_i$。对于每一个历史单元，模型会预测其属于 $PASS$、$Compress$ 还是 $Drop$。关键信息被原样保留，冗长但有用的信息被压缩，而完全失效的试错过程则被直接剔除。

更巧妙的是，模型会生成一个**活动状态索引**（**Active-State Index**）$U_t$。它就像一个动态的置顶备忘录，将当前未解决的报错、依然有效的约束和待办目标直接提取并放置在历史记录的最前方。

这一机制确保了Agent无需在几万Token的“垃圾堆”里翻找当前到底卡在了哪里。

### 对外拦截：动作投影机制

对外拦截被称为**动作投影**（**Action Projection**）。即使上下文很干净，Agent有时也会钻牛角尖。

例如，Agent可能已经发现了代码问题，却连续十几次用脚本去验证无关逻辑，而不是直接运行修改后的测试用例。此时，动作投影会在动作真正下发给环境前进行评估：




{% raw %}$$ P_{act}(s, q, H_t, a_t) = (d_t, \rho_t) $${% endraw %}



这里的 $d_t \in \{Pass, Reject\}$。如果动作被判定为低价值或与历史轨迹相悖，HarnessBridge会将其拦截（$Reject$），不再消耗环境步数。

同时，它会向Agent返回基于轨迹的反馈 $\rho_t$。这个反馈包含了具体的疑虑、支持该判断的历史证据以及可执行的改进建议，从而引导Agent迅速回到正轨。

### 统一指令微调与数据构建

为了训练这个“智能副官”，该研究并没有设计两个独立的网络，而是采用了**统一指令微调**（**Unified Instruction Fine-tuning**）。

通过将双向投影任务统一转化为条件生成问题，研究人员在一个轻量级的Qwen3.5-0.8B模型上进行了微调。

在数据构建方面，本文利用指令微调的模型在不同干预机制下生成轨迹，并通过强力LLM裁判进行严格过滤。只有那些压缩忠实、拦截证据确凿的样本才会被保留，从而保证了HarnessBridge不会产生幻觉或抹除关键信息。

### 核心实验与泛化表现

实验结果展示了极强的工程吸引力。在SWE-bench Verified和Terminal-Bench 2.0这两大硬核评测基准上，HarnessBridge展现出了压倒性的优势。

在Token消耗和成功率的联合考量上，HarnessBridge打破了常规脚手架的性能天花板。以最新的GPT-5.4-Nano为例，接入HarnessBridge后，其在终端任务上的成功率从15.7%跃升至22.5%。

更惊人的是其效率提升：平均Token消耗从9.80M断崖式下跌至0.91M，降幅超过90%。即便是对于极其强大的GPT-5.4，在维持53.9%成功率不变的前提下，Token使用量也削减了约89.5%。

此外，尽管HarnessBridge仅在Qwen模型的SWE-bench轨迹上进行过训练，它却能完美泛化到Terminal-Bench这样完全不同的环境，并且能直接适配Claude、DeepSeek等未见过的商业模型。这证明它学到的是底层的高效交互范式，而非对特定生成器的过拟合。

### 机制分析：它到底学到了什么？

本文对HarnessBridge的内部行为进行了细致的分类统计，揭示了其聪明的处理策略。

在观察投影中，它表现出了极强的选择性。对于那些信息密度高但极易过时的动作（如纯推理过程、文件浏览、搜索），其压缩率高达20%至40%。相反，对于包含决定性验证信号的测试执行输出，压缩率仅为3.1%。

这说明模型并非在无脑截断，而是懂得区分“瞬时探索信息”与“长效验证信息”。同时，高压缩率的类别往往也是高活动状态提取率的类别，这意味着它将长文本压缩后，精准地把精华提炼进了置顶备忘录中。

### 结论与工程启示

HarnessBridge将人工编写的静态交互逻辑升格为了端到端可学习的策略。

它为行业带来了重要的工程启示：在无脑扩大**大语言模型**（**Large Language Model, LLM**）的上下文窗口之外，我们其实有更具性价比的选择。用一个极其廉价的轻量级模型作为前置网关，实时压缩历史并拦截低智操作，不仅能节省巨额的API费用，还能显著提升主模型的决策质量。

当然，该方法也存在局限性。当前的高质量监督数据极度依赖于成功轨迹的提炼，如果在某些极其开放且罕见的环境中，如何自动化地收集到高质量的拦截反馈与压缩范例，仍是未来需要攻克的难题。
