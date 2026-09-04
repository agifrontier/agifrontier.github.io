---
layout: default
title: "SkillComposer：Agent技能自我进化，跨级提升模型4.5分"
description: "在当前的智能体（Agent）开发中，开发者常常陷入一个两难境地。如果为特定任务编写高度定制的策略，这些技能往往无法迁移到其他场景。反之，如果提取过于抽象的通用技能，在面对具体任务时又显得指导不足。这种“特化”与“泛化”的矛盾，一直是限制智能体在复杂推理任务中表现的瓶颈。"
arxiv_id: "2606.06079"
topics:
  - "AI Agent"
  - "基础模型"
tags:
  - "AppWorld"
  - "LiveCodeBench v6"
  - "SkillComposer"
  - "create/improve/merge"
  - "offline/online/hybrid deployment"
  - "rejection sampling"
related_tutorials:
  - "training-task-reasoning-llm-agents-for-multi-turn-task-planning-via-single-turn-"
  - "remember-me-refine-me-a-dynamic-procedural-memory-framework-for-experience-drive"
  - "skillos-learning-skill-curation-for-self-evolving-agents"
  - "resource2skill-distilling-executable-agent-skills-from-human-created-multimodal-resources"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">SkillComposer: Learning to Evolve Agent Skills for Specification and Generalization</p>

在当前的**智能体**（**Agent**）开发中，开发者常常陷入一个两难境地。

> **ArXiv URL**：http://arxiv.org/abs/2606.06079v1

如果为特定任务编写高度定制的策略，这些技能往往无法迁移到其他场景。
反之，如果提取过于抽象的通用技能，在面对具体任务时又显得指导不足。

这种“特化”与“泛化”的矛盾，一直是限制智能体在复杂推理任务中表现的瓶颈。
现有的技能构建方法大多将其视为一次性的抽取过程，缺乏对技能质量的动态控制。

为了打破这一僵局，来自新加坡国立大学、同义实验室和浙江大学的研究团队提出了一项名为 SkillComposer 的全新框架。
该框架让语言模型能够在推理阶段实现技能的“自我进化”。
它不仅能有效解决特化与泛化的张力，还成功让一个小参数量的模型跨级别提升了 27B 大模型的表现。

### 技能构建的根本张力

Agent 技能通常由可复用的自然语言指令组成，包含名称、触发条件和具体执行步骤。
通过将这些技能加载到上下文中，模型无需更新参数即可利用过往经验。

然而，当前的技能库往往依赖高质量的人工编写，或者从成功的历史轨迹中生硬提取。
当模型尝试自主生成技能时，由于缺乏质量控制和抽象机制，往往会导致负面收益。

研究团队深刻洞察到，高质量的技能演进需要两个正交的维度：
一方面是**泛化**（**Generalization**），即在相似任务中提取可迁移的通用策略；
另一方面是**特化**（**Specification**），即将技能精准适配到特定任务的模式中。
现有的静态抽取方法完全无法兼顾这两者。

### 核心机制：三阶技能进化

为了实现技能的动态演进，本文将技能构建过程解耦为三个可学习的核心操作。
我们可以用“打造专属工具”的过程来类比这三个操作，以直观理解其内在逻辑：

**第一阶：创建（Create）—— 初步打样**
当智能体在没有任何技能指导下完成任务时，SkillComposer 会从原始的执行轨迹中提取出基础的过程性知识，形成初始技能。
这就像铁匠根据一次成功的操作，初步打制出一把原型工具。
公式化表示为：


{% raw %}$$ s={\rm SkillComposer\_{create}}(x,{\rm LLM}(x)) $${% endraw %}


其中 $x$ 为输入任务，$s$ 为生成的技能。

**第二阶：合并（Merge）—— 熔炼与泛化**
随着技能库的增长，会出现许多相似但局部的技能。
合并操作会将两个语义相似的技能 $s_1$ 和 $s_2$ 融合成一个更广泛、更具迁移性的通用技能。
这就如同铁匠将两把功能单一的旧工具熔炼，打造出一把多功能的瑞士军刀，从而驱动了技能的“泛化”。


{% raw %}$$ s={\rm SkillComposer\_{merge}}(s\_{1},s\_{2}) $${% endraw %}



**第三阶：改进（Improve）—— 打磨与特化**
当智能体使用现有技能 $s_{\rm o}$ 解决新任务时，可能会发现原技能存在覆盖不到的细节。
改进操作会基于新的执行反馈，对原有技能进行精细化调整。
这就像铁匠根据工人在特定流水线上的反馈，对工具的握把进行专门打磨，从而实现了技能的“特化”。


{% raw %}$$ s={\rm SkillComposer\_{improve}}(x,s\_{\rm o},{\rm LLM}(x,s\_{\rm o})) $${% endraw %}



### 多模态部署策略

基于上述三种原子操作，SkillComposer 能够灵活应对不同的实际应用场景，并衍生出三种部署模式：

<img src="/images/2606.06079v1/x1.webp" alt="Refer to caption" style="width:85%; max-width:450px; margin:auto; display:block;">

1. **离线模式（Offline）**：利用训练集，通过反复的“创建”和“合并”构建一个通用的技能库。在推理时，模型直接检索并使用相关技能。
2. **在线模式（Online）**：从零开始，面对新流任务时即时“创建”并不断“改进”技能。这种模式无需先验数据，极大地驱动了特化能力。
3. **混合模式（Hybrid）**：结合两者优势，先用离线库初始化，再通过在线演进进行任务特化。

### 训练秘方：拒绝采样与性能反馈

虽然大语言模型本身具备零样本执行上述操作的能力，但输出质量参差不齐。
为了强化这三种能力，研究团队引入了**拒绝采样**（**Rejection Sampling**）机制来合成监督微调数据。

这里的核心判定标准是“**增量通过率**（**Delta Pass Rate**）”。
只有当候选技能让执行器的首选通过率（$pass@1$）相对于基线提升超过特定阈值 $\epsilon$ 时，该样本才会被接受并用于训练。

例如，对于创建操作的性能增量计算如下：


{% raw %}$$ \Delta\_{\text{create}} =\text{pass@}1(\text{LLM}(x,s)) \quad-\text{pass@}1(\text{LLM}(x))\geq\epsilon\_{\rm c} $${% endraw %}



通过这种严苛的收益驱动过滤，模型真正学会了如何生成“有用”的技能，而不是堆砌无效的指令。

### 实验突破与核心发现

研究团队在一个 4B 参数的模型（Qwen3.5-4B）上进行了微调训练，并在多项基准测试中进行了验证。

#### 跨级提升与卓越表现
最令人瞩目的结果是其跨模型的泛化能力。
尽管 SkillComposer 的训练数据和核心控制器只有 4B 参数，但它生成的技能可以完美赋能 Qwen3.5-27B 这样更大体量的执行器。
在代理任务基准 $\tau^2$-Bench 上，在线模式让 27B 模型的表现提升了 4.5 分；
在代码生成基准 LiveCodeBench v6 上，提升了 3.4 分。

#### 合并与改进的正交性
本文对技能库进行了降维可视化分析。

<img src="/images/2606.06079v1/x3.webp" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

从上图可以看出，经过“合并（Merge）”操作后，原本密集的技能簇被有效压缩，技能库变得更加稀疏且通用，大幅减少了冗余。
消融实验证实，合并和改进各自针对技能的不同质量维度，缺一不可。

#### 迭代进化优于重复采样
在实际工程中，人们常常通过多次采样（$pass@k$）来碰运气获取正确答案。

<img src="/images/2606.06079v1/x4.webp" alt="Refer to caption" style="width:80%; max-width:300px; margin:auto; display:block;">

但如上图所示，在相同的推理预算 $k$ 次尝试下，SkillComposer 的在线迭代进化策略始终显著优于单纯的独立重复采样。
随着尝试次数的增加，这种通过结构化知识沉淀带来的优势差距会越来越大。

### 局限性与工程启示

尽管成果丰硕，该研究也坦承了目前的局限性。
基于拒绝采样的数据收集过程需要大量推理来计算通过率，计算成本高昂。
此外，目前的实验主要在 Qwen 系列模型上验证，不同架构的普适性仍待探索。

对于 AI 开发者而言，SkillComposer 提供了一个极具价值的工程启示：
不要再试图构建庞大且静态的完美 Prompt 库。
未来的 Agent 架构应当具备一套内建的“技能新陈代谢系统”。
让模型在运行中自主打样、自我熔炼、自我打磨，才是迈向更强通用人工智能的务实路径。
