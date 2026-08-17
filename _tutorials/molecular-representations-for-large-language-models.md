---
layout: default
title: "牛津大学提出MolJSON：大模型分子推理准确率飙升至98.5%的全新格式"
description: "当最先进的大语言模型试图解开复杂化学反应的奥秘时，它们却经常在最基础的环节栽跟头。为什么一个能写出完美代码的AI，却连简单的环状分子结构都容易认错？该研究指出，罪魁祸首并非模型本身的推理能力不足，而是我们给模型喂入数据的“方言”不对。长期以来，化学界习惯使用现成的文本格式来表示分子结构。"
arxiv_id: "2605.01822"
topics:
  - "基础模型"
tags:
  - "GPT-5"
  - "IUPAC"
  - "LLMs"
  - "MolJSON"
  - "SMILES"
  - "constrained generation"
related_tutorials:
  - "zero-shot-performance-prediction-for-probabilistic-scaling-laws"
  - "scaling-and-context-steer-llms-along-the-same-computational-path-as-the-human-br"
  - "peek-context-map-as-an-orientation-cache-for-long-context-llm-agents"
  - "toward-general-purpose-robots-via-foundation-models-a-survey-and-meta-analysis"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">Molecular Representations for Large Language Models</p>

当最先进的大语言模型试图解开复杂化学反应的奥秘时，它们却经常在最基础的环节栽跟头。
为什么一个能写出完美代码的AI，却连简单的环状分子结构都容易认错？

> **ArXiv URL**：http://arxiv.org/abs/2605.01822v1

该研究指出，罪魁祸首并非模型本身的推理能力不足，而是我们给模型喂入数据的“方言”不对。
长期以来，化学界习惯使用现成的文本格式来表示分子结构。
但这些专为传统计算工作流和人类阅读设计的格式，让大语言模型（**Large Language Model**, **LLM**）吃尽了苦头。

为此，牛津大学的研究团队提出了一种专为LLM设计的全新分子表示格式：**MolJSON**。
在包含超过七万个问题的严苛测试中，MolJSON彻底颠覆了传统格式的统治地位。
作为输入格式，它让GPT-5在最短路径推理任务中的准确率飙升至98.5%。
这一突破不仅为AI化学助手扫清了认知障碍，更揭示了数据表示形式对模型性能的决定性影响。

### 传统序列化格式的认知枷锁

在探讨MolJSON的设计机制之前，我们需要明白现有化学信息学格式为何让大模型感到棘手。
分子在本质上是图结构（**Graph**），其中原子是节点，化学键是连通节点的边。
为了让基于序列处理的LLM能够读取分子，科学家通常需要将图结构“展平”成一维文本。

以最经典的 `SMILES` 格式为例，它通过图的遍历来生成连续的字符串。
这种做法会在文本中引入复杂的括号来表示结构分支，并使用数字来表示环的闭合点。
用大白话说，这就像是让大模型根据极其晦涩的“左转右转”纯文字指令，在脑海中硬生生拼凑出一张复杂的迷宫地图。
这种语法规则不仅对大模型极不直观，而且在处理多环复杂分子时，极易产生解析灾难。

而基于化学命名规则的 `IUPAC` 格式同样充满了繁杂的前缀、后缀和位置编号。
至于 `InChI` 和 `SELFIES` 等格式，在研究团队的初期评估中也暴露出了严重的兼容性问题。
这些格式强加的句法和语义约束，严重消耗了模型原本可用于逻辑推理的算力。

### MolJSON：回归图本质的直白表达

既然让模型“脑补”图结构如此困难，为何不直接把图的节点和边喂给它？
这正是MolJSON设计的核心哲学。
MolJSON放弃了将图线性化的执念，转而采用一种高度结构化的 JSON 模式（**JSON Schema**）。

在MolJSON中，每个分子被清晰地拆解为两个基础组件。
第一个是 `atoms` 数组，每个条目明确记录一个原子的唯一标识符和对应的元素符号。
第二个是 `bonds` 数组，直接指明哪两个原子标识符之间存在何种阶数的化学键。

这种显式的图编码方式，彻底解除了传统格式强加给分子结构的重重约束。
更巧妙的是，它完美契合了当前各大主流LLM广泛支持的结构化输出（**Structured Output**）模式。
模型不需要再去学习如何匹配括号，也不需要理解复杂的环闭合编号。
为了确保化合价的正确分配，MolJSON还设计了 `charges` 和 `aromatic_n_h` 两个可选的稀疏字段。

### 多维度基准测试下的性能跃升

为了验证MolJSON的实际威力，研究团队设计了一套专门旨在隔离“表示形式影响”的基准测试。
测试对象涵盖了GPT-5-nano、GPT-5-mini、GPT-5以及Claude Haiku 4.5等多款前沿模型。
这套测试不考察深奥的化学专业知识，而是直接考验模型的“结构读写与空间导航能力”。

**分子翻译任务：消除格式转换壁垒**

在分子翻译（**Molecular Translation**）任务中，模型需要将一种格式的分子无损转换为另一种格式。
实验结果揭示了令人震惊的性能差异。
当GPT-5尝试将 `IUPAC` 名称转换为 `MolJSON` 时，准确率达到了71.0%。
然而，当同样的输入要求被转换为 `SMILES` 时，准确率暴跌至43.7%。

<img src="/images/2605.01822v1/fig-translation-large-matrix-row-low.jpg" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">

这充分说明，生成符合严苛语法的 `SMILES` 字符串，是极具挑战性的。
反过来，当MolJSON作为输入格式时，它同样帮助模型在生成其他格式时取得了更优的翻译表现。

**最短路径推理：以更少推理开销获得更高准确率**

最短路径任务（**Shortest-Path Reasoning**）要求模型计算分子中两个特定卤素原子之间的最短键数。
这要求模型必须在内部准确重构并遍历整个分子图结构。
在这项考验中，MolJSON展现出了无可匹敌的绝对优势。

<img src="/images/2605.01822v1/fig-gpt5-shortest-path-accuracy-and-tokens-two-panel-horizontal.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

以GPT-5为例，使用MolJSON作为输入时，模型成功回答了98.5%的问题。
相比之下，`SMILES` 格式的准确率为92.2%，而 `IUPAC` 仅有82.7%。
更令人瞩目的是其在推理效率上的巨大优势。
处理MolJSON格式时，模型平均只需消耗1021个输出推理Token。
而处理 `SMILES` 时，这一数字大幅飙升至1854个。
这印证了前文的比喻：让模型去解译迷宫指令极其消耗推理算力，而直接给它提供地图则省时省力。

**受限分子生成：实现对复杂拓扑的精准操控**

受限生成任务（**Constrained Generation**）高度模拟了真实场景下的结构解析应用。
模型需要根据特定的环数量和拓扑约束，凭空生成符合所有设定条件的分子。
在这个环节，MolJSON作为输出格式的可靠性得到了最大程度的验证。

<img src="/images/2605.01822v1/constrained_generation_aggregate_accuracy_by_model_format.jpg" alt="Refer to caption" style="width:85%; max-width:450px; margin:auto; display:block;">

GPT-5在生成MolJSON时，交出了95.3%的优异答卷。
而当它被强迫输出传统格式时，`IUPAC` 的准确率降至76.3%，`SMILES` 更是只有可怜的64.0%。

后续的系统性错误分析进一步表明，随着分子重原子数量的增加，传统格式的错误率急剧上升。
特别是在面对包含稠环（**Fused Rings**）或螺环的复杂拓扑系统时，`SMILES` 和 `IUPAC` 往往会彻底失效。
而MolJSON在面对这些复杂环形拓扑时，依然展现出了极强的鲁棒性。

### 局限性分析与未来工程启示

当然，作为一个极具前瞻性的初代版本，MolJSON并非完美无缺。
该研究坦言，当前的Schema设计依然存在许多可以进一步优化的空间。
例如，现有的化学键由“源”和“目标”键显式表示，但鉴于化学键的无向性质，采用双元素列表或许更为合理。
此外，目前的MolJSON还未涵盖立体化学（**Stereochemistry**）特征以及原子的三维空间坐标信息。
在处理需要精确空间构型的复杂药物设计任务时，这也可能成为限制其应用的一个短板。

尽管存在改进空间，MolJSON的横空出世依然为 AI for Science 领域带来了极其深刻的工程启示。
它无情地打破了化学信息学界长期以来对传统文本格式的深度路径依赖。
即便 `SMILES` 和 `IUPAC` 在大模型的庞大预训练语料中占据了海量篇幅，模型依然在面对完全未经专门训练的MolJSON时表现更佳。

这证明了一个残酷但在工程上极具价值的事实。
如果我们在数据表示层面上就让模型陷入了认知错位，投入再多的参数规模和推理算力也只会是事倍功半。
对于未来致力于构建自主化学智能体系统的架构师而言，结论已经非常清晰。
拥抱像MolJSON这样显式的、结构化的高级图编码，才是彻底释放大语言模型科学推理潜能的正确道路。
