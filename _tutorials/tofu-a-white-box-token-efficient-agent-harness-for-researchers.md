---
layout: default
title: "省28.4% Token！开源Agent框架ToFu性能反超Claude"
description: "当前的智能体系统正面临一个尴尬的瓶颈：即便是回复一句简单的“你好”，系统底层也可能消耗成千上万个Token。对于复杂的多步推理任务，这种累积的计算成本更是高得令人咋舌。与此同时，市面上成熟的Agent产品（如ClaudeCode）多为闭源的黑盒系统。"
arxiv_id: "2607.11423"
topics:
  - "AI Agent"
tags:
  - "ToFu"
  - "agentic harness"
  - "local deployment"
  - "multilingual capability"
  - "orchestration code"
  - "token efficiency"
related_tutorials:
  - "what-limits-agentic-systems-efficiency"
  - "budget-aware-tool-use-enables-effective-agent-scaling"
  - "the-harness-effect-how-orchestration-design-sets-the-token-economics-of-enterprise-agentic-ai"
  - "tthe-test-time-harness-evolution"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">ToFu: A White-Box, Token-Efficient Agent Harness for Researchers</p>

当前的智能体系统正面临一个尴尬的瓶颈：即便是回复一句简单的“你好”，系统底层也可能消耗成千上万个Token。对于复杂的多步推理任务，这种累积的计算成本更是高得令人咋舌。

> **ArXiv URL**：http://arxiv.org/abs/2607.11423v1

与此同时，市面上成熟的Agent产品（如Claude Code）多为闭源的黑盒系统。这种由服务端静默更新的专有模型，往往让注重隐私和可复现性的研究人员望而却步。

为了打破这一僵局，美团、牛trans研究团队以及东北大学联合推出了全新的开源研究型智能体框架——ToFu。

该研究在SWE-bench Verified基准测试中证明，ToFu不仅在代码修复准确率上超越了Claude Code，其平均Token消耗更是大幅降低了28.4%。

本文将深入解析ToFu框架的核心架构与关键机制，探究它是如何在限制上下文膨胀的同时，实现性能逆势上扬的。

### 框架概览与定位

构建一个强大的Agent系统，**大型语言模型**（**Large Language Models, LLMs**）只是其中一环。决定Agent行为逻辑的，是围绕模型构建的外围编排代码，即Harness（框架/线束）。

如果把LLM比作一位聪明绝顶但缺乏记忆的“CEO”，那么Harness就是统管一切的“总经办”。

这个总经办负责整理历史档案（上下文管理）、调度各类工具（执行环境）、把控任务进度，并确保送呈给CEO的报告精简且切中要害。

ToFu作为一个白盒研究工具，将其“总经办”架构拆解为六大核心模块：

<img src="/images/2607.11423v1/x4.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

1.  **用户接口**：支持Web UI与社交平台机器人接入，允许用户干预推理过程，提供检查点恢复与高风险操作的人工审批。
2.  **智能体编排核心**：整个系统的控制中枢，负责任务拆解、协调“规划-工作-评估”的推理流，并管理工具调用与错误回退。
3.  **模型抽象层**：提供统一的网关，抹平不同大模型厂商在API和流式协议上的差异，使得系统可以灵活切换不同的底层模型。
4.  **能力运行环境**：执行实际工具调用的沙盒。ToFu通过**模型上下文协议**（**Model Context Protocols, MCPs**）原生支持了GitHub、Notion等外部应用的无缝接入。
5.  **状态与知识**：持久化的存储层，负责记录长期记忆、执行历史与可复用的技能。
6.  **上下文管理**：构建发送给模型的短期工作记忆，其核心目标是在控制Token成本的同时，最大化信息密度。

在这六大模块中，ToFu真正拉开与竞品差距的，是其在“信息密度把控”上的精细设计。

### 核心机制分析

要在多轮对话中控制Token用量，简单粗暴的截断会丢失关键信息。ToFu采用了一套更聪明的策略组合。

#### 三层上下文压缩

当CEO（模型）处理超长任务时，“总经办”不能把过去十几个小时的会议记录原封不动地递交上去。ToFu设计了三层上下文压缩机制。

最核心的是**查询感知语义压缩**（**Query-aware Semantic Compaction**）。

当上下文窗口接近可用极限时，ToFu会唤醒一个轻量级模型（例如 GPT-4o-mini）作为秘书。

这位秘书会针对当前的最新查询，对历史轮次进行重写：高度保留关键技术决策，压缩常规的有用信息。

对于偏离当前主线的无关对话，则直接剔除。而当前正在进行的最核心对话轮次，始终保持完整不变。

这种压缩方式生成了紧凑的工作状态摘要，使任务能够在极低的信息损耗下继续推进。

#### 记忆与检索子系统

除了短期压缩，对于跨任务的长期知识，ToFu摒弃了传统的全局注入模式。

记忆被存储为结构化的Markdown记录，并附带元数据标签。ToFu不会将所有记忆硬塞进每次请求的Prompt中，而是仅向模型提供一个紧凑的“可用记忆指示器”。

模型在推理时，按需调用检索工具提取特定记忆。这就好比总经办只给CEO提供了一份目录，需要详细数据时再去档案室调取。

#### 多语言增强与群集调度

对于非英语用户，ToFu引入了“先翻译后推理”的工作流。

当用户输入非英语请求时，系统内部会将其转换为英语，交由英文能力更强的模型主干进行推理，最后再将结果翻译回母语展示。

在这个过程中，代码块和特殊标记的部分会被严格保护，避免语义损坏。

面对复杂任务，ToFu还配备了依赖感知的**有向无环图**（**Directed Acyclic Graph, DAG**）群集调度器。

多个子智能体可以并行处理独立任务，而存在依赖关系的子任务则会自动等待前置结果，极大提升了执行效率。

### 学术写作的得力助手

值得一提的是，ToFu不仅仅是一个代码修复工具。它深入集成了学术界常用的工具链。

通过开源的Overleaf MCP适配器，ToFu可以直接连接到用户的Overleaf项目空间。

它可以读取LaTeX源码、分析文档结构、修改摘要、管理辅助文件，甚至直接触发编译并获取PDF日志。

这使得ToFu能够覆盖从阅读文献（内置论文阅读器支持双栏PDF解析）到修改手稿的完整学术闭环。

<img src="/images/2607.11423v1/x3.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

### 实验结论：更少的Token，更强的性能

该研究在SWE-bench Verified数据集上，对ToFu、闭源的Claude Code以及开源的OpenCode进行了全面对比。

实验选用了Claude opus 4.6、GLM 5.1和DeepSeek-v4-pro作为底层模型。

结果显示，在三个底层模型上，ToFu均取得了最高的 $Pass@1$ 准确率。

相较于Claude Code，ToFu的平均修复通过率提升了3.8个百分点。更令人瞩目的是，ToFu的平均Token使用量大幅降低了28.4%（在某些模型上甚至降低了43.6%）。

虽然OpenCode在Token消耗上偶有优势，但其任务成功率始终垫底，暴露出成本与性能的严重失衡。

这一数据揭示了一个重要的反直觉现象：**更多的计算量并不意味着更好的性能**。

在测试时扩展（Test-time scaling）的浪潮下，无脑增加Token注入反而可能引入噪声。ToFu用更精简的上下文赢得了更高的准确率，证明了Harness设计的巨大优化空间。

### 局限性与工程启示

尽管ToFu在代码场景下展现了卓越的性能，但本文也坦诚了其现阶段的局限性。

首先，针对广泛科研助手场景的评估目前仅限于小规模的人类偏好研究，缺乏大规模真实用户测试的数据支撑。

其次，对于Token效率与任务通过率之间的定量权衡关系，目前尚未建立系统的分析模型，这依然是一个值得深挖的学术空白。

从工程落地的角度来看，ToFu为行业提供了两点重要启示：

1. **上下文质量优于数量**：暴力堆叠长文本窗口不仅昂贵，而且容易分散模型的注意力。通过语义压缩和按需检索构建高密度的短期记忆，是提升Agent鲁棒性的关键。
2. **白盒框架是创新的基石**：依赖黑盒商用API进行Agent架构研究，宛如在流沙上建塔。ToFu在MIT协议下的开源，为研究人员评估编排逻辑、工具调用策略提供了一个坚实且可控的基准。
