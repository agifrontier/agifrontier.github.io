---
layout: default
title: "告别“单机时代”：首篇AI Agent通信协议双维分类指南"
description: "当无数大语言模型驱动的Agent在企业客服、代码生成和数据分析领域遍地开花时，一个致命的系统性瓶颈已悄然浮现。这些Agent虽然具备强大的推理与执行能力，但它们大多是彻头彻尾的“语言孤岛”。不同厂商、不同架构的Agent各自为战，无法顺畅调用外部数据，更难以实现多智能体间的无缝协作。"
arxiv_id: "2504.16736"
topics:
  - "AI工程"
  - "RAG"
tags:
  - "AI agent protocols"
  - "LLM agents"
  - "collective intelligence"
  - "context-oriented protocols"
  - "domain-specific protocols"
  - "inter-agent communication"
related_tutorials:
  - "agent-data-protocol-unifying-datasets-for-diverse-effective-fine-tuning-of-llm-a"
  - "transformer-enhanced-relation-classification-a-comparative-analysis-of-contextua"
  - "toward-general-purpose-robots-via-foundation-models-a-survey-and-meta-analysis"
  - "personal-llm-agents-insights-and-survey-about-the-capability-efficiency-and-security"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">A Survey of AI Agent Protocols</p>

当无数大语言模型驱动的Agent在企业客服、代码生成和数据分析领域遍地开花时，一个致命的系统性瓶颈已悄然浮现。
这些Agent虽然具备强大的推理与执行能力，但它们大多是彻头彻尾的“语言孤岛”。
不同厂商、不同架构的Agent各自为战，无法顺畅调用外部数据，更难以实现多智能体间的无缝协作。
这种碎片化的现状，与TCP/IP协议诞生前的早期互联网如出一辙，极大限制了算力与智能的规模化涌现。

> **ArXiv URL**：http://arxiv.org/abs/2504.16736v3

如果不能建立统一的通信标准，多智能体网络（Multi-Agent System）就永远只存在于实验室中。
本文首次对当前浩如烟海的AI Agent通信协议进行了全面摸底，提出了一套严谨的双维分类框架。
该研究不仅为开发者梳理了底层逻辑，更揭示了迈向大规模分布式集体智能（Collective Intelligence）的核心路径。

<img src="/images/2504.16736v3/x1.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

### 架构瓶颈

在讨论专门的通信协议前，一个不可回避的疑问是：我们为什么不能直接用传统的API或图形界面？
传统API调用虽然效率极高，但缺乏动态适配的灵活性，且标准碎片化严重。
图形用户界面（GUI）虽然对人类极度友好，但它并非为AI系统的原生交互而设计，解析成本极高。
至于基于XML或HTML的网页抓取方式，其脆弱性和复杂性阻碍了Agent向更广泛场景的扩展。

相比之下，专用的Agent协议结合了高效率、操作边界宽泛以及AI原生兼容三大优势。
我们可以将Agent通信协议理解为一套严密的“跨国外交规则”。
没有规则，各方只能鸡同鸭讲；有了规则，才能实现精准的情报交换与联合行动。

### 分类范式

为解决协议领域的混乱，本文提出了一种系统性的双维分类架构。
第一维度考察**交互对象**（**Object Orientation**）。
该维度区分了协议是致力于解决Agent与外部环境的连接（上下文导向），还是致力于解决Agent与Agent之间的沟通（智能体间协作）。
第二维度考察**应用场景**（**Application Scenario**）。
该维度将协议划分为适用于广泛任务的通用协议，以及针对特定环境定制的特定领域协议。

这种分类犹如为“外交规则”确立了基本法。
有的规则专门指导外交官如何查阅当地档案，有的规则约束多国大使如何同桌谈判。
针对不同的业务诉求，开发者能够借此迅速定位到最契合的技术栈。

### 外脑挂载

大型语言模型在处理实时数据或私有领域知识时，往往存在严重的幻觉或知识盲区。
因此，Agent需要频繁向外伸手，挂载“外脑”以获取执行任务所需的上下文。
在通用上下文导向协议中，Anthropic推出的**模型上下文协议**（**Model Context Protocol, MCP**）堪称行业标杆。

MCP采用经典的客户端-服务器架构，将大模型底座与海量的外部资源接口进行了解耦。
通过这套协议，Agent不需要针对每一个新工具去重写提示词（Prompt）工程。
它极大地提升了系统的互操作性与安全性，是当前构建稳健智能体系统的基石。

而在特定领域，开源的`agents.json`规范提供了另一种轻量级解法。
它构建于OpenAPI标准之上，允许网站通过结构化的JSON文件，向Agent直接声明其AI兼容接口。
这种设计最大限度地保留了传统API的无状态特性。
它引入了工作流（Flows）概念，将复杂的多步API调用转化为大模型极易消化的依赖链路，显著降低了开发门槛。

### 互操作性

如果说上下文协议解决了Agent“借用工具”的问题，那么智能体间协议则直面了更复杂的“跨阵营协作”挑战。
这里不再是单向的数据索取，而是涉及身份认证、意图对齐和状态同步的平等协商。

开源社区力推的**智能体网络协议**（**Agent Network Protocol, ANP**），其野心在于复刻人类互联网的崛起。
它致力于定义标准的连接机制，为数以十亿计的异构Agent构建一个安全高效的协作网络。
Google提出的**A2A协议**（**Agent2Agent**）则更加务实。
它专注于企业级环境下的无缝集成，标准化了能力发现、用户体验协商以及任务状态管理。

特别值得一提的是**Agora协议**。
在异构模型网络中，系统往往面临着通用性、效率与可移植性之间的“通信不可能三角”。
Agora允许Agent根据当前任务的频次与复杂度，自主协商通信方式。
对于高频任务，Agent可选用高效的结构化协议以降低计算延迟。
对于未知或复杂场景，它们则平滑切换至自然语言进行沟通，完美平衡了计算成本与通信灵活性。

此外，还有专注跨信任边界价值交换的AITP协议。
它结合了区块链技术，允许Agent在不同组织间安全地传递结构化数据。
这些通用协议共同编织了一张超越单机算力的庞大协同网络。

### 领域特化

当Agent走出数字世界，或者需要与人类进行高风险决策互动时，通用协议便显得力不从心。
特定领域的通信规则开始发挥不可替代的作用。

在人机交互领域，**PXP协议**（**Predict and eXplain Protocol**）专门为提升双向可理解性而生。
它引入了严格的有限状态机模型。
Agent在交互时，必须为输出打上“确认”、“反驳”、“修改”或“拒绝”这四个标准化标签。
无论是放射科诊断还是药物合成路径规划，PXP都确保了人类专家与AI之间的认知对齐。

而在机器人交互（Robot-Agent）场景中，物理空间的坐标与几何共识成为了协议的核心考量。
例如**空间群体协议**（**Spatial Population Protocols, SPPs**）专门解决分布式机器人的定位问题。
通过该协议，即使初始坐标系完全混乱，机器人也能仅通过两两交互（询问距离或向量），最终就全局坐标系达成共识。
这类协议深刻体现了具身智能在空间推理上的严苛要求。

在系统级治理层面，**语言模型操作系统**（**Language Model Operating System, LMOS**）更是展现了宏大的底层重构愿景。
它将架构解耦为应用协议层、传输层与身份安全层。
通过引入万维网联盟的**去中心化标识符**（**Decentralized Identifiers, DIDs**），LMOS确保了跨组织交互的绝对可信。

<img src="/images/2504.16736v3/development2.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

### 工程启示

纵观各类Agent协议的演进，我们发现单纯堆砌参数和算力已经无法满足复杂场景的需求。
对于一线工程师与研究者而言，未来的系统架构设计必须跨越几道核心门槛。

首先是分层架构的必然性。
未来的Agent网络不会依赖单一的超级协议，而是像OSI七层模型一样层层嵌套。
底层通过区块链或零知识证明解决身份与隐私保护问题，上层则专注于语义对齐与意图解析。

其次是协议的动态适应能力。
静态的接口声明最终会被淘汰。
例如在处理海量长文本或进行多步骤推理时，设模型的输出状态序列为 $x_1, x_2, ... x_n$，协议必须能够实时监控生成过程。
下一代网络需要Agent能够根据延迟、算力带宽和安全约束，动态“降级”或“升级”其通信策略。

最后，群组协作基础设施（Group-based Interaction）将成为下一波红利。
我们不再仅仅关注两个Agent如何对话，而是关注数百个微型Agent如何自发形成临时联盟。
真正的集体智能，正是孕育在这些日益完善、看似枯燥的标准化规则与连接之中。
告别“单机时代”，拥抱智能体互联网，才是通向通用人工智能（AGI）最坚实的工程阶梯。
