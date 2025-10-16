---
layout: default
title: "LLM$\times$MapReduce-V3: Enabling Interactive In-Depth Survey Generation through a MCP-Driven Hierarchically Modular Agent System"
---

# LLM$\times$MapReduce-V3: Enabling Interactive In-Depth Survey Generation through a MCP-Driven Hierarchically Modular Agent System

- **ArXiv URL**: http://arxiv.org/abs/2510.10890v1

- **作者**: Haoyu Wang; Shuo Wang; Siyu Lin; Jie Zhou; Zhu Zhang; Zihan Zhou; Maosong Sun; Zhiyuan Liu; Yu Chao

- **发布机构**: Modelbest Inc.; Nanyang Technological University; Peking University; Tsinghua University

---

# TL;DR
本文提出了一种名为 LLM×MapReduce-V3 的分层模块化智能体系统，该系统通过模型-上下文-协议 (MCP) 驱动的动态规划，实现了交互式、可定制的深度综述论文生成。

# 关键定义
*   **LLM×MapReduce-V3**: 一个为生成长篇综述而设计的分层模块化智能体系统。它将核心功能分解为独立的、可组合的 MCP 服务器，并通过一个高级规划器智能体动态协调工作流，同时支持用户深度参与。
*   **模型-上下文-协议 (Model-Context-Protocol, MCP)**: 一种标准化的函数调用机制和开放标准，它允许将 AI 助手（智能体）连接到不同的工具源。在本文中，MCP 将各个功能模块（如骨架初始化、摘要生成）封装成独立的服务器，实现了系统的模块化和可扩展性。
*   **分层模块化智能体系统 (Hierarchically Modular Agent System)**: 一种多智能体架构，其中各个功能组件被实现为独立的“原子”服务器。这些原子服务器可以被聚合成更高级别的服务器，形成一个层次化结构，由一个高级规划器智能体进行动态编排。
*   **Orchestra Server (编排服务器)**: 系统中的一个核心组件，充当基于大语言模型的轻量级规划器。它根据当前任务的中间输出和可用工具的描述，动态生成下一步的指令，从而协调多个模块的协作，实现自适应、非线性的工作流。

# 相关工作
目前，人工智能驱动的自动化研究在信息检索和内容生成方面取得了显著进展，代表性系统有 WebGPT、Self-RAG 和 GPT-Researcher 等。然而，这些系统通常缺乏足够的用户参与和灵活性。在综述生成领域，虽然 AutoSurvey、InteractiveSurvey 等工具出现，但它们往往将用户锁定在僵化、“一站式”的工作流中，缺乏对过程的迭代优化和定制化能力。

同时，模型-上下文-协议 (MCP) 作为一种连接模型与工具的开放标准，已在 Alita、AgentDistill 等工作中展现出构建自适应智能体系统的潜力。

本文旨在解决现有综述生成系统刚性强、定制化能力弱、用户干预不足的问题。通过引入基于 MCP 的分层模块化架构和动态规划器，本文致力于构建一个开放、灵活且支持人机协作的深度综述生成系统。

# 本文方法

本文提出的 LLM×MapReduce-V3 是一个采用多智能体范式的生态系统，各个专用智能体在不同阶段处理任务，并通过 MCP 协议与一系列功能服务器进行交互。

<img src="/images/2510.10890v1/main.jpg" alt="系统工作流图" style="width:85%; max-width:600px; margin:auto; display:block;">
<center>系统智能体-服务器生态系统工作流。用户首先指定主题，系统通过分析智能体、搜索智能体、骨架智能体和写作智能体协同工作，完成文献检索、大纲构建与优化、以及最终的论文撰写。</center>

### 系统设计

系统由一组专用智能体 $\mathcal{A}=\{A\_{1}, A\_{2}, A\_{3}\}$（分析、骨架、写作智能体）和一个 MCP 服务器生态系统 $\mathcal{S}$ 组成。智能体与服务器之间的连接 $\mathcal{E}$ 是在每个工具调用轮次中动态确定的：


{% raw %}$$
\mathcal{E}=\mathrm{MCP}(A\_{i}(\mathrm{output}_{i-1},\mathrm{plan}),\phi(\mathcal{A}_{i}))
$${% endraw %}


其中，$\phi(A\_{i})$ 定义了智能体 $A\_{i}$ 可访问的服务器子集。

每个服务器 $S\_i$ 通过 MCP 协议暴露一组工具 $\mathcal{T}(S\_{i})$。智能体通过以下形式调用工具：


{% raw %}$$
\text{invoke}:\mathcal{A}\times\mathcal{T}\times\mathcal{I}\rightarrow\mathcal{O}
$${% endraw %}


其中 $\mathcal{I}$ 和 $\mathcal{O}$ 分别是输入和输出空间。

*   **分析智能体 (Analysis Agent)**: 负责解析用户意图，通过多轮对话与用户达成共识，明确研究视角。随后，它协调**搜索智能体 (Search Agent)** 获取参考文献，并调用**分组服务器 (Group Server)** 对文献进行聚类，构建结构化的知识树。
*   **骨架智能体 (Skeleton Agent)**: 负责构建和优化综述的全局大纲。在**编排服务器 (Orchestra Server)** 的协调下，它依次执行大纲初始化、摘要生成和多轮优化，形成最终的精细化大纲。
*   **写作智能体 (Writing Agent)**: 根据精炼后的大纲，综合利用文献摘要，生成连贯的综述章节，并保证引文的准确性和学术写作规范。
*   **可替换的外部智能体**: 系统设计具有高度可扩展性。例如，**搜索智能体**可以被无缝替换为其他实现（如特定领域的数据库或机构知识库）。用户还可以定义自己的智能体（如学术批判、格式化智能体）来满足特定需求。

### MCP 实现框架
本文方法的核心创新在于将 LLM×MapReduce-V2 的过程重构为一系列独立的、可组合的 MCP 服务器，并通过一个动态规划器进行智能编排。

*   **原生服务器构建**:
    *   **分组服务器 (Group Server)**: 对检索到的文献进行主题和方法上的聚类，为后续大纲构建提供一个结构化的输入。
    *   **编排服务器 (Orchestra Server)**: 系统的“大脑”，是一个基于 LLM 的动态规划器。它根据当前状态和可用工具，决定下一步调用哪个服务器，实现了自适应的非线性工作流。
    *   **骨架初始化服务器 (Skeleton Initialization Server)**: 根据用户确定的研究角度和分组后的文献，生成一个高层次的章节大纲。
    *   **摘要服务器 (Digest Server)**: 为每篇参考文献生成摘要和对现有大纲的改进建议，为大纲的优化提供内容感知的修订信号。
    *   **骨架优化服务器 (Skeleton Refine Server)**: 采用受卷积神经网络 (Convolutional Neural Networks, CNN) 启发的多层迭代过程，对大纲进行内部（节内）和跨部（节间）的优化，以增强其连贯性、一致性和信息量。

*   **通过多轮工具使用的迭代优化**:
    该优化过程是一个多轮、基于工具的自演化框架。**编排服务器**实现了一个规划函数 $\pi$，它根据执行历史 $\mathcal{H}$ 和当前上下文 $\mathcal{C}$ 来选择最优的工具序列 $\mathcal{T}^{\*}$：
    

    {% raw %}$$
    \pi:\mathcal{H}\times\mathcal{C}\rightarrow\mathcal{T}^{*}
    $${% endraw %}


    在每个决策点 $t$，给定当前的大纲状态 $x^{(t)}$ 和历史 $h^{(t)}$，智能体选择一个动作 $\nu^{(t)}=\pi(x^{(t)},h^{(t)})$，并通过状态转移函数 $x^{(t+1)}=f(x^{(t)},u^{(t)})$ 对大纲进行迭代更新。这种机制使得骨架智能体能够根据实时评估结果，自适应地调用摘要服务器和骨架服务器，逐步提升大纲质量。

### 人机交互
系统在关键决策点引入了人机交互，以确保生成内容与用户目标对齐。
*   **达成共识**: 在初始阶段，系统通过多轮对话与用户互动，共同确定研究范围和核心视角。
*   **反馈整合**: 在大纲生成后，系统会呈现给用户进行审查和修改。用户可以提出结构调整、内容侧重等建议。用户可在任意阶段评估中间产出，其反馈会影响后续模块的执行。

# 实验结论
本文将 LLM×MapReduce-V3 与其他主流的深度研究和综述生成系统进行了功能对比和人工评估。

**功能对比**
如下表所示，相比于其他系统，本文提出的 LLM×MapReduce-V3 是首个全面整合了深度用户交互、模块化设计、MCP 标准化、自定义工具集成以及综述任务特定优化的开放解决方案。


| 系统 | 用户交互 | 模块化 | MCP集成 | 自定义工具 | 综述优化 | 开源 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Perplexity DR | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ | $\times$ |
| Gemini DR | $\checkmark$ | $\times$ | $\times$ | $\times$ | $\times$ | $\times$ |
| WebGPT | $\checkmark$ | $\sim$ | $\times$ | $\times$ | $\times$ | $\times$ |
| ResearchAgent | $\checkmark$ | $\checkmark$ | $\times$ | $\sim$ | $\times$ | $\sim$ |
| CoSearchAgent | $\sim$ | $\sim$ | $\times$ | $\times$ | $\times$ | $\times$ |
| Search-o1 | $\times$ | $\sim$ | $\times$ | $\times$ | $\times$ | $\checkmark$ |
| CrewAI | $\sim$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$ | $\checkmark$ |
| Alita | $\times$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\times$ | $\checkmark$ |
| AutoSurvey | $\times$ | $\times$ | $\times$ | $\times$ | $\checkmark$ | $\checkmark$ |
| SurveyX | $\sim$ | $\sim$ | $\times$ | $\times$ | $\checkmark$ | $\checkmark$ |
| **LLM×MapReduce-V3 (本文)** | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ | $\checkmark$ |

*注：$\checkmark$ = 完全支持; $\sim$ = 有限支持; $\times$ = 不支持*

**人工评估**
研究招募了五名领域专家，对本文系统、Gemini DeepResearch 和 Manus AI 在十一个主题上生成的文章进行评估。评估标准包括：大纲质量、内容长度和整体质量。

结果表明，与竞品相比，本文系统生成的文章在文献综述方面覆盖面更广，内容长度显著更长，并且在内容深度、结构连贯性和流畅性方面均表现出强大的性能，获得了专家评审的高度评价。

**最终结论**
LLM×MapReduce-V3 引入了一种基于 MCP 的模块化架构，成功克服了传统封闭式智能体系统的刚性。通过支持开放集成的可定制智能体和服务器，系统实现了前所未有的灵活性和可扩展性。其“人在环路”的设计确保了产出与人类专家意图的高度对齐，在综述生成任务上取得了卓越表现，并展示了其在更广泛知识密集型任务中的应用潜力。