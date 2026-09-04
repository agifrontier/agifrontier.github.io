---
layout: default
title: "终结Agent评测乱象！AgentBeats提出AAA新范式，吸引超700个智能体实测"
description: "随着大语言模型技术的演进，能够自主编写代码、浏览网页甚至控制电脑的Agent（智能体）正迎来爆发式增长。然而，当你开发出一个全新的Agent，想知道它到底有多强时，噩梦就开始了。当前，Agent领域的评测正处于一种极度“碎片化”的状态。"
arxiv_id: "2606.13608"
topics:
  - "AI Agent"
  - "AI评测"
tags:
  - "A2A protocol"
  - "AAA"
  - "AgentBeats"
  - "Agentified Agent Assessment"
  - "MCP protocol"
  - "agent-agnostic interface"
related_tutorials:
  - "empowering-real-world-a-survey-on-the-technology-practice-and-evaluation-of-llm-"
  - "ai-agent-systems-architectures-applications-and-evaluation"
  - "auditing-agent-harness-safety"
  - "measuring-harness-induced-belief-divergence-in-multi-step-llm-agents"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">AgentBeats: Agentifying Agent Assessment for Openness, Standardization, and Reproducibility</p>

随着大语言模型技术的演进，能够自主编写代码、浏览网页甚至控制电脑的 Agent（智能体）正迎来爆发式增长。然而，当你开发出一个全新的 Agent，想知道它到底有多强时，噩梦就开始了。

> **ArXiv URL**：http://arxiv.org/abs/2606.13608v1

当前，Agent 领域的评测正处于一种极度“碎片化”的状态。市面上的基准测试（Benchmark）通常基于固定的、以 LLM 为中心的测试框架，它们对输入格式、工具 API 和环境控制都有自己的一套假设。这意味着，如果你想在 $N$ 个基准上评测 $M$ 个 Agent，你需要编写 $N \times M$ 份定制化的集成代码。这不仅耗时耗力，还会导致测试环境与真实生产环境脱节。

为了解决这一行业痛点，卡内基梅隆大学、加州大学伯克利分校等多家顶尖机构的联合团队，提出了一种全新的评测范式：**智能体化智能体评测**（**Agentified Agent Assessment, AAA**），并推出了具体的落地系统 **AgentBeats**。该系统不仅统一了评测接口，还在一场为期五个月的公开挑战赛中，成功吸引了近800个 Agent 参与实测。

### 打破 $N \times M$ 集成地狱：AAA 评测新范式

传统的评测方式为什么难用？因为基准测试和被测 Agent 之间是“强耦合”的。评测框架需要强行接入 Agent 的内部逻辑，导致每换一个 Agent 就要改一次代码。

本文提出的 AAA 范式，其核心思想非常直观：**既然运行评测（如设置环境、提供工具、模拟用户指令、判定结果）本身就是一个标准的 Agent 任务，为什么不让基准测试也变成一个 Agent 呢？**

打个比方，传统的评测就像是为每一款新车（Agent）专门修建一条带有各种传感器的专属测试跑道（Benchmark）。而 AAA 范式则是雇佣了一位“考官”（Judge Agent），这位考官可以坐进任何一辆车里，用行业通用的对讲机（标准协议）向车辆下达指令，并观察车辆的表现。

<img src="/images/2606.13608v1/x1.webp" alt="Refer to caption" style="width:80%; max-width:300px; margin:auto; display:block;">

在 AAA 范式下，所有的互动都通过标准化的协议进行：
1.  **A2A 协议**：用于任务管理和 Agent 之间的通信。
2.  **MCP（Model Context Protocol）协议**：用于工具的访问与调用。

通过这种方式，评测逻辑与 Agent 的底层实现被完全剥离。开发者只需要确保自己的 Agent 支持 A2A 和 MCP 协议，就可以无缝接入任何支持该标准的基准测试中。集成的复杂度瞬间从 $N \times M$ 骤降到了 $N + M$。

### AgentBeats：AAA 范式的工程化落地

理论很美好，但要在复杂的真实世界中落地，还需要解决开源、隐私和可复现性等工程挑战。为此，研究团队开发了 **AgentBeats** 系统。

AgentBeats 并不是一个简单的脚本，而是一个支持完整评估生命周期的平台。考虑到有些 Agent 是开源的（如提供 GitHub 仓库或 Docker 镜像，即 Agent Blueprints），而有些则是闭源的商业服务（即 Agent Instances），AgentBeats 设计了五种操作模式：

*   **本地模式与代理模式**：适合开发者在本地调试。
*   **托管模式与云端模式**：适合集中式的公开评测打榜。
*   **持续集成（CI）模式**：完全解耦，利用 GitHub Actions 等流水线自动化运行。

这五种模式确保了无论是独立开发者还是商业公司，都能在保护隐私的前提下，以最符合自身工作流的方式参与标准化评测。

### 社区级检验：765个 Agent 的大规模实测

AAA 范式究竟通不通用？研究团队并没有停留在理论阶段，而是联合举办了一场长达五个月的公开挑战赛。

这场比赛吸引了来自全球的开发者，最终收到了 298 个 Judge Agent（覆盖编程、网页浏览、医疗、多智能体游戏等12个类别）以及 467 个接受测试的 Subject Agent。

<img src="/images/2606.13608v1/x5.webp" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">

通过对参赛代码的分析，研究人员发现了一个有趣的现象：虽然绝大多数 Agent 是用 Python 编写的，但也出现了 TypeScript 和 Rust 的实现。这证明了 AAA 范式的跨语言兼容性。此外，近 78% 的 Judge Agent 包含自然语言提示词（Prompts），这意味着开发者非常乐意通过“大白话”来定义评测逻辑，大大降低了构建新基准的门槛。

### 编码 Agent 深度对决：揭示“协同适应”效应

为了进一步验证 AAA 范式在复杂任务中的高保真度，研究团队进行了一项硬核的案例研究：在同一套标准化管道下，评测当前最顶尖的四个编码 Agent。

参赛选手包括：
*   Claude Opus 4.7 + Claude Code
*   GPT-5.4 + Codex CLI
*   Gemini 3.1 Pro + OpenCode
*   Qwen3.5（开源）+ mini-SWE-agent

它们在 DevEval、SWE-Bench Pro 和 Terminal-Bench 2.0 三个主流基准上展开了角逐。结果显示，**没有单一的 Agent 系统能在所有基准上占据绝对统治地位**。例如，GPT-5.4 在函数级补全（DevEval）上表现最佳，而 Claude Opus 4.7 则在仓库级问题解决（SWE-Bench Pro）上拔得头筹。

更具学术价值的是，得益于 AAA 框架的解耦特性，研究团队进行了一项**“脚手架交换实验”（Harness-Swapping Experiment）**。他们尝试让 GPT-5.4 搭配 Claude Code 运行，让 Claude Opus 4.7 搭配 Codex 运行。

实验揭示了一个强烈的**协同适应（Co-adaptation）**现象：在绝大多数情况下，基础模型只有在使用其“原生”搭配的脚手架时，才能发挥出最高的问题解决率。当互换脚手架后，性能平均下降了 5.3 个百分点。不过，这种绑定关系也带来了取舍，比如 GPT-5.4 在使用其原生 Codex 时，消耗的输入 Token 数量显著多于使用 Claude Code。这些深度的对比结论，在传统的、未标准化的评测体系中是极难被准确测量出来的。

### 总结与工程启示

《AgentBeats》这篇论文不仅是对当前 Agent 评测乱象的一次精准切脉，更是给出了一剂极具实操性的良方。

对于广大的 AI 开发者而言，这带来了两点重要启示：
第一，未来的 Agent 开发应当主动拥抱 A2A 和 MCP 等标准化协议。一旦完成适配，你的 Agent 就能以极低的成本接入全网的评测体系，真正做到“一次开发，到处评测”。
第二，设计基准测试的思路需要转变。不要再试图通过写死代码来控制 Agent 的每一步，而是应该尝试将基准测试“Agent 化”，通过下发清晰的指令和提供标准的环境接口，让 Agent 在接近真实的生产环境中展现其真正的能力。

统一标准的号角已经吹响，Agent 评测的“大一统”时代或许就在眼前。
