---
layout: default
title: "Agentic Data Environments：哥大打造AI安全底座，AIM准确率提升49.8%"
description: "随着大型语言模型的普及，自动化正迎来从“辅助生成”向“自主执行”的转折点。现代AIAgent不仅能够编写代码、浏览网页，更能自主调用API、操作终端甚至查询生产数据库。然而，当前的AI自动化往往止步于“只读”阶段，例如通过检索增强生成（RAG）回答问题或汇总财务报表。"
arxiv_id: "2607.07397"
published_at: "2026-08-18T15:54:17.700008+08:00"
topics:
  - "AI Agent"
  - "AI安全"
tags:
  - "Agentic Data Environments"
  - "active data substrates"
  - "agentic automation"
  - "autonomous agents"
  - "bounded failure modes"
  - "execution substrate"
related_tutorials:
  - "auditing-agent-harness-safety"
  - "the-alignment-waltz-jointly-training-agents-to-collaborate-for-safety"
  - "what-makes-a-harness-a-harness-necessary-and-sufficient-conditions-for-an-agent-harness"
  - "toucan-synthesizing-15m-tool-agentic-data-from-real-world-mcp-environments"
---

<p class="paper-original-title" lang="en">Agentic Data Environments</p>

<img src="/images/2607.07397v1/A__title.webp" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

随着大型语言模型的普及，自动化正迎来从“辅助生成”向“自主执行”的转折点。现代 AI Agent 不仅能够编写代码、浏览网页，更能自主调用 API、操作终端甚至查询生产数据库。然而，当前的 AI 自动化往往止步于“只读”阶段，例如通过检索增强生成（RAG）回答问题或汇总财务报表。这种设计的优势在于能够限制潜在的灾难性破坏；但当 Agent 被赋予修改环境的权限时，例如从整理账单直接跨越到提交法律认可的退税申请，每一次错误就不再仅仅是一次“幻觉”，而是意味着罚款、法律纠纷或是系统崩溃。

> ArXiv URL：https://arxiv.org/abs/2607.07397v1

哥伦比亚大学（Columbia University）的最新研究提出了 **Agentic Data Environments**（代理化数据环境）这一全新理念。该研究明确指出，要让自主 Agent 真正创造规模化价值，不能仅靠扩展大模型的参数或上下文窗口，必须从系统层面同时做到两件事：极大提升 Agent 获取信息的质量，并严格限制其执行失败带来的破坏性后果。

简而言之，数据系统需要从被动的状态存储器，演进为专为 AI Agent 打造的安全、活跃的执行基底。这项工作为未来数据系统的设计重新定义了核心挑战。

<img src="/images/2607.07397v1/overview.webp" alt="Agentic Data Environments 概览" style="width:85%; max-width:600px; margin:auto; display:block;">

### 从被动数据库到代理化执行基底

传统的数据库系统是现代计算的支柱，主要服务于人类预先定义好的应用程序。然而，Agent 在完成真实世界任务时，面对的远不仅是结构化的数据库，还包括文件系统、API、配置系统、命令行工具以及各种外部微服务。

当一个 Agent 发起一个 HTTP 请求时，它可能会触发服务器端的业务逻辑，继而启动后台任务、调用外部 API、修改文件并更新数据库。这个执行链条牵扯到海量的系统状态。这也就意味着，面向 Agent 的“数据管理”必须超越传统的 DBMS（数据库管理系统），延伸到更为广阔的数据环境中。在这个异构环境中，如何让 Agent 高效且安全地探索、规划和试错，是当前基础设施面临的首要问题。

为了打破这种瓶颈，哥伦比亚大学的研究者从两方面解构了 Agentic Data Environments 的核心支柱：一方面是通过信息重塑增强 Agent 的认知与执行能力，另一方面是通过沙盒分支和数据流控制保障其对真实环境的介入安全。

### 增强认知：打造 Agent 原生的信息生命周期

由于 Agent 任务的复杂性，如今的模型失误往往源于缺乏有效的信息输入，而非缺乏逻辑推理能力。即使是最强大的基础模型，如果在海量无序的数据湖中找不到结构化的线索，也会频繁遭遇失败。为了解决这一问题，研究提出了三项互补的信息管理机制，让数据更加贴合 Agent 的消费习惯。

<img src="/images/2607.07397v1/im-system.webp" alt="信息管理系统架构" style="width:85%; max-width:600px; margin:auto; display:block;">

#### Agentic Information Management (AIM)：放弃单一的 RAG

目前业界处理非结构化数据的主流做法，是将其全部转化为向量嵌入（Embeddings）并送入 RAG 架构，或者将其硬塞进长上下文窗口中。这种一刀切的表示方法往往忽略了 Agent 具体任务所需的内在逻辑结构。比如在一个包含数月多轮对话的数据集中，将所有对话切块检索，会彻底破坏时间顺序、说话人身份以及多会话之间的交叉关联。

因此，研究提出了 Agentic Information Management (AIM)。AIM 不再把数据当作一堆静态文本，而是作为一个多智能体系统，主动分析原始数据源，并根据目标任务预测最佳的存储架构（Schema）。它会自动生成提取管道，将原始文本转换为关系型数据库结构。

<img src="/images/2607.07397v1/aim.webp" alt="AIM 工作流示意" style="width:85%; max-width:450px; margin:auto; display:block;">

以 LoCoMo 多会话数据集为例，AIM 会自主建构“用户”、“会话”、“信息”、“兴趣事件”等互相关联的数据表。当目标 Agent 随后需要回答“某个用户平时如何放松”时，它只需通过调用 SQL 技能过滤特定表项，而无需在成百上千条无序对话中大海捞针。

实验结果揭示了这种结构化转型的巨大红利。与专为 Agent 设计的流行记忆系统 Mem0 相比，AIM 的准确率提升了 $49.8\%$；与基于顶尖 RAG 架构的 Octen 相比，准确率提升了 $15.82\%$。在计算效率上，AIM 比最先进的 Agent 记忆系统 GAM 快了 $4.18\times$。这证明了将数据建模过程前置，由 AIM 持续优化 Schema，远比强行让大模型在推理时消耗 Token 梳理线索要高效得多。

#### Agentic Information Retrieval (AIR)：应对数据湖中的迷失

在数据结构化之前，Agent 必须先在广袤的异构数据湖（数以百万计的文档和数据集集合）中找到正确的信源。研究表明，针对数据湖检索任务，目前七款前沿模型的端到端准确率均低于 $23\%$，且它们主要死于找不到对应的数据源，而非推理出错。

<img src="/images/2607.07397v1/lakeqa.webp" alt="数据湖检索挑战" style="width:85%; max-width:450px; margin:auto; display:block;">

这就催生了 Agentic Information Retrieval (AIR) 的需求。AIR 倡导在 Agent 和数据湖之间建立一个专门的“语义层”，对存在哪些数据、数据的含义以及不同源之间的潜在关联进行摘要。此外，针对数据分布过于离散的情况，AIR 要求系统不仅要解决单点检索，还要联合推理不同数据集的组合方式（例如跨表连接、实体消解），以最优的计算成本规划推理路径。

#### Agentic Data Elicitation (ADE)：发掘隐性结构

除显性数据外，真实系统中还存在大量隐性结构，例如未被文档记录的表关系、隐藏的性能瓶颈模式或控制参数与系统指标的关联。Agentic Data Elicitation (ADE) 的核心目标是通过主动实验或被动收集，将这些环境中的隐性规则抽取为显式制品（如统计摘要、复用的查询提示或调优知识库），并固化到环境中供后续 Agent 共享使用。这使得环境能够随着 Agent 的持续运行而自动“变聪明”。

### 构筑安全底座：让探索不再带来破坏

赋予 Agent 获取和重组数据的能力仅仅是第一步。真正限制 Agent 进入读写应用场景的障碍是“状态破坏风险”。复杂的长视距任务通常不能靠一步到底的线性推理完成，它们依赖于蒙特卡洛树搜索（MCTS）、反思与回溯等试错手段。如果在真实环境中直接修改状态，后果不堪设想。为此，Agentic Data Environments 引入了两项基础机制来兜底安全：状态分支与数据流控制。

#### 跨组件的秒级状态分支 (Branching)

为了让 Agent 安全地规划与试错，底层系统必须支持将当前的中间状态进行分支（Branch），隔离所有的探索性操作。这不仅仅是做一个数据库快照那么简单。

传统的数据库虽然提供了 MVCC（多版本并发控制）和各类基于事务的回滚机制，但无法应对高频次、大规模的推演探索。例如在使用 Neon 等支持分支的现代数据库引擎运行千步 MCTS 时，受限于并发限制，往往只能执行 $3\%$ 的步骤。此外，Agent 操作的系统并非孤立的数据库，还涉及应用程序内存、文件系统以及依赖关系极其复杂的进程上下文。

<img src="/images/2607.07397v1/os.webp" alt="跨环境的分支管理" style="width:85%; max-width:600px; margin:auto; display:block;">

当一个 Python 脚本连接到数据库时，如果在沙盒中仅对数据库进行分支，Python 进程中缓存的元数据将会过期报错；如果仅对 Python 进程做分支，脚本内的推演更新又会直接泄露到生产数据库中。因此，分支必须能够捕获跨组件的一致状态（Transient Dependencies）。

哥伦比亚大学团队为此开发了 Checkpoint-lite (Chkpt) 系统，通过 Copy-on-Write 技术仅对变更部分进行共享，避免了沉重的容器化封包开销。在测试中，Chkpt 捕获文件系统状态仅耗时 $66\,\mathrm{ms}$，甚至在对 $1\,\mathrm{GB}$ 内存和文件系统联动捕获时，Chkpt 也仅需 $1.46\,\mathrm{s}$，大幅优于基于 Podman 和 CRIU 的 $8.84\,\mathrm{s}$。这种级别的隔离让 Agent 可以在“平行时空”中任意尝试危险的更新指令。

#### 数据流控制 (Data Flow Control, DFC)

解决了系统状态隔离的问题，并不等于解决了数据本身的合法性问题。即便在合法的探索空间内，Agent 也可能组合出违反监管政策的数据流。例如，在自动税务核算中，Agent 可能会把敏感的员工个人差旅开销与其他数据汇聚起来并调用外部 API。

目前很多系统依赖 LLM 探查用户的 Query 或 Prompt 是否违规。然而这种基于概率的审核在复杂策略面前不仅性能低下，且缺乏刚性保证。研究发现，让 GPT 或 Claude 模型针对包含 $100$ 条返回结果的查询执行合规性检查，不仅每次需耗时 $0.8$ 到 $2.2$ 秒，F1 得分更是仅有可怜的 $0.4$。

Agentic Data Environments 提出将数据流控制下沉至 DBMS 的查询引擎内部。通过计算关系溯源（Relational Provenance），确保政策语义满足优化器不变量（Optimizer-invariant），系统仅根据实际贡献输入数据的源头来进行决策判断，而不在乎物理执行计划被如何重写。该机制在五大主流数据库引擎（如 DuckDB、PostgreSQL 等）中测试表明，执行这种严格的 DFC 策略开销几乎为 0，并且能提供百分之百的确定性保障。而对于超出数据库范畴的数据转移，DFC 还将沿组件调用的生命周期网络持续传播这些溯源标签。

### 总结与展望

Agentic Data Environments 提供了一种系统级的设计范式，用来弥合大模型认知与物理世界工程执行之间的鸿沟。

<img src="/images/2607.07397v1/virtue.webp" alt="自动化能力的良性循环" style="width:85%; max-width:600px; margin:auto; display:block;">

正如上图所示，这个生态环境促成了一个良性循环：通过 AIM、AIR 和 ADE 这类代理化信息管理手段，系统持续为 Agent 输送结构化的高质量信号，极大地放大了自动化的上限收益；另一方面，依托于亚秒级的全局状态分支机制与强制数据流控制，系统将试错和违规动作严格限制在隔离区中，兜住了下限的安全性风险。

更深远的意义在于，这种能够廉价创建分支、记录衍生并管理奖励信号的数据底座，正是下一代基于强化学习（RL）进行 Agent 后训练所急需的基础设施。将数据系统从被动容器向 Agent 执行底座的重塑，很可能是决定我们在未来几年内能否真正见证“自主数字员工”全面落地的关键。
