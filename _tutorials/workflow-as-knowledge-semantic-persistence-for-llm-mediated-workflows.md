---
layout: default
title: "不是分散日志，而是持久化知识：LLM工作流语义框架"
description: "大语言模型（LLM）的应用正在发生一场不可逆转的结构性转变：从简单的单轮提示词（Prompting），全面转向由工具调用、检索、分支、检查点和人工审批构成的显式工作流（ExplicitWorkflows）。"
arxiv_id: "2607.08740"
published_at: "2026-08-18T15:31:33.905351+08:00"
topics:
  - "基础模型"
tags:
  - "LLM-mediated workflows"
  - "derive vs infer"
  - "executor-controlled capability policy"
  - "inspectable/resumable workflows"
  - "knowledge substrate"
  - "live-image thinking"
related_tutorials:
  - "llama-guard-llm-based-input-output-safeguard-for-human-ai-conversations"
  - "rethinking-supervised-fine-tuning-emphasizing-key-answer-tokens-for-improved-llm"
  - "alpacafarm-a-simulation-framework-for-methods-that-learn-from-human-feedback"
  - "an-information-theoretic-framework-for-robust-large-language-model-editing"
---

<p class="paper-original-title" lang="en">Workflow as Knowledge: Semantic Persistence for LLM-Mediated Workflows</p>

<img src="/images/2607.08740v1/A__title.webp" alt="" style="width:85%; max-width:600px; margin:auto; display:block;">

大语言模型（LLM）的应用正在发生一场不可逆转的结构性转变：从简单的单轮提示词（Prompting），全面转向由工具调用、检索、分支、检查点和人工审批构成的显式工作流（Explicit Workflows）。市面上的各种 Agent 框架通过图结构、节点状态和内存机制，在一定程度上解决了“如何让模型可控运行”的执行问题。

> ArXiv URL：https://arxiv.org/abs/2607.08740v1

然而，将控制流显式化，并没有解决一个更深层的表征问题——在现有的体系中，工作流的定义通常作为源代码、声明式配置或数据库记录存在，运行时的中间模型输出则被降维成了海量的运行日志、链路追踪（Traces）或对话历史。这种松散的关联，使得系统在遇到中断恢复、逻辑审计或决策溯源时变得极为困难。

《Workflow as Knowledge: Semantic Persistence for LLM-Mediated Workflows》一文指出，现有的执行机制掩盖了工作流真正的本体论意义。文章提出了一种全新的概念模型：**不要仅仅把工作流看作产生知识的“管道”，工作流本身、运行实例、推理记录和上下文快照，都应该作为持久化的“知识对象”（Knowledge Objects）存在于一个共享的知识基底中。**

这项工作并不是在推销某一个具体的开源代码库，而是为未来的 LLM 系统设计提供了一套类似于 Lisp 环境的语义架构。以下我们将深入解析这种“工作流即知识”的设计理念究竟有什么不同，以及它为什么可能成为下一代 Agent 框架的标准规范。

### 解构现有系统：为什么记录日志还不够？

在当前的工程实践中，开发者通常围绕模型构建一套被称为“Agent Harness”（智能体挂载框架）的控制与运行时机制。这套框架通过一系列 API 调用、大语言模型请求和胶水代码串联起业务逻辑。

当开发者试图回溯一次模型失控或决策偏差时，往往只能去翻阅离散的执行日志。这些日志记录了“时间点 T 发生了调用 A 并返回了结果 B”，但失去了上下文语义：模型当时能看到什么背景信息？这步调用是基于确定性的代码规则还是模型的自由推理？审批此操作的人类用户又是基于什么论据做出的决定？

为了解决这种断层，研究者首先在概念上将整个系统拆解为三个明确的层次：

1. **底层（运行时服务层）**：负责提供模型适配器、外部工具、外部进程交互以及底层的持久化和索引设施。这部分是纯粹的物理执行能力。

2. **中层（控制层）**：包含领域特定语言（DSL）机器及其执行器（Executor）。它的职责是解释上层声明的对象，组装上下文，对模型和工具调用进行中介，验证结果并执行转移动作。

3. **高层（语义层）**：这是整个模型的核心，包含工作流定义（Workflow Definitions）、工作流实例（Workflow Instances）及其链接的推理记录、审批记录和审议记录。

<img src="/images/2607.08740v1/mermaid-001.55d3bc13c172.webp" alt="Figure 1. Semantic workflow objects are interpreted by the DSL-machine control layer, which coordinates runtime services and writes back workflow instances, mediated effects, and records of inference, approval, and panel activity." style="width:85%; max-width:600px; margin:auto; display:block;">

在这种架构中，控制层读取语义对象，并在运行时将其执行结果“写回”为持久化的语义对象。这种双向关系确保了所有有意义的动作都不会以“过眼云烟”的形式消散，而是变成了系统中可供查询和复用的结构化数据。

### 从执行持久化到语义持久化

当前很多先进的图架构 Agent 系统（如 LangGraph）已经引入了检查点（Checkpointing）机制，允许系统在特定节点保存状态，以便实现容错或时间旅行（Time Travel）。但这仍然属于**执行持久化（Execution Persistence）**——其核心目的是为了让程序能够继续跑下去。

相比之下，本文主张的是**语义持久化（Semantic Persistence）**。这受到早期 Lisp 机器“实时镜像”（Live-image）思想的启发，在那种环境中，程序结构和运行数据在内存中统一表示，随时可以被反省和修改。在语义持久化下：

- **工作流定义**：不再是一段只读的代码，而是一个包含了输入声明、资源、状态模式、守卫规则和步骤的知识对象。

- **工作流实例**：不仅是一个内存里的线程，而是一个持续存活的实时对象，它的状态本身就是一个语义检查点。

- **推理记录**：模型的每一次关键判断，都会附带当时的上下文快照一起打包保存，而不仅仅是把生成的文本拼接到对话记录里。

<img src="/images/2607.08740v1/mermaid-002.b44c1e4dfbff.webp" alt="Figure 2: Execution persistence retains runnable state, checkpoints, logs, traces, and outputs; semantic persistence treats workflow definitions, workflow instances, inference records, and context snapshots as first-class knowledge objects." style="width:85%; max-width:600px; margin:auto; display:block;">

这种机制带来的最直接好处是极强的可审计性。当一个决策产生争议时，审查者提取出的不仅仅是一个最终文档，而是整个工作流实例，其中明确包含了当时模型能够获取哪些知识来源、受限于什么安全策略，以及它推导出该结果的具体链路。

### 核心分水岭：derive 与 infer 的语义隔离

如果说把状态持久化是工程上的优化，那么对计算本质的严格区分则是这篇论文在语义层做出的最重要贡献。

在很多混杂的 LLM 脚本中，传统的函数调用（比如正则提取一个数字）和模型调用（比如让模型判断一句话的情感）被同等对待，都视为获取某个变量的操作。然而，它们在认识论上的可靠性是完全不同的。为了解决这个问题，论文在抽象 DSL 中引入了两个泾渭分明的原语：`derive` 和 `infer`。

- **`derive`（派生）**：代表针对已有状态的**确定性计算**。无论是格式化文本、运行一段校验脚本，还是进行路由规则匹配，只要过程是完全封闭且确定的，统统归为 derive。执行器会计算其结果并绑定到状态中，它的依赖关系是确定且无歧义的。

- **`infer`（推断）**：代表**LLM介导的判断**。这包括分类、内容合成、语义审查或代码生成。它是不确定的、带有黑盒性质的。

在工作流中执行 `infer` 时，执行器必须经过极为严格的流程：首先根据上下文范围组装出一份**上下文快照（Context Snapshot）**，接着附带上当前的策略要求发起模型调用，得到候选结果后，还需要执行器验证才能将值绑定到状态空间中。

<img src="/images/2607.08740v1/mermaid-003.fbfb487bc625.webp" alt="Figure 3. derive computes over available state; infer requests mediated LLM judgment whose recorded value may influence an executor-applied declared branch." style="width:85%; max-width:450px; margin:auto; display:block;">

这种区分极为关键，它剥夺了 LLM 直接操纵系统底层逻辑的权力。在很多试图让 LLM 扮演“全能上帝”直接生成代码并执行的方案中，风险往往不可控。而在这个架构下，LLM 只是一个被严格限制在 `infer` 槽位中的“中介组件”，它没有权力去改变执行规则，它的输出结果充其量只能影响执行器下一步选择哪条预设分支。

### 重新定义执行器（Executor）的角色

明确了对象的分类和计算的边界后，执行器的角色就从单纯的“代码解释器”，变成了类似于法律程序中的“法官”或“登记员”。

执行器不仅要负责实例化工作流、检查安全策略（Policy checks）、调动运行时资源，更重要的是它控制着知识基底的写入权限。每一次状态转换完成后，执行器都会向底层知识基底写入一条关联的持久化记录。这使得系统拥有了一套内在的血统追踪机制。

<img src="/images/2607.08740v1/mermaid-004.d3d720fe32c3.webp" alt="Figure 4. The executor as controller of workflow instantiation, policy checks, runtime-mediated resources, transitions, and knowledge-substrate persistence." style="width:80%; max-width:300px; margin:auto; display:block;">

### 人类介入机制的深化：审批（Approval）与审议（Panel）

随着 Agent 应用进入高价值领域，“人类在环（Human-in-the-loop, HITL）”成为了必需品。但当前的系统通常将所有的交互一刀切地视为“请求人类输入（Human Prompt）”或一个简单的确认按钮，这种粗糙的处理抹杀了人类参与决策的结构化价值。

基于对数十个实际工作流制品的分析，论文提炼出了两种截然不同的人类介入原语：

1. **`approval`（审批）**：这代表一种单纯的权力放行操作。比如系统在发送真实邮件、执行删库操作或结束某个循环之前，执行器会暂停，将上下文快照提交给具有相应权限的用户。用户的选择通常是二元的（同意、拒绝、或推迟）。记录下来的 `approval` 对象是权限执行的审计证据。

2. **`panel`（审议）**：这代表一种结构化的、复杂的评审过程。比如要求专家委员会对模型生成的两份设计方案进行评估。审议不仅要求给出一个结论，还会包含选项比较、论证过程、辩论记录等。这种过程产生的 `panel` 记录，其内部包含了比单纯的 approval 丰富得多的语义网络。审议本身可能不会立即触发状态转移，而是将其决定固化为数据，供执行器在随后的预设分支中引用。

通过在架构层面原生区分这两者，开发者可以清晰地界定在什么时候系统需要的是“授权”，什么时候系统需要的是“人类智慧的注入”。

### 从工具控制走向知识沉淀的深远影响

回顾过去几年 AI 工程的演进，我们经历了从关注单次补全质量，到使用 RAG 扩大信息范围，再到引入反思、重试和规划来增强推理稳定性的过程。现在，开发者们正将大量的业务逻辑编码成图节点和工具描述。

这篇论文的真正价值在于，它跳出了“如何让任务执行得更快更准”的战术思维，站到了系统工程和知识管理的战略高度。它告诉我们，如果我们将 LLM 生成的中间判断视若敝屣，用完即弃，那么随着系统的运行，我们只是消耗了算力，却并没有积攒下可信的系统“经验”。

当我们将工作流视为知识（Workflow as Knowledge）时，一次长周期的科研文献审查任务，或者一次复杂的跨部门合同起草，其运行完毕后留下的不再是一个干瘪的“成功”信号和一大串无法对齐的 Token 流，而是一个包含定义、实例、严格区分计算性质的步骤记录以及附带快照的决策树。

这对于需要极高可解释性的医疗、法律、金融等合规性行业而言，可能是打通大模型落地最后一公里的关键概念。未来的 Agent 框架不仅需要能跑得通循环和分支，更需要具备将其自身执行逻辑映射到持久化语义空间的能力，从而使那些黑盒的模型推理真正融入人类文明的知识网络中。
