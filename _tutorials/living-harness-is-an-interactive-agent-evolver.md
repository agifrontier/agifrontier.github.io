---
layout: default
title: "Living-Harness：不改模型与工具，让Agent把失败变成可复用的持久修复"
description: "在多轮交互与工具调用的复杂任务中，大语言模型（LLM）Agent经常陷入一种令人沮丧的怪圈：在当前轮次或单次重试中，Agent明明通过反思纠正了错误；但只要开启新任务，面对极其相似的场景，它依然会重蹈覆辙。"
arxiv_id: "2607.26598"
published_at: "2026-09-04T13:15:08.122237+08:00"
topics:
  - "AI Agent"
tags:
  - "Evolution-SOP"
  - "LLM agents"
  - "Living-Harness"
  - "MultiWOZ-2.4"
  - "bounded harness updates"
  - "episodic memory"
related_tutorials:
  - "agent-harness-engineering-a-survey"
  - "from-prompts-to-contracts-harness-engineering-for-auditable-enterprise-llm-agents"
  - "llmtimesmapreduce-v3-enabling-interactive-in-depth-survey-generation-through-a-m"
  - "longhorizon-harness-advancing-long-horizon-agents-for-real-world-tasks"
---

<p class="paper-original-title" lang="en">Living-Harness Is an Interactive-Agent Evolver</p>

<img src="/images/2607.26598v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在多轮交互与工具调用的复杂任务中，大语言模型（LLM）Agent 经常陷入一种令人沮丧的怪圈：在当前轮次或单次重试中，Agent 明明通过反思纠正了错误；但只要开启新任务，面对极其相似的场景，它依然会重蹈覆辙。

> ArXiv URL：https://arxiv.org/abs/2607.26598v1

这种现象暴露出当前智能体架构的本质缺陷——**局部重试（Task-local retry）的纠错经验会随着任务生命周期的结束而被直接丢弃，无法沉淀为持久的程序性资产**。在现实工业级落地中，为了保证 Agent 的稳定运行，工程师通常会为其套上一层外周支架（Harness），包含固定的提示词模板、工具集合、环境上下文、标准工作流（SOP）以及评估接口。然而，传统的 Harness 一旦部署即处于完全静态固化的状态。它既不能自发吸收交互中的失败教训，也无法动态更新未来的决策路径。

由阿里巴巴、香港城市大学、香港科技大学、密歇根大学以及浙江大学联合提出的 **Living-Harness**，正是为了打破静态支架的局限而生。该研究提出了一种自演化的交互式 Agent 支架架构：在绝对冻结工具定义与基础上下文（Base Context）的前提下，将每一次交互轨迹及后验评估信号，沉淀为情景记忆（Episodic Memory）与状态图（State Graph）的双重演化，使 Agent 能够跨任务持续积累程序性修复经验。

在 $\tau^2$-Bench 与 MultiWOZ-2.4 两个极具代表性的多轮交互与工具调用基准上，Living-Harness 相比最强的交互式基线分别取得了 10.07 和 9.91 个百分点的 Pass@1 提升。更关键的是，演化所得的支架知识能够直接以“纯检索”的方式零成本迁移给其他未参与演化的大模型底座。

<img src="/images/2607.26598v1/figure1.webp" alt="静态支架与动态演化支架的区别" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 从“文字教训”到“程序性修复”

过去几年，学术界与工业界尝试过多种提升 Agent 稳定性的路径，主要分为两大派系。

第一类是**支架与工作流预先设计**。例如人工撰写更严格的系统提示词、固化状态机流转或者在离线阶段进行流程优化。这类方案构建的支架高度受限且完全静态，无法在部署后根据线上遇到的未定义异常进行动态修补。

第二类是**反思与经验记忆机制**，如 Reflexion 等。它们让模型在失败后写下一段自然语言总结或反思，以备下次尝试。然而，泛泛的自然语言反思往往无法直接转化为精确的系统行为。例如，模型在反思中写道“当用户情绪激动时应该转接人工客服”，但它并未明确定义触发转人工的精准前置状态、所必需调用的具体工具名称、参数格式以及转接后的状态转移。结果是，在下一次任务中，即便检索到了这段反思，模型依然会因为缺乏具体的执行协议而遗漏关键工具调用。

Living-Harness 的核心切入点正在于：**将单次失败转化为跨任务的“程序性修复”（Persistent Procedural Repair）**。这种修复必须满足四个苛刻条件：

1. **证据落地**：更新必须由完整的交互轨迹和客观的评估信号所驱动，而非模型不可靠的自我臆断；

2. **持久且具边界**：经验必须跨越任务生命周期留存，同时严格限制适用域，防止经验泛化过度导致负迁移；

3. **边界受控**：只能演化可演化的状态部分，外部工具接口和核心业务上下文必须保持冻结，防止 Agent 在演化过程中越权或破坏系统安全边界；

4. **双重知识表征**：必须同时承载解释“为何失败与如何恢复”的经验知识，以及规范“状态转移与确定性动作”的工作流知识。

### 核心机制：双重状态容器与演化 SOP

Living-Harness 构建了一个严密的“执行–评估–更新”（Rollout–Evaluate–Update）闭环。整个系统的演化状态表示为：




{% raw %}$$S_d^{(n)} = \left(\mathcal{R}_d^{(n)}, G_d^{(n)}\right)$${% endraw %}



其中 $\mathcal{R}_d^{(n)}$ 代表记录经验知识的情景记忆，$G_d^{(n)}$ 代表记录工作流知识的状态图，$n$ 为演化周期。

这两套容器分别承担不同的认知分工。情景记忆专注于记录**触发条件、失败模式与恢复行为**（Trigger, Failure Pattern, Recovery Action）。当遇到相似的执行困境时，情景记忆为模型提供细粒度的应对经验。状态图则是一个包含状态节点、修复边与转移规则的拓扑结构，专注于记录**状态机级别的转移逻辑**。一旦某个特定的任务状态被触发，状态图会强制给出合法的工具调用边与下一步流转节点，从而把容易被遗忘的动作固化为工作流图谱的一部分。

连接交互反馈与容器更新的核心枢纽，是研究团队提出的 **Evolution-SOP**（$\psi_d$）。Evolution-SOP 是一个固定在领域层面的仲裁协议，它独立于 Agent 自身的执行模型，在每轮交互结束后介入。

整个更新流程分为严谨的三步：

1. **后验证据提取（Posterior Extraction）**：在交互结束并获得评估信号 $y_n$ 后，演化模块结合任务描述 $x_n$、完整执行轨迹 $\tau_n$、冻结的基础配置 $C_d^{\mathrm{act}}$ 以及评估结果，抽取出结构化的演化证据 $e_n$。

2. **结构化拆解（Evidence Structuring）**：Evolution-SOP 将证据分别解析为针对情景记忆的候选条目 $u_n^{\mathcal{R}}$ 和针对状态图的候选修改边 $u_n^G$。

3. **一致性准入校验（Commit Gating）**：并非所有提取出的反思都能写入支架。候选修复必须经过严格的校验，包括其证据支撑强度、任务适用边界、以及是否违反冻结的工具与核心业务规则。只有通过校验的候选条目，才能被正式写入或强化到全局状态 $S^{(n+1)}$ 中；凡是证据不足或存在规则冲突的更新，均会被直接拦截。

在后续的任务执行中，系统根据当前任务生成上下文查询向量 $q_n$，从情景记忆和状态图中分别检索 Top-$k$ 个相关片段，融合成程序性上下文（Procedural Context）$\kappa_n$ 并动态注入提示词。这样一来，执行层的大模型在做决策时，不仅能看到任务输入，还能清晰看到过往在此类场景下沉淀出的确定性避坑指南与转移规则。

### 理论透视：程序状态 POMDP 视角

为了从数学上严格阐明为什么这种外部演化能持续降低 Agent 的决策失误率，本文引入了带有“程序状态”的部分可观测马尔可夫决策过程（POMDP）分析框架。

在经典 POMDP 中，环境真实状态 $s_t$ 不可直接观测，智能体必须依赖历史交互轨迹 $h_t$ 维持一个信念状态（Belief State）$b_t(s) = P(s_t = s \mid h_t)$。而在 Living-Harness 下，智能体的认知状态被扩展为联合状态 $\widetilde{s}_t^{(n)} = (s_t, z^{(n)})$，其中 $z^{(n)}$ 表示从演化支架中检索出的程序知识。

联合信念状态的更新形式遵循：




{% raw %}$$B_{t+1}^{(n)}(s', z) \propto \Omega(o_{t+1} \mid s') \sum_{s} T(s' \mid s, a_t, z) B_t^{(n)}(s, z)$${% endraw %}



论文给出的核心推论指出，带有支架程序上下文的联合信息空间所对应的贝叶斯预测误差下界，严格小于或等于仅依赖裸交互历史的信息空间：




{% raw %}$$\mathcal{B}(\mathcal{I}_{s,z}) \le \mathcal{B}(\mathcal{I}_s)$${% endraw %}



这一结论在理论层面上揭示了 Living-Harness 的优势根源：**智能体在复杂环境中的失误，往往不是因为单步推理能力不足，而是因为环境转移概率中缺少了关键的程序性约束**。通过将后验修复经验具象化为 $z$，支架实际上压缩了环境转移的不确定性，从而在根本上降低了模型做出错误动作选择的概率。

### 实验验证：显著超越传统反思与静态基准

研究团队在两个高难度的交互式多轮评测基准上进行了详尽的评估：一个是包含 Retail（零售）、Airline（航空）和 Telecom（电信）场景的真实工具调用基准 **$\tau^2$-Bench**；另一个是经典的面向任务型多轮对话基准 **MultiWOZ-2.4**。实验选用 GPT-5.2（中等推理强度）作为底层模型，并在演化更新时统一该模型底座。

实验结果展现出 Living-Harness 极具说服力的性能表现：

在 $\tau^2$-Bench 上，Living-Harness 的平均 Pass@1 达到了 **83.09%**。相比于最强的交互基线 Reflexion（73.02%），取得了 **10.07 个百分点的绝对提升**；这一表现甚至小幅超越了以 Gemini 3 Pro 旗舰模型为底座的单轮静态基准平均表现（82.92%）。

在 MultiWOZ-2.4 上，Living-Harness 的平均 Pass@1 达到 **65.50%**，大幅超过了最强交互基线 ReasoningBank（55.59%），提升幅度达 **9.91 个百分点**，同时领先 Reflexion 多达 12.40 个百分点。尤为突出的是，在横跨两到三个业务领域的长链路复杂任务组中，Living-Harness 的优势更加显著，证明了状态图与结构化记忆在处理多跨度复杂业务时的优异迁移能力。

<img src="/images/2607.26598v1/Figure3.webp" alt="消融实验对比图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么必须同时需要 SOP、记忆与图？

为了探究各项组件在闭环中的真实贡献，研究团队设计了细致的消融实验（如上图所示）：

- **剥离 Evolution-SOP（w/o Evolution-SOP）**：保留记忆与状态图容器，但去除领域 SOP 引导的后验抽取和校验机制，退化为无管制的通用信息抽取。结果显示，平均性能由 83.09% 骤降至 73.38%，降幅近 10 个百分点。这证明仅仅拥有记忆和图容器是远远不够的，缺乏严密的准入校验和结构化解析，容器很快会被低信噪比的反思和错误经验污染。

- **剥离情景记忆（w/o Memory）**：禁用情景记忆的检索与写入，平均表现下降至 77.34%。

- **剥离状态图（w/o State Graph）**：禁用状态图的维护，平均表现下降至 79.50%。

消融数据有力证明，情景记忆所代表的“因果复盘认知”与状态图所代表的“拓扑转移规约”是高度互补的，二者协同构成了完备的程序性资产。

<img src="/images/2607.26598v1/Figure4.webp" alt="执行修复案例分析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

上图所呈现的真实案例直观展现了这种机制的精妙之处。在第 0 轮周期中，传统 Reflexion 方案虽然在反思中精准写出“应当将用户转接至人工代理”，但由于其反思无法指导具体动作，随后的重试中依然反复漏掉实际的 `transfer_to_human_agents()` 工具调用。

而在 Living-Harness 下，Evolution-SOP 将这一失败捕获后，一方面在情景记忆中建立了“检测到终止挂起条件 $\rightarrow$ 执行人工转接动作”的条目，另一方面在状态图中显式拉起了一条连接挂起状态与转接工具调用的修复边。当进入第 1 周期面对相似困境时，Agent 从支架中直接检索出这条确定的状态转移边，毫不犹豫地完成工具调用，单次便成功完成了任务。

### 极具应用价值的发现：演化支架可跨模型无缝迁移

除了在同构模型下的演化累加，论文中还包含一个极具工业落地价值的深度发现：**由 GPT-5.2 演化得到的 Living-Harness 状态，在完全冻结后，可以直接以只读检索的方式赋能给其他从未参与演化的大模型。**

当把演化好的记忆与状态图迁移给 Gemini 3 Pro、GLM-5、Qwen3-max 和 Kimi-k2 等不同底座时，所有模型在目标领域上的得分均呈现出普适性上涨。尤其是在基础模型极难破局的极端困难场景（如 MultiWOZ 中的 Taxi 域），GLM-5、Qwen3-max 和 Kimi-k2 的裸机或传统方案 Pass@1 原本均为 0.00%，但在加载了 GPT-5.2 沉淀下的支架后，准确率直接跳升至 43.08%、45.13% 和 45.13%。即使是性能本身极其强悍的 Gemini 3 Pro，在接入外部支架后同样获得了稳步提升。

这一结果充分表明，Living-Harness 所积累的程序性知识并非与特定模型的参数权重深度绑定，而是具备高度抽象、独立且通用的外部业务逻辑。它证明了为 Agent 构建“外挂式持续演化系统”的可行性：团队可以用高规格模型在后台演化出成熟的业务支架，随后直接装配给轻量化或低成本的开源模型运行推理，从而在兼顾安全与成本的同时最大化系统可靠性。

### 结语

长期以来，提升大模型 Agent 解决复杂任务能力的思路往往聚焦于两端：要么向上追求模型底座单步推理能力的突破，要么向下依赖人工耗时耗力维护复杂的状态机规则。

Living-Harness 提供了一条极具启发性的中间道路：**保持模型参数与底层工具接口稳固不变，而在模型外部建立一个遵循准入规则、能够将历史教训沉淀为图谱与记忆的活性支架**。让 Agent 在真实的交互风浪中，不仅学会如何回答眼前的一句话，更学会把每一次跌倒的教训铸造成未来通行的路标。
