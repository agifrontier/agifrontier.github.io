---
layout: default
title: "ExpG：以经验沉淀定义工具边界，阿里等让小模型调用反超大模型"
description: "为此，研究团队提出了名为 ExpG （Experience-Driven adaptive Guidance）的自适应引导机制。该方法不再把工具调用视为孤立、静态黑盒事件，而是将其转化为可归因、可沉淀、可复用的结构化经验池。通过动态构建每个工具的能力边界与最佳实践，ExpG 实现了全链路的稳健调用。"
arxiv_id: "2608.03403"
paper_published: "2026-08-04"
published_at: "2026-09-21T13:15:08.283101+08:00"
topics:
  - "AI Agent"
tags:
  - "ExpG"
  - "adaptive guidance"
  - "equivalence-class-based selection"
  - "execution trajectories"
  - "experience acquisition"
  - "experience distillation"
related_tutorials:
  - "evoroute-experience-driven-self-routing-llm-agent-systems"
  - "verltool-towards-holistic-agentic-reinforcement-learning-with-tool-use"
  - "skillsentry-reliable-skill-execution-for-llm-agents-via-runtime-assurance"
  - "latent-on-policy-self-distillation"
---

<p class="paper-original-title" lang="en">Towards Robust Tool Use in Agents via Experience-Driven Adaptive Guidance</p>

<img src="/images/2608.03403v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在智能体（Agent）系统的落地过程中，工程团队普遍面临一个尴尬的现实：尽管大语言模型在学术跑分上的推理能力突飞猛进，但一旦将它们置于真实业务链路中去调用外部工具（Tools / APIs），整个系统就会展现出惊人的脆弱性。同一个 API，在测试环境中表现完美，在生产环境中却会因上下文细微漂移而频频崩塌；更糟糕的是，很多外部工具在报错时往往只返回诸如“Error 500”或者包含逻辑错误的“假成功”信号，导致模型在缺乏有效反馈的前提下陷入无限盲目重试。

> ArXiv URL：https://arxiv.org/abs/2608.03403v1

来自阿里巴巴、山东省数字服务计算技术与系统重点实验室以及香港理工大学的研究团队，在最新论文中直击这一核心痛点。他们认为，**当前 Agent 系统的瓶颈正在从纯粹的“模型能力”转向“执行过程的鲁棒性”**。

为此，研究团队提出了名为 **ExpG**（Experience-Driven adaptive Guidance）的自适应引导机制。该方法不再把工具调用视为孤立、静态黑盒事件，而是将其转化为可归因、可沉淀、可复用的结构化经验池。通过动态构建每个工具的能力边界与最佳实践，ExpG 实现了全链路的稳健调用。最引人注目的实验结果在于：**装备了 ExpG 的 8B 级别小模型，在 MetaTool、API-Bank 和 BFCL-V3 等多个主流基准上，表现不仅大幅超越原模型（综合 Avg@3 提升 7.41 个百分点），甚至反超了参数规模大得多的未武装模型**。

<img src="/images/2608.03403v1/overview.webp" alt="真实世界中的工具调用挑战与 ExpG 的整体收益" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么生产环境下的工具调用如此脆弱？

以往大多数提升 Agent 工具使用能力的工作，无论是微调（SFT / RL）还是提示工程（ReAct、Few-shot），都隐式依赖两项脆弱的理想化假设：

第一，**行为可预测性假设**。研究者默认只要 Agent 学会了某项工具的调用规范，该工具在任何环境上下文下都能稳定返回预期结果。但在真实生产环境中，外部环境具备动态性和漂移性。以最简单的搜索或计算工具为例，当输入上下文存在细微语义偏置、或者上下游工具链之间存在隐性依赖时，相同的参数可能直接触发异常。

第二，**反馈信息完整性假设**。以往方案假定工具在执行失败时，会提供详实、结构化的报错堆栈，Agent 能够借此进行“自我反思”（Self-Reflection）并纠错。但在工业级 API 中，工具返回的信息往往伴随极高噪声，甚至是粗粒度且缺乏鉴别度的。比如，一个计算器发生内部逻辑截断，却依然返回 `status: 200` 并附带一个错误数值；或者十种不同的调用参数错误，最终都统一包装成一句无意义的“Execution failed”。在这类模糊信号下，自反思不仅难以奏效，还会诱导模型在错误的推理路径上越走越远。

如果不能在 Agent 执行前明确工具的**能力边界（Capability Boundaries）**与**最佳实践（Best Practices）**，智能体在面对未知或非稳态场景时，本质上就只是在进行盲目的“试错游戏”。

### ExpG 架构：将碎片轨迹炼化为工具级指引

为了跳出“单次调用即遗忘”的被动循环，ExpG 建立了一套生命周期涵盖**经验获取（Acquisition）**、**经验提炼（Distillation）**与**经验复用（Reuse）**的自主演化机制。

<img src="/images/2608.03403v1/fmpic.webp" alt="ExpG 架构总览：经验获取、提炼与复用的三阶段闭环" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### 1. 经验获取：多维度归因构建结构化样本

在 Agent 执行历史任务的轨迹中，包含着大量成功与失败的隐性线索。ExpG 首先通过评估器（$\mathrm{LLM}_{\mathrm{evaluator}}$）对轨迹进行细粒度回溯，把每一次调用抽象为一个三元组 $E = \langle h, m, c \rangle$。

其中，$h$ 是经验的唯一哈希标识；$m$ 汇聚了元数据，包括工具名称、Schema 规范、智能体上下文、入参、响应体以及耗时和 Token 开销；核心在于经验内容 $c$，评估器并不只输出一句笼统的评价，而是输出一个 $d$ 维二值向量 $scores = [s_1, \ldots, s_d] \in \{0, 1\}^d$，分别从参数合法性、上下文匹配度、工具前后依赖等多个维度对调用质量进行切片打分，并辅以简短的归因解释。

这种设计让每一次调用都具备了统一表征，为后续的分类和筛选奠定了数学基础。

#### 2. 经验提炼：等价类划分与双维度总结

拥有大量调用记录后，如果简单把所有案例塞进 Prompt，不仅会导致上下文窗口膨胀，还会将随机噪声和长尾错误带入新的推理中。ExpG 在此引入了两个精细设计的步骤：

*   **基于等价类的代表性采样（Equivalence-Class Selection）**：根据多维打分向量 $\kappa(E) = scores(E)$，ExpG 将同一种工具的历史调用划分为不同的等价类 $\{\mathcal{C}_k\}_{k=1}^K$。每一个等价类代表该工具在特定场景下的典型调用模式（如“参数完全合法且高效”、“格式错误但逻辑正确”等）。研究团队提出了一套兼顾**原始数据分布对齐**与**多样性覆盖**的配额分配算法，确保过滤后的经验池既能反映工具在实际运行中的真实问题频次，又不会遗漏小概率但致命的异常模式。

*   **定性规范与定量指标的融合指引**：在筛选出的代表性经验基础上，总结器（$\mathrm{LLM}_{\mathrm{summarizer}}$）通过思维链（Chain-of-Thought）将碎片化的调用经验升级为工具级的指引。这一指引包含质性分析（核心功能定位、成功调用范式、常见陷阱、防御性最佳实践）与量化统计（平均成功率、评分分布、时间与 Token 成本画像）。

#### 3. 经验复用：动态上下文与稳态约束的自适应分流

并非所有提炼出的指引都采用相同的注入方式。ExpG 对经验池的一致性进行持续监控：




{% raw %}$$ \textsc{Type}(guidance_{\tau}) = \begin{cases} \text{stable}, & \text{if } f(\mathcal{E}_{\tau}) \\ \text{dynamic}, & \text{otherwise} \end{cases} $${% endraw %}



当某个工具的调用记录积累充分（$\lvert \mathcal{E}_{\tau} \rvert \ge Q/2$），且主流模式占比高度收敛（$\lvert \mathcal{C}^* \rvert \ge \alpha \lvert \mathcal{E}_{\tau} \rvert$）时，该指引被标记为 **stable（稳态）**。稳态指引代表该工具的最佳实践已经确立，ExpG 会将其直接内嵌到工具的定义结构（Schema Constraints）中，作为不可违背的硬性契约；反之，若工具处于探索期或调用模式离散度高，则标记为 **dynamic（动态）**，作为临时上下文 Prompt 动态按需注入。

这种“动静分离”的复用逻辑，既保持了面对多变环境时的灵活性，又保障了高频成熟路径的确定性。

### 实验评测：跨阶段任务上的全面突破

为了检验 ExpG 的普适性，研究团队选择了涵盖智能体工具使用不同阶段的三大主流基准：侧重工具选择的 **MetaTool**、侧重复杂多步调用的 **API-Bank**，以及业界难度极高的函数调用基准 **BFCL-V3**。评测模型横跨 GPT-5 nano、DeepSeek-V3 以及 Qwen3 系列模型，并引入了 Few-shot、DRAFT（文档动态优化）和 Mem0（记忆增强框架）作为对比基线。

评测采用三轮独立运行的平均成功率（Avg@3）与至少成功一次的通过率（Pass@3）作为双重核心指标。

从对比数据来看，ExpG 在所有模型规格和所有基准数据集上均取得了第一名。以参数量较小的 Qwen3-8B 为例，在加入 ExpG 机制后：

*   在 MetaTool 上，Avg@3 从无增强状态下的 64.92% 攀升至 72.85%，Pass@3 从 70.73% 提高到 78.41%；

*   在综合评测中，总体 Avg@3 提升了 **7.41 个百分点**，Pass@3 提升了 **6.92 个百分点**；

*   在更严格考察首次调用准确度的 Pass@1 指标上，差距进一步拉大，体现出引导机制在抑制盲目试错上的即时效果。

更为关键的发现是**跨参数层级的逆袭**：配置了 ExpG 的 Qwen3-8B 模型，在多项子任务中的得分直接追平甚至反超了未启用该机制的 Qwen3-32B 乃至更大规模模型。这一现象充分佐证：很多时候 Agent 工具调用的失败，并不是基础模型缺乏泛化语法或推理知识，而是缺乏精准的操作上下文与边界防护。

<img src="/images/2608.03403v1/ablation.webp" alt="ExpG 各阶段消融实验对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 消融与深度分析：核心收益到底来自哪里？

为了厘清 ExpG 各个组件的独立贡献，论文展开了细致的消融与对比分析。

#### 1. 阶段递进的消融验证

如上图消融曲线所示，以 Qwen3-8B 为基座：

*   若仅保留**经验获取**阶段，将其原始归因结果直接注入推理，性能已经明显优于传统的 Few-shot 学习。这说明经过多维度归因的结构化数据，提供给模型的监督信号远比原始问答轨迹更清晰。

*   加入**经验提炼**阶段后，模型表现迎来了最显著的跳升。这一步将散乱调用凝炼成“边界与最佳实践”，滤除了无效噪音，证明了全局经验聚合的必要性。

*   引入**经验复用**的动静分流机制后，系统在长期多任务流中表现出更强的稳定性，最终构成了完整的性能高点。

#### 2. 为什么等价类采样优于传统采样？

在提炼阶段，如何从庞大的历史调用中挑选代表性样本是核心难题。论文对比了三种采样策略：均匀抽样的 Instance-balanced、强制各类等量的 Class-balanced，以及 ExpG 提出的 Equivalence-class-based。

<img src="/images/2608.03403v1/sampling_comparison_3d_d_q.webp" alt="不同采样方法在分布对齐（KL 散度）与多样性（熵）上的表现对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

对比维度包括与原始分布的一致性（通过负 KL 散度衡量）以及类别多样性（通过香农熵衡量）。实验表明：

*   Instance-balanced 虽然与原分布拟合好，但在长尾异常模式上的覆盖极差，多样性指标低；

*   Class-balanced 虽能覆盖稀有模式，但严重扭曲了真实环境中的失败发生概率，导致总结器对高频错误产生误判；

*   **ExpG 的等价类采样方法则在两者之间达成了极佳的帕累托最优**：在高采样比下不仅保持了极高的熵（多模式覆盖），还能在分布对齐指标上媲美甚至超越全量均匀采样。

#### 3. 弱模型能否实现“经验自演化”？

ExpG 在主实验中默认采用高阶模型（如 Qwen3-max）担任评估器与总结器。一个自然的质疑是：这种提升是否仅仅是大模型的“离线蒸馏降维打击”？

团队通过下调评估器与总结器的尺寸进行了压力测试。数据显示，尽管采用更小的模型（如 Qwen3-32B 或 Qwen3-8B）作为裁判会导致经验提取质量轻微受损，但即使**完全使用 Qwen3-8B 自身作为评估器与总结器**，ExpG 依然在三大数据集上带来了稳健的大幅度提升，其收益相较于基线依然显著。这直接证明了 ExpG 机制的内在自洽性：**Agent 系统完全有能力通过自身历史轨迹的结构化自省，实现无外力介入的工具使用自主演化**。

### 迈向“可训练工具”的新范式

在实际案例研究中，作者梳理了 ExpG 解决的典型故障：它帮助模型精确辨识出语义相近但参数限制完全不同的两个金融查询工具；它纠正了多步调用中后置工具对前置工具输出键名的错误假设；更重要的是，它在下游工具返回毫无意义的“Server Timeout”时，成功引导模型根据经验指引执行降级策略，而不是反复重试同一错误路径。

过去很长一段时间，学术界与工业界往往将工具视为给定的、死板的外部接口，所有的自适应压力都被推给了中心大语言模型。而 ExpG 提供了一种颠覆传统认知的视角：**工具本身应当是“可进化的”，工具与智能体交互的协议应当随着执行经验的沉淀而动态重构**。

这种机制为工业级 Agent 系统的架构设计带来了实质性启发。在未来的落地实践中，我们或许不再需要针对每一个频繁变动的内部 API 进行耗时费力的大模型微调，也不必堆砌臃肿繁杂的静态系统提示词。通过在运行时部署类似 ExpG 的经验提炼网关，智能体能够在实际业务环境中“边走边学”，在一次次失败与纠偏中为每一个工具自动摸清边界、立好规矩，最终实现真正稳健可靠的自主交互。
