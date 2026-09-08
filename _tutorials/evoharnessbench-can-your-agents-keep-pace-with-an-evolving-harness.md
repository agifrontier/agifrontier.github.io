---
layout: default
title: "EvoHarnessBench：装具不断扩增，Agent为何反而遭遇“装具诱发遗忘”？"
description: "然而由 Salesforce Research、北卡罗来纳大学教堂山分校（UNC Chapel Hill）与威斯康星大学麦迪逊分校（UW-Madison）联合提出的最新基准 EvoHarnessBench ，却揭示了一个反直觉且严峻的现实： 仅仅是外部装具的规模扩大。"
arxiv_id: "2609.04280"
paper_published: "2026-09-03"
published_at: "2026-09-08T13:15:08.162312+08:00"
topics:
  - "AI Agent"
tags:
  - "EVOHARNESSBENCH"
  - "LLM agents"
  - "deployment evaluation"
  - "harness evolution"
  - "harness-induced forgetting"
  - "self-evolving adaptation"
related_tutorials:
  - "adaptation-of-agentic-ai"
  - "search-over-self-edit-strategies-for-llm-adaptation"
  - "kimi-dev-agentless-training-as-skill-prior-for-swe-agents"
  - "is-your-code-generated-by-chatgpt-really-correct-rigorous-evaluation-of-large-la"
---

<p class="paper-original-title" lang="en">EVOHARNESSBENCH: Can Your Agents Keep Pace with an Evolving Harness?</p>

<img src="/images/2609.04280/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在大模型智能体（LLM-based Agent）的落地实践中，开发者往往把注意力放在基座模型的推理能力或单次 Prompt 的工程设计上。然而，决定一个 Agent 究竟能观察到什么、能执行什么动作的核心基础设施，其实是它所处的“装具”（Harness）——即围绕模型构建的外部工具箱（Tools）、可复用技能库（Skills）以及可协同调用的子专家智能体（Specialist Agents）。随着企业业务的持续迭代，这套装具绝非一成不变，而是处于无休止的动态扩增之中。例如 Salesforce 的 Agentforce 生态持续接入新系统，OpenAI 的开源技能库也在频繁合并新技能与新接口。

> ArXiv URL：https://arxiv.org/abs/2609.04280

直觉上，赋予 Agent 更多的工具与专家，它的能力边界理应单调递增。然而由 Salesforce Research、北卡罗来纳大学教堂山分校（UNC Chapel Hill）与威斯康星大学麦迪逊分校（UW-Madison）联合提出的最新基准 **EvoHarnessBench**，却揭示了一个反直觉且严峻的现实：**仅仅是外部装具的规模扩大，就会导致 Agent 在原本能够轻松解决的旧任务上频繁溃败，诱发所谓的“装具诱发遗忘”（Harness-Induced Forgetting）**。即使底层模型权重毫无变化、旧任务所需的 API 依然完好可用，Agent 依然会迷失在庞大的干扰项与复杂的路由拓扑中。

EvoHarnessBench 首次将持续学习的研究视角从传统的“任务流非平稳”切换到了“外部装具非平稳”。该基准基于真实企业级验证环境构建了 17 条多阶段装具流，涵盖 802 个任务、520 个可执行工具、42 项参考技能与 62 个专家智能体。实验进一步表明，当前主流的自演化机制（如记忆库、自省 Prompt）在演进装具面前极其脆弱，“保留旧能力”与“适应新能力”甚至常常呈现出不可调和的负向拉扯。

<img src="/images/2609.04280/framework.webp" alt="EvoHarnessBench 核心评估框架：外部装具逐级演进，暴露装具诱发遗忘与适应困难" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 从任务变化到装具演化：被忽视的非平稳挑战

传统的 Agent 评估基准大多基于静态假设：系统配备一组固定的工具，并在给定的任务集上运行。即便是涉及持续学习或终身学习的研究，也普遍将环境的“非平稳性”（Non-Stationarity）设定在任务流上——任务的主题在变，但 Agent 可用的工具接口与系统框架是死板锁定的。另一类探索自演化智能体（Self-Evolving Agents）的工作，虽然允许系统积累经验或生成技能，但那属于 Agent 内部主导的代码生成或记忆更新，考察的是生成技能本身的质量，而非面对外部环境突变时的鲁棒性。

真实工业场景的运行逻辑恰恰相反。底层模型为了追求稳定性通常是冻结的，上层任务需求也是恒定的（例如处理退货、生成财务报表），真正每天都在剧烈变动的是外部装具：安全团队接入了新的权限验证工具、开发团队上线了一批第三方 SaaS 插件、运维团队配置了新的业务流程 SOP。

这种外部装具的持续膨胀带来了两重根本性困难。首先是**能力保留（Retention）**：虽然此前解题所需的工具依然在装具中，但它们现在被淹没在上百个功能相似的干扰候选项中，原本清晰的决策路径被严重稀释。其次是**累积经验下的适应（Adaptation under Accumulated Experience）**：如果 Agent 拥有自演化或经验积累模块（如少样本记忆库、轨迹反思），这些经验全是在狭窄的旧装具下沉淀出来的。当更优越的新工具被引入时，陈旧的经验不仅无法提供助力，反而会形成先入为主的偏见，将系统死死锚定在低效甚至失效的过时路径上。

<img src="/images/2609.04280/dataset_overview.webp" alt="EvoHarnessBench 数据集统计与真实生态演变趋势的对齐关系" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 嵌套装具流与双重评测协议

为了精确解耦外部装具扩增带来的冲击，EvoHarnessBench 确立了确定性的装具流构建范式。基准基于 EnterpriseOps-Gym（EOG）和 Agentic Last Exam（ALE）两个高难度沙盒环境构建。整个评测围绕“外层装具演化（Outer Harness Evolution）”展开，装具流被切分为若干个离散阶段：




{% raw %}$$ \mathcal{H}_{1} \subseteq \mathcal{H}_{2} \subseteq \cdots \subseteq \mathcal{H}_{T} $${% endraw %}



其中每个 $\mathcal{H}_{t}$ 代表阶段 $t$ 向系统暴露的装具能力集合。能力的投放遵循现实世界的扩展逻辑：先释放最基础、最通用的核心能力，再逐步向低频、特定的长尾扩展能力铺开。与之对应，测试集中的每一个任务 $x_i$ 都附带有严格的最小能力子集标注 $\mathcal{C}_i$。任务的登场时机 $t_i$ 被严格约束在它首次完全可解的阶段：




{% raw %}$$ t_i \coloneqq \min \{ t : \mathcal{C}_i \subseteq \mathcal{H}_t \} = \max_{c \in \mathcal{C}_i} r(c) $${% endraw %}



这种设计的精妙之处在于，它施加了明确的“新能力使用压力”：每一个阶段新加入的任务，都至少强制依赖一项当前阶段刚刚解锁的新工具、新技能或新智能体，杜绝了靠旧方案蒙混过关的可能。在这一外层演化轴线之下，EvoHarnessBench 设立了两种互补的评估模式：

1. **部署评测（Deployment Evaluation）**：在此模式下，系统在各阶段之间不携带任何历史经验，每个阶段的装具都被视作全新的黑盒独立评估。这一设置剥离了一切算法记忆因素，纯粹测量基座模型在面对不断扩大的候选池时，能否在利用新能力的同时抵抗干扰。

2. **自演化适应评测（Self-Evolving Adaptation Evaluation）**：允许系统在各阶段维护一个持久化的适应状态 $z_t$（包括情景记忆、反思规则或提示词）。在阶段 $t$，系统可以通过专门的适应切分集来更新 $z_t$，随后在保留任务与新增任务上同时进行检验。该模式直接衡量历史经验究竟是成了 Agent 进化的跳板，还是成了作茧自缚的枷锁。

评测指标不仅考察绝对成功率，还引入了前向迁移率（Forward Transfer, FWT）与后向迁移率（Backward Transfer, BWT）。FWT 衡量系统利用过去经验适应当前阶段新任务的能力，而 BWT 则量化了随着装具演进，系统在历史任务表现上的增益或滑坡。

<img src="/images/2609.04280/fwt_bwt_tool_verifier_compact.webp" alt="工具演化维度下的前向迁移与后向迁移分析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 工具轴演化：检索干扰与记忆表征的双刃剑

在工具演化维度（Evolving Tools），EvoHarnessBench 覆盖了多达 520 个可执行的真实 API。测试结果直观展现了“装具膨胀”对单智能体系统的杀伤力。

在完全理想化的参考对照组中（即给每个任务只提供它所依赖的 Oracle 最小工具集），基于 GPT-5 的 ReAct 系统能达到极为可观的解题率。然而一旦接入完整的累积工具目录，性能便发生显著坍塌。模型并非不知道怎么执行具体 API，而是在面对几十个参数格式各异、语义高度重叠的工具描述时，产生了严重的注意力涣散。

为了对抗这种退化，研究团队评估了多种持久化自演化机制，包括纯轨迹回放（Raw Memory）、结构化反思检索（ReasoningBank、MemToolAgent）、提示词优化（GEPA）以及元装具代码调优（Meta-Harness）。实验结果给出了关于“经验记忆”的深刻启示：

经验的价值完全取决于它能否有效缩减 Agent 在庞大工具空间中的搜索半径。简单的原始轨迹回放（Raw Memory）效果有限，而像 MemToolAgent 和 ReasoningBank 这类提取出结构化决策逻辑的方案取得了显著领先（成功率分别达到 $38.6\%$ 与 $36.9\%$，大幅高于基础 ReAct 的表现）。

但更耐人寻味的是记忆表征的粒度陷阱。当研究人员尝试在 ReasoningBank 中显式注入详尽的工具 Schema（参数说明与调用模式）时，性能不仅没有上升，反而剧烈下挫至 $30.3\%$。过多的结构化细节反客为主，挤占了上下文窗口并引入了新的格式幻觉。而在多智能体系统（如 AutoGen 与 DeLM）中，共享记忆甚至成了一剂毒药：抽象或泛化的经验容易诱导各子智能体产生冗余的工具探索行为，导致系统协同成本激增、通信陷入死循环。

<img src="/images/2609.04280/fwt_bwt_skill_verifier_compact_claude.webp" alt="技能演化维度下的迁移表现" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 技能轴演化：分阶段累积反成性能瓶颈

不同于开箱即用的工具，技能（Skills）代表着封装好的操作流程、业务 SOP 或多步骤推理规范。在技能演化维度（Evolving Skills），随着外部注入的流程规范从基础操作扩展到复杂的长尾组合，系统面临的挑战转化为“程序化依赖理解”。

<img src="/images/2609.04280/skills_share.webp" alt="技能学习中的表征覆盖率与阶段共享特征" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

除了直接向 Agent 注入外部标准技能外，研究团队还考察了自演化技能学习机制——让 Agent 在演进过程中根据环境反馈自动抽象并提炼出可复用的代码技能函数。实验揭示出两个关键瓶颈：

第一，**生成技能的“虚假繁荣”**。通过对比模型自主生成的技能库与基准标注的核心能力覆盖率发现，Agent 提炼出的很多技能虽然在短期的某些任务上取得了成功反馈，但其抽象出来的逻辑与真实环境中的核心子技能空间只有很低的重合度。这种粗糙的泛化在装具演进到下一个阶段时就会暴露出脆弱性。

第二，**序列累积的负面效应**。在面对相同总量的任务经验时，让 Agent 沿着分步演化的装具流逐级迭代技能，其最终效果反而落后于将所有数据一次性喂给模型的批处理模式（即使是表现最优的 Batch Teacher Feedback 算法，其分阶段渐进累积的准确率也仅为 $22.1\%$，明显低于一次性全量学习的 $23.6\%$）。这表明当前的技能自演化算法缺乏自我校准与重构机制，早期在狭隘上下文下固化的劣质技能代码，会在后续阶段作为基石继续污染新技能的构建。

<img src="/images/2609.04280/fwt_bwt_agent_verifier_compact.webp" alt="专家智能体演化维度下的迁移与衰减" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 智能体轴演化：路由漂移引发断崖式遗忘

如果说工具与技能的演进还局限在单智能体的认知负荷内，那么在专家智能体轴线（Evolving Agents）上，多智能体协同网络暴露出的问题则更为尖锐。在此轴线上，装具的演进体现为主控智能体可调配的专业子智能体池不断扩张（涵盖 62 个不同的 Specialist Agents）。

多智能体系统在此展现出了惊人的装具诱发遗忘现象。在初期阶段能够稳定解决的问题，当子智能体池扩充后，表现出现了断崖式下滑。

<img src="/images/2609.04280/routing_quality.webp" alt="路由质量评估：专家池扩大导致分发准确率显著下降" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了排查遗忘的真实根源，论文追踪了任务在不同装具阶段的委派分布（Delegation Distribution）。原先最直观的假设是：新增的专家智能体充当了“噪音分流器”，把原本该给老专家的任务给抢走了。然而深度分析推翻了这一猜想：在新专家加入后，被遗忘任务被分派给新不相关专家的概率并没有出现系统性的暴增。



决定任务成败的真正元凶是**路由漂移（Routing Drift）**与**关键委托覆盖的丢失**。如图所示，那些在新装具下遭遇失败的任务，其委派漂移率从未发生遗忘时的 $13\%$ 飙升至 $25\%$。随着可选专家变多，中心调度器的决策网络在细微扰动下解构了原本稳定成型的协作回路，它不再去呼叫那些完成任务所必须依赖的特定核心专家。换言之，**不是新专家的“越俎代庖”导致了错误，而是调度器在眼花缭乱的选择中丢失了对正确协同链路的坚持**。

### 核心结论：前向适应与后向保留的不可调和性

综合工具、技能、智能体三大维度的海量实验，EvoHarnessBench 提炼出了当前大模型 Agent 架构面临的核心矛盾，其中最值得全行业警惕的结论体现在两个层面：

首先是**装具诱发遗忘的普遍性**。如前文所述，在底层模型参数冻结的纯黑盒评测中，仅仅是暴露给 Agent 的外围装具规模扩大，基准后向迁移率（BWT）在各大主流框架中均呈现出普遍的负值，性能滑坡幅度在部分长链路场景下甚至达到 $-34.7\%$。这种遗忘完全由上下文环境的熵增所引起，是传统模型端防遗忘算法（如正则化、LoRA 冻结）无法触及的盲区。

其次是**适应与保留的拉扯困境**。在自演化机制的参与下，数据呈现出极度割裂的态势：那些能够大幅提升当前阶段新任务得分（高 FWT）的策略，往往会加速对旧任务解法的破坏（极低的负 BWT）；反之，旨在巩固旧有调用模式的保守记忆机制，又使 Agent 面对功能强大的新工具时视若无睹。系统整体宏观指标的微弱上升，往往掩盖了微观层面上新老能力的剧烈博弈。

这为未来的 Agent 架构演进指明了清晰的技术探索方向：

单纯依赖单层、被动的提示词追加或记忆外挂已经无法应对复杂的生产环境。Agent 系统亟需引入**双层适应（Bi-Level Adaptation）机制**——内层回路负责在既定装具下快速积累经验，而外层回路必须具备元认知监控能力，能够周期性地识别并清理那些因外部装具升级而过时、甚至产生误导作用的历史经验；同时，调度中心必须具备鲁棒的子空间隔离策略，防止庞大的长尾装具在底层反向击穿核心业务路径。在通往真正生产级自主智能体的道路上，让系统学会与不断演进的外部装具共存，其重要性绝不亚于打磨模型本身的推理上限。
