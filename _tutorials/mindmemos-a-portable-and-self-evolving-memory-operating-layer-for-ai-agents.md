---
layout: default
title: "华为提出MindMemOS：从静态存储走向自演化记忆，LOCOMO准确率达到94%"
description: "MindMemOS 从根本上重构了记忆的表征形态，提出“实体–属性–时间”三维组织图谱，并配套开发了一套能够主动演化的算法引擎：在线通过紧凑检索与动态图融合吸收上下文，离线通过类似人类睡眠的“做梦（Dreaming）”机制合并冗余与化解冲突；同时，它利用用户的隐式纠错信号完成校准。"
arxiv_id: "2608.12428"
paper_published: "2026-08-12"
published_at: "2026-09-18T13:15:07.871050+08:00"
topics:
  - "知识系统"
  - "AI Agent"
tags:
  - "Entity-Property Timestructure"
  - "Higher-Order Pattern Discovery"
  - "Implicit Corrective Feedback"
  - "LOCOMO"
  - "MindMemEvolve"
  - "MindMemOS"
related_tutorials:
  - "higher-order-linear-attention"
  - "the-missing-layer-of-agi-from-pattern-alchemy-to-coordination-physics"
  - "from-atomic-actions-to-standard-operating-procedures-iterative-tool-optimization"
  - "beyond-the-capability-boundary-zeroth-order-optimization-for-self-evolving-llm-a"
seo_title: "MindMemOS: A Portable and Self-Evolving Memory Operating Layer for AI Agents"
---

<p class="paper-original-title" lang="en">MindMemOS: A Portable and Self-Evolving Memory Operating Layer for AI Agents</p>

<img src="/images/2608.12428v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

赋予大语言模型（LLM）长期记忆，一直是通往自主 Agent 的关键路径。然而，当前主流的 Agent 记忆系统大多陷入了一种僵局：记忆层在部署上线那一刻起，其模式定义、信息组织与检索逻辑就基本固化了。即便引入了外部向量数据库或图谱，系统也只是机械地做增删改查。当交互周期从几天拉长到数月，大量冗余、陈旧、甚至自相矛盾的事实就会像垃圾一样在库中堆积，导致检索召回逐步劣化。更为致命的是，执行任务时积累下的过程性经验和操作技能，往往与事实性记忆完全割裂，无法沉淀为 Agent 持续进化的能力。

> ArXiv URL：https://arxiv.org/abs/2608.12428v1

针对这种静态记忆架构的顽疾，华为技术团队推出了 **MindMemOS**。这项工作不再将记忆简单视作外挂的检索数据库，而是将其定位为一层可自演化、高便携的 Agent 记忆操作系统。MindMemOS 从根本上重构了记忆的表征形态，提出“实体–属性–时间”三维组织图谱，并配套开发了一套能够主动演化的算法引擎：在线通过紧凑检索与动态图融合吸收上下文，离线通过类似人类睡眠的“做梦（Dreaming）”机制合并冗余与化解冲突；同时，它利用用户的隐式纠错信号完成校准，甚至能将日常调用工具的真实轨迹反哺为可复用的结构化技能。

从基准测试结果来看，MindMemOS 在长程对话记忆基准 **LOCOMO** 上斩获了 94.03% 的整体准确率，显著超越了此前最前沿的 EverOS 与 Mem0；在复杂个性化评测 **PersonaMem** 上达到 70.63%；而在基于真实轨迹的技能演化实验中，其驱动的 **MindSkillEvolve** 使得电子表格操作基准 **SpreadsheetBench** 的任务成功率在初始基线之上净增 9.2 个百分点。

<img src="/images/2608.12428v1/homepage_combined.webp" alt="MindMemOS 在多项长程记忆与技能演化基准上的综合表现" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 静态外挂记忆的三重困境

在深入 MindMemOS 的架构之前，有必要理清为什么现有的 Agent 记忆方案难以支撑开放场景的长期运行。当前学术界与工业界主要沿着两条线索探索记忆机制：一条是模型内隐式记忆，如通过 LoRA 适配器、测试期微调或专用的隐空间向量来编码状态；另一条则是外挂系统式记忆，即依赖文本块、向量库与知识图谱的工程编排，代表性方案包括 Mem0、memU、Zep 等。

内隐式记忆通常强依赖于特定的底座模型架构和昂贵的训练开销，难以在各类异构 Agent 框架间平滑迁移。而外挂式记忆虽然移植性好，却往往受制于预先设定的规则体系，暴露出三大核心缺陷。

其一是表示粒度的两难。如果完全采用自由文本的非结构化存储，在跨越数十个对话会话后，时间序列上的因果推理与细粒度属性更新几乎无法保证；如果采用人工预先定义的结构化 Schema，在面对现实世界开放域任务的漂移时，固定字段又会迅速失效，无法灵活提取出“用户的决策风险偏好”等深层高阶特征。

其二是增量写入带来的信息熵增。用户在不同交互阶段的想法会变化，甚至会推翻先前的设定。若只做在线流式提取，记忆库中必然会出现大量同一实体的新旧矛盾条目，导致检索器在多跳问答中命中相互打架的噪声碎片。

其三是陈述性记忆与程序性技能的脱节。大部分系统把精力放在了“记住用户喜欢什么”这类事实性陈述上，而 Agent 最核心的生产力资产——在特定环境中调用工具、排查报错、纠正步骤的**程序性经验**，却被当作一次性日志丢弃，无法转化成随用随新的 Agent 技能库。

MindMemOS 的核心突破，在于将记忆层与上层 Agent 解耦，构建了一个包含记忆结构层与算法生命周期层的完整底座。

<img src="/images/2608.12428v1/system-architecture.webp" alt="MindMemOS 整体系统分层架构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 实体–属性–时间：三维记忆图谱的动态构建

面对真实世界信息的开放性与演变特征，MindMemOS 放弃了将记忆压缩为单一向量或平面键值对的做法，设计了以**实体（Entity）**、**属性（Property）**和**时间（Time）**为坐标轴的三维记忆结构。

在这个三维坐标系中，实体充当了锚点的角色，用来绑定跨对话、跨文档捕获的具象对象（如特定的人、软件项目、办公工具或物理地点）。属性维度则不局限于一阶显式事实（例如“张三住在深圳”），还允许承载跨多轮观察综合推断出的二阶/高阶模式（例如“张三倾向于在周五下午做出激进的技术选型”）。时间维度则为每一个“实体–属性”对串联起了一条不可篡改的版本演进时间线，使得系统能够在不粗暴覆盖历史上下文的前提下，精准标记属性的生效期、过期态与最新状态。

<img src="/images/2608.12428v1/memory_structure_3d.webp" alt="实体、属性与时间构成的三维记忆结构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了在兼顾灵活度的同时保障结构化精度，MindMemOS 在记忆写入（Memory-Add）阶段提供了两种互补策略：开箱即用、无预设模式的 **MindVanilla** 模式，以及由领域建模指导的 **MindSchema** 模式。

在 **MindSchema** 模式下，系统经历四个关键步骤将动态对话转化为三维子图：

1. **情境切分（Episode Segmentation）**：系统内置基于 LLM 的切分模块，支持同步与异步模式。在异步流式处理中，消息先进入缓冲池，直至检测到话题发生实质漂移时自动聚合成完整的情境切片，从而摆脱了按物理会话轮次切分导致的语义割裂。

2. **三维记忆抽取**：将切片映射到目标场景的模式定义中，提取实体、属性及带有绝对时间戳的取值，同时自动生成一个情境兜底实体，确保那些尚未被模式显式覆盖的长尾背景信息不丢失。

3. **实体对齐与融合（Entity Fusion）**：将新提取的子图作为检索 Query，拉取全局图谱中的近邻实体，完成同义对齐与冲突校验。若实体绑定了高阶模式定义，融合引擎还会触发深层推理，重新提炼该实体的行为画像。

4. **图结构合并（Graph Merge）**：通过节点合并与关系边更新，将局部子图无缝拼接到全局三维网络中。

<img src="/images/2608.12428v1/modeling_guided_memory_construction.webp" alt="MindSchema 驱动的三维记忆动态构建流程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在检索端，三维图结构如果直接进行无节制的全局图展开，会产生巨大的 Token 消耗与延迟。为此，系统采用了**紧凑检索（Compact Search）**机制。它在外层由一个轻量化 Agent 控制器进行迭代式规划，决定何时终止检索；在内层则通过 BM25 稀疏检索与密集向量嵌入的倒数秩融合（RRF），执行双向多路径遍历。系统既可以自“实体”向下钻取其属性时间线（前向遍历），也可以根据属性线索反查所属实体网络（反向遍历），在极窄的候选空间内实现高召回率。

<img src="/images/2608.12428v1/compact_search.webp" alt="具备内外层协作的紧凑双向检索模块" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 做梦与纠错：维持记忆生命力的自我进化

如果说在线写入是记忆系统的“白天进食”，那么离线维护机制就是维持认知健康的“夜间睡眠”。增量追加式的写入不可避免地会在图谱中留下局部碎片与相互冲突的版本。MindMemOS 为此设计了两大自演化柱石：离线的 **Dreaming（做梦机制）** 与交互驱动的 **Feedback（反馈校准机制）**。

Dreaming 是一个完全离线运行的图谱重构进程。系统不会愚蠢地遍历整个知识库，而是以设定的回溯窗口为基准，挑选尚未固化的新写入记录作为种子节点，仅将其局部邻域内的活跃记忆聚集为以实体为中心的簇。随后，系统执行严格的“检测–执行”两阶段流水线：首先通过专门的 LLM 提示词扫描该簇内的逻辑矛盾、同义冗余、互补碎片与陈旧过时条目；经过置信度校验后，再下发保守的变异计划。变异计划遵循非破坏性原则，优先进行语义合并、关系边补全与过时标记，并完整保留新旧条目之间的溯源链路（Provenance）。实验显示，该机制在保持甚至提升问答精度的前提下，能带来约 22.5% 的活跃记忆体积压缩率（AMCR），有效抑制了记忆膨胀。

<img src="/images/2608.12428v1/dreaming_flow.webp" alt="离线记忆整合的 Dreaming 工作流" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

与机器自主整理互为补充的，是人在回路的 **Feedback** 机制。现实中，人类用户很少会像数据库管理员那样直接下达类似“请把记忆库里的字段 A 修改为 B”的显式指令，绝大多数纠偏隐藏在普通的对话互动中——例如用户抱怨“我昨天就跟你说过我不喝美式咖啡了”或者推翻 Agent 的某个错误方案。

MindMemOS 精巧地区分了显式反馈与隐式反馈。针对隐式反馈，系统以后台异步方式解析最近的会话轮次，检测用户的纠错语气与意图，并将该信号打上持久化范围标签：是仅对当前任务生效的**任务临时型（Task-Temporary）**，还是带有限定条件的**场景特定型（Scenario-Specific）**，抑或是全局通用的**长期规则（Long-Term）**。只有后两类才会被编排为合法的记忆变更事务，而临时型指令则被严格阻断在持久化存储之外。这一判断有效避免了将“这封邮件今天别发”误解为“以后永远不要发邮件”的过度泛化灾难。

<img src="/images/2608.12428v1/feedback_flow.webp" alt="显式与隐式反馈处理工作流" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

更具前瞻性的是，MindMemOS 引入了针对记忆 Schema 本身的演化算法 **MindMemEvolve**。过去，记忆图谱抽什么字段完全靠工程师盲猜。MindMemEvolve 采用评估信号驱动的进化搜索策略：系统在特定场景的任务验证集上，利用大模型评估器对检索问答质量打分，并将错误归因转化为变异算子的驱动力。大模型在此不再扮演随机突变发生器，而是根据评测反馈中的失分原因，主动对实体属性定义进行针对性修补（Induced Mutation），并结合跨代交叉（Crossover），自主挖掘出该场景最需要的高阶特征表达。

### 技能自演化：将交互轨迹沉淀为工具执行力

Agent 区别于简单聊天机器人的核心在于工具调用与复杂任务落地。MindMemOS 在业界首次将记忆操作系统的边界从“数据存储”推进到了“技能（Skill）的生命周期管理”，提出了由轨迹驱动的 **MindSkillEvolve** 框架。

在许多实际落地场景中，开发者为 Agent 编写的 Prompt 模板或 API 调用逻辑往往在特定边界条件下频频翻车。MindMemOS 复用了记忆层的 Add 接口，配合轻量化的 SDK Hook，能够在 Agent 运行任意任务时无感录制其执行轨迹（Trajectory）。更重要的是，SDK 会将该轨迹与当前生效的技能版本进行哈希锚定与版本绑定，在云端维护一条带有可追溯链路的技能版本链（Version Chain）。

当积累了足够数量的未消化轨迹后，MindSkillEvolve 自动触发进化分析。它将离散的多步工具调用提炼为包含任务目标、转折点、工具失败模式与有效策略的证据链，支持两种进化模式：

- **无监督自省演化**：聚合同一技能在多个复杂案例中的执行通病，自主修正技能指导文本中的边界防御与调用建议；

- **打分监督演化（Evolve-Sup）**：利用外部任务得分作为强监督信号，强化高分轨迹中涌现出的优良策略（如前置参数检查、分步重试机制），惩罚低分轨迹中的常见死锁行为。

通过这种方式，Agent 在面对专业软件、办公脚本等高难度环境时，不再依赖工程师没完没了地人工修补 Prompt，而是能够“越用越熟练”。

### 实验评测：长程问答与实战技能的双重突破

为了全面检验系统的记忆保真度、推理能力与演化有效性，MindMemOS 在权威评测集上进行了细致对比。

在长上下文与多会话对话记忆的黄金基准 **LOCOMO**（涵盖 10 组超 300 轮交互的宏大对话、近 2000 个复杂测试题）上，MindMemOS 表现出极高的记忆召回与推理水准。实验严格对齐了当前顶会主流工作 EverOS 的评测基线与底层模型环境。在单跳（Single-hop）、多跳（Multi-hop）、时间序列（Temporal）和开放域（Open-domain）四个主要维度上，MindSchema 模式均取得了优异成绩，最终录得 94.03% 的总准确率，相较于此前处于领先地位的 EverOS（93.05%）提升近 1 个百分点，相较于 Zep（85.22%）和 MemOS（80.76%）更具显著优势。即便在不依赖任何人工领域建模配置的 MindVanilla 极速模式下，系统依然取得了 87.60% 的总准确率，证实了其紧凑混合图检索机制本身的底座实力。



而在面向长期用户画像与风格自适应推荐的 **PersonaMem** 基准测试中，由于引入了针对用户实体的二阶/高阶属性推理机制，MindMemOS 的整体准确率达到了 70.63%，大幅拉开了与通用记忆系统的差距。这证明了三维实体结构在建模诸如“性格特征偏好”“隐式消费限制”等高阶属性时的不可替代性。

更具说服力的是技能演化在真实环境中的成效。在对工具调用精度要求极高、稍有参数或步骤偏差就会导致执行报错的电子表格基准 **SpreadsheetBench** 上，MindSkillEvolve 展现了惊人的实战价值。通过对 Agent 历史执行轨迹的持续提炼与监督进化，Agent 在该基准上的任务成功率在初始基线技能的基础上**净提升了 9.2 个百分点**。这种无需人类参与、纯靠“实战反思”获得的工具技能跃升，展示了记忆操作系统作为 Agent 核心基础设施的巨大潜力。

### 走向自适应的 Agent 记忆新范式

回顾 MindMemOS 的设计，它的根本启发在于：**记忆不应当是静态的“保险箱”，而应该是一个具有代谢与学习能力的“有机体”**。

在架构层面，它通过实体、属性、时间三维坐标系，让多轮交互中的事实变更有迹可循，优雅化解了非结构化文本的模糊性与固定 Schema 的僵化性；在运行机制层面，它通过模拟人类大脑的离线做梦和用户隐式意图解析，实现了记忆的主动提纯与自我修复；在能力延展层面，它打破了事实数据与执行逻辑的壁垒，让 Agent 能够将操作教训内化为技能版本。

随着多智能体协作、云端协同与具身智能等场景逐步铺开，运行在端侧与云侧的各类 Agent 势必面临更长的部署生命周期。MindMemOS 所确立的这套便携式、自演化记忆操作层，不仅为高阶 Agent 的个性化与长期自洽运行提供了坚实的技术蓝本，更指明了下一代 Agent 底座系统从“单纯检索数据”走向“持续积累认知”的演进方向。
