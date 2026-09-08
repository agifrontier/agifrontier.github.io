---
layout: default
title: "腾讯 SiriusDeliver：交付时长从 228 降至 23 分钟！数仓任务交付 Agent 落地实践"
description: "为了解决这一痛点，腾讯联合武汉大学提出了面向生产级数仓任务交付的端到端自动化智能体系统 —— SiriusDeliver 。该系统不仅能写代码，更接管了从业务需求解析、环境元数据对齐、工件全生命周期质检到平台自主提交与自愈诊断的完整链路。"
arxiv_id: "2608.09185"
paper_published: "2026-08-10"
published_at: "2026-09-08T13:15:08.162312+08:00"
topics:
  - "AI Agent"
  - "数据工程"
tags:
  - "DW"
  - "SiriusDeliver"
  - "Tencent Cloud WeData"
  - "artifact lifecycle control"
  - "delivery automation agent"
  - "dependency-aware orchestration"
related_tutorials:
  - "dacomp-benchmarking-data-agents-across-the-full-data-intelligence-lifecycle"
  - "dataflow-an-llm-driven-framework-for-unified-data-preparation-and-workflow-autom"
  - "zero-shot-self-orchestration-with-ledger-based-control-for-improved-llm-coding-p"
  - "researcharena-evaluating-sabotage-and-monitoring-in-automated-ai-rd"
---

<p class="paper-original-title" lang="en">SiriusDeliver: Automating Data Warehouse Delivery at Tencent</p>

<img src="/images/2608.09185v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在绝大多数技术开发者的直觉中，大语言模型（LLM）与代码智能体（Coding Agent）最擅长的工作莫过于“写代码”。只要给出 Schema 和业务口径，模型就能快速吐出一段高质量的 SQL 或 PySpark 脚本。然而，这种基于单点代码生成的设想，在真实的企业级数据仓库（Data Warehouse）生产环境中往往会迅速碰壁。

> ArXiv URL：https://arxiv.org/abs/2608.09185v1

腾讯内部的一项实地调研揭示了一个残酷的行业现实：在数据工程师端到端的任务交付流程中，真正的 SQL 编写耗时占比不足 25%。剩下超过 75% 的精力，全被淹没在复杂的上下文检索、元数据对齐、调度拓扑配置、权限校验、试运行以及执行失败后的跨平台排错中。一个看似简单的指标上线，背后牵涉的是跨系统的元数据联动与严格的依赖编排。

<img src="/images/2608.09185v1/motivation.webp" alt="传统人工交付与智能体驱动交付的对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了解决这一痛点，腾讯联合武汉大学提出了面向生产级数仓任务交付的端到端自动化智能体系统 —— **SiriusDeliver**。该系统不仅能写代码，更接管了从业务需求解析、环境元数据对齐、工件全生命周期质检到平台自主提交与自愈诊断的完整链路。目前，SiriusDeliver 已在腾讯云 WeData 平台上线，并在包含微信支付、腾讯广告、安全风控等核心业务场景中常态化运行两个月。线上 A/B 测试显示，它将数仓任务的端到端交付时长中位数从 228 分钟压缩至 23 分钟，工程师的人工精力投入从 95 分钟直降至 11 分钟，同时实现了 87.2% 的端到端交付成功率。

### 为什么通用 Coding Agent 无法胜任数仓交付？

当前业界以 SWE-bench 等基准为代表的代码智能体，核心假设大多建立在“仓库级软件工程”之上：在受控的单机代码库中定位 Bug、编辑文件并通过单元测试。但生产级数据仓库的交付逻辑与纯代码开发有着本质的区别。在真实生产环境落地过程中，通用智能体面临着三大难以逾越的鸿沟：

1. **多阶段任务的强依赖编排（Multi-stage task composition）**：数仓任务交付并非一步到位的代码生成，它横跨多个强耦合阶段。以最常见的离线数据同步为例，模型在生成同步配置前，必须先确认源表与目标表的 Schema、主键增量策略、历史分区情况，并与上下游作业的调度拓扑建立依赖。通用智能体往往采用单次规划（One-shot planning），极易遗漏隐式前置条件或打乱执行顺序，导致生成出的配置无法被平台接纳。

2. **交付工件的高确定性与隐性风险（Trustworthy artifact delivery）**：数据工程交付的不是一个孤立的代码片段，而是一整套“工件束（Artifact Bundle）”，包括执行代码、调度 DAG、重试策略、运行时资源参数以及权限凭证。语法正确的 SQL 极有可能因为脏数据、分区倾斜或调度依赖错位引发静默的数据质量事故。通用智能体仅靠事后看报错日志的被动修复（Reactive repair），在动辄处理海量数据的生产数仓中试错成本极其高昂。

3. **平台规范与环境的持续演化（Continuous platform adaptation）**：生产平台的 API、调度引擎底层策略、数据字典以及各业务团队的工程规范处于持续迭代之中。依赖静态 Prompt 或硬编码规则的智能体极易迅速失效；而传统终身学习方法往往优化的是模型即时推理的隐式权重，缺乏可解释、可审计、可快速回滚的企业级技能维护机制。

针对这三大系统性断层，SiriusDeliver 并没有盲目加大基础模型的参数量，而是重新设计了一套契合 DataOps 生产范式的智能体架构。

<img src="/images/2608.09185v1/framework.webp" alt="SiriusDeliver 架构概览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 架构解法：解构数仓交付的三大核心引擎

如图所示，SiriusDeliver 的整体工作流将业务需求转化为平台可执行工件的闭环。这一闭环由三大紧密协作的核心模块支撑：分层交付智能体（Hierarchical Delivery Agent）、工件生命周期控制（Artifact Lifecycle Control）以及轨迹驱动的技能演化（Trace-driven Skill Evolution）。

#### 1. 分层交付智能体：维护显式交付状态

为了避免单次规划带来的幻觉与步骤遗漏，分层交付智能体不再把任务当成黑盒处理，而是形式化定义了一个生产就绪的工件束：$A = \langle G, C, P \rangle$。其中，$G$ 代表工作流拓扑结构（单节点或复杂的 DAG 图），$C$ 是节点级的可执行脚本（支持 SQL、PySpark、Flink SQL 等），$P$ 则是涵盖调度周期、依赖配置、资源告警与权限参数的平台配置集。

<img src="/images/2608.09185v1/delivery_agent.webp" alt="分层交付智能体结构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

智能体通过分层技能库组织能力，细分为场景技能、上下文技能、工件生成技能与平台交互技能。它采用“计划-路由-执行-反思”的递进循环，并显式维护一个状态记忆模块。该记忆模块同时存储短期的会话上下文（元数据凭证、依赖解析进度）和长期的工程资产（历史高频配置模板、压缩后的优秀交付轨迹）。这种设计确保了智能体在每一步决策时，清楚掌握当前工件束缺失了哪项必要证据，必须调用什么技能去补齐，杜绝了无前置条件的盲目生成。

#### 2. 工件生命周期控制：双阶段可信校验

即便智能体生成了结构完整的工件束，直接推送到生产集群也是极度危险的。为此，SiriusDeliver 引入了涵盖“预执行诊断（Pre-execution diagnosis）”与“后执行诊断（Post-execution diagnosis）”的双阶段防护网。

<img src="/images/2608.09185v1/lifecycle_control.webp" alt="工件生命周期控制机制" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在预执行阶段，系统拒绝仅依赖大模型的主观判断，而是结合平台确定性规则与元数据语义进行交叉验核：

- **静态规则断言**：通过 AST 语法树解析代码、校验平台参数边界、核查上下游节点调度周期是否存在死锁或不匹配；

- **权限与环境探针**：主动校验当前账号是否具备目标库表的写入权限、集群资源队列配额是否合规；

- **语义逻辑审查**：调用针对性 Prompt，重点审查增量更新条件是否存在全表扫描隐患、主外键关联逻辑是否合理。

只有完全通过预执行门禁的工件束，才会被允许提交至底层 WeData 引擎试运行。若平台执行抛出异常，后执行诊断模块介入。它不仅捕获裸日志，还会联动元数据知识库检索类似的历史排错记录，将复杂的底层引擎调用栈翻译为带有上下文的归因解释，指导智能体完成针对性修复，避免陷入无意义的循环重试。

#### 3. 轨迹驱动的技能演化：从成功中压缩，从失败中修复

如何让智能体系统随着平台规范升级而持续进化？SiriusDeliver 的解法是将每次端到端交付的完整轨迹沉淀下来，在离线阶段进行受限的“技能演化”。

<img src="/images/2608.09185v1/skill_evolution.webp" alt="轨迹驱动的技能演化机制" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

以往的经验回流机制往往只关注失败案例（通过错误反馈修补漏洞），SiriusDeliver 则提出了双向演化机制：

- **针对失败轨迹进行修复**：聚类分析未通过校验或平台报错的共性原因，促使大模型微调特定技能的执行边界或补充缺失的前置检查规则；

- **针对成功轨迹进行剪枝压缩**：在实际生产中，智能体可能调用了冗余的元数据接口或经历了不必要的迂回推理。演化模块通过分析成功的交付路径，识别冗余步骤并压缩调用链路，形成更高效的精简技能。

所有演化生成的候选技能更新，都会进入影子沙箱校验，经过自动化回归测试与平台工程师的人工审核后，才打上版本号推送上线，并保留一键回滚机制。这彻底解决了企业级系统在引入 LLM 自进化时最为担忧的“黑盒破坏稳定性”问题。

### 实验与实战剖析：确定性提升与成本下降

为了验证 SiriusDeliver 的实际效果，研究团队开展了严格的离线控制对照实验与为期数月的线上真实业务评估。

#### 离线评测：复杂计算场景下的韧性

离线基准测试基于腾讯云 WeData 抽取的 200 个真实生产案例（冻结切片，包含实时同步、离线同步、实时计算、离线计算四类任务各 50 例），且演化训练数据与测试集严格物理隔离。

实验对比了通用代码生成智能体在仅挂载技能库、以及挂载 SiriusDeliver 框架下的表现。在以 Claude Code 为底座的评测中，引入 SiriusDeliver 带来了全方位的收益：

- **端到端成功率**：平均成功率提升了 14.5 个百分点，从 71.5% 跃升至 86.0%。尤其在最具挑战的离线计算任务（涉及复杂的 Multi-table Join、分区生命周期管理与调度配置）中，成功率从基线的 62.0% 提升到了 78.0%；

- **交付效率与推理开销**：交互轮数从平均 8.4 轮缩减至 5.4 轮，单任务离线端到端交付耗时从 16.0 分钟缩短至 10.2 分钟。更为关键的是，Token 消耗量降低了 30%（从 124.0K 降至 86.9K）。这直接证实了轨迹演化中“成功路径压缩”机制的有效性，证明高可靠性并不一定要靠无限堆叠思考轮数来实现。

消融实验进一步厘清了各模块的贡献权重：

- 剥离**分层技能编排**，系统端到端成功率暴跌 11.5 个百分点（降至 74.5%），在数据同步类任务中崩塌尤为严重，大量失败源于元数据未齐备时的盲目生成；

- 剥离**工件生命周期控制**，成功率下跌 8.0 个百分点，大量由于语法合规但调度/语义存在暗病的问题在提交后爆发；

- 移除**技能演化模块**后，虽然短期内成功率仅小幅下滑 2 个百分点，但平均 Token 消耗骤增至 112.3K，系统的推理成本与冗余动作明显增加。

#### 线上生产部署与 A/B 测试

纸上得来终觉浅，真正的试金石是高并发、高严苛的生产环境。SiriusDeliver 部署于腾讯云 WeData 后，覆盖了微信支付、腾讯广告、安全风控等 6 个核心业务线，支持 3,600 名月活工程师，在两个月内处理了 18,240 次交付会话。

在全量统计中，系统取得了 **87.2% 的端到端交付成功率**，其中 **73.5% 的任务实现了完全无需人工修改的自主提交**。

为期一个月的严格线上 A/B 测试进一步量化了该系统带来的工程效能变革：

```

A/B 测试核心指标对比（人工基线 vs SiriusDeliver）：

- 交付总耗时中位数：228 分钟  ──>  23 分钟  （缩减 89.9%）

- 工程师实际精力投入： 95 分钟  ──>  11 分钟  （缩减 88.4%）

- 首个工件交付时长：  44 分钟  ──>  2.6 分钟 （缩减 94.1%）

- 人工强制介入比例： 100%     ──>  21%      （大幅下降）

- 最终交付成功率：   92.4%    ──>  91.8%    （保持同一水平）

```

测试数据清晰表明，SiriusDeliver 在几乎没有牺牲最终交付质量的前提下，将数仓任务的生产周期缩短了一个数量级。过去需要工程师耗费大半天时间反复沟通需求、翻查血缘、手动配 DAG 并守在控制台看日志的流程，如今被压缩到了半小时以内的自动化流水线中。

### 工程师视角的范式演进：从写代码到审工件

从 SiriusDeliver 的架构实践中，可以提炼出构建企业级垂直领域 Agent 的三条关键认知：

第一，**摆脱“Prompt 即系统”的轻量化幻想**。在涉及基础设施和生产配置的领域，纯基于 Prompt Engineering 的通用 Agent 根本无法跨越工业落地的可靠性红线。必须将领域特有的状态机、工件契约、静态分析规则深度嵌入到 Agent 的规划与执行回路上。

第二，**确定性工具与生成式 LLM 必须严格解耦又深度咬合**。SiriusDeliver 的聪明之处在于，它没有试图让大模型去肉眼判断字段是否存在冲突、DAG 拓扑是否有环，而是让确定性静态检查工具和元数据引擎充当大模型的“严苛考官”。大模型负责高灵活度的意图翻译与错误溯源，底层平台工具负责不可逾越的规则底线。

第三，**经验闭环是降低落地成本的核心抓手**。大模型落地的推理成本与企业效益是一场拉锯战。通过对真实执行轨迹的双向挖掘，把复杂的逻辑下沉固定为紧凑的工具技能，既能降低对大模型上下文窗口的滥用，也能规避系统随着生产环境演化而日趋退化的风险。

SiriusDeliver 在腾讯复杂数仓场景下的长期平稳运行表明，大模型在企业内部的真正价值，绝不仅仅是辅助工程师写完那 25% 的业务代码，而是彻底重构那被繁杂、断裂且重复的平台运维所占据的 75% 的交付全流程。
