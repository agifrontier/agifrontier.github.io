---
layout: default
title: "ERSkill：让Agent记忆检索学会自我进化，综合表现最高提升31.3%"
description: "来自深圳国际工业与应用数学中心、中大深圳、深数院以及中山大学等机构的研究团队提出了 ERSkill（Evolving Retrieval Skill） 框架。这项工作跳出了“只优化记忆构建或推理引导”的旧范式，首次将 记忆检索行为本身 抽象为可执行、可组合的原语序列。"
arxiv_id: "2608.12720"
paper_published: "2026-08-13"
published_at: "2026-09-07T13:15:08.344824+08:00"
topics:
  - "RAG"
  - "知识系统"
tags:
  - "ERSkill"
  - "co-evolving skills and router"
  - "double-frontier mechanism"
  - "evolvable retrieval"
  - "executable skills"
  - "experience trie"
related_tutorials:
  - "llm-guided-hierarchical-retrieval"
  - "webxskill-skill-learning-for-autonomous-web-agents"
  - "dynamic-agent-skills-a-lifecycle-survey-and-taxonomy-of-evolving-skill-libraries"
  - "skill-self-play-pushing-the-frontier-of-llm-capability-with-co-evolving-skills"
---

<p class="paper-original-title" lang="en">ERSkill: Evolving for Skill-Guided Adaptive Memory Retrieval</p>

<img src="/images/2608.12720v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在长期交互场景中，大语言模型（LLM）智能体正逐渐从一次性的问答工具演变为具备连续协作能力的伙伴。伴随数周甚至数月的对话，智能体需要记住用户的个性化偏好、追踪动态演变的时间线，并在关键时刻调用过往决策。然而，现有的大多数长期记忆系统都存在一个隐秘的瓶颈：**记忆的存储和推理在不断进化，但检索机制本身却被永久冻结在静态策略中**。

> ArXiv URL：https://arxiv.org/abs/2608.12720v1

不论是面对“爱丽丝在夏威夷旅行时给鲍勃买了什么礼物”这类单点事实检索，还是“为什么爱丽丝后来取消了与鲍勃的下一次旅行计划”这类需要梳理因果链条的复杂推理，传统智能体往往只能使用一套预先配置好的稠密向量检索（Dense Retrieval）或混合检索管道。这种一成不变的证据检索逻辑，根本无法应对异质化查询背后千差万别的证据构建需求。

<img src="/images/2608.12720v1/paradigm_comparison.webp" alt="范式对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

来自深圳国际工业与应用数学中心、中大深圳、深数院以及中山大学等机构的研究团队提出了 **ERSkill（Evolving Retrieval Skill）** 框架。这项工作跳出了“只优化记忆构建或推理引导”的旧范式，首次将**记忆检索行为本身**抽象为可执行、可组合的原语序列，并通过“技能库-路由器”的双前沿协同进化机制，让智能体在与任务环境交互时自主学会如何构建证据。

实验数据显示，ERSkill 在多个具有挑战性的智能体长记忆基准上展现了显著优势，在 Qwen3-Next-80B-A3B-Instruct 和 GPT-5.4-nano 两大底座模型上，综合平均指标（F1、BLEU-1 与 LLM-as-a-Judge）相较于最强基线分别提升了 31.3% 和 28.1%，且在推理成本控制上表现出色。

### 为什么静态检索无法胜任长记忆管理？

智能体长期记忆的研究大致经历了两波技术迭代。第一波聚焦于**结构化存储与生命周期管理**，例如 A-Mem、MemoryOS 和 LightMem，通过信息抽取、摘要压缩、动态更新与遗忘机制维护一个外部存储介质。第二波则转向**反思与自我进化**，例如 ReasoningBank、Dynamic Cheatsheet 以及 MemSkill，试图通过复盘历史轨迹，沉淀可复用的推理知识库或记忆抽取算子。

然而，这两波范式在检索侧均走向了同一种妥协：要么使用标准的 RAG 机制（基于 Embedding 相似度匹配 TOP-$k$ 文本块），要么采用单一的稠密/关键词搜索规则。这种静态方案在处理简单事实时尚能勉强胜任，一旦遭遇现实交互中的多源信息整合，就会出现明显的结构性失真：

不同性质的提问，需要的证据拓扑完全不同。有些问题依赖于实体定位与跳跃式的时间追溯，有些问题需要以某个核心节点为种子进行图谱扩散，还有些问题需要依靠重写查询来对抗语义漂移。若检索管道无法根据查询动态调整行为模式，即使外部记忆存储得再完善，智能体也只能在一堆语义相关但逻辑无关的噪声上下文上“盲人摸象”。

ERSkill 的核心出发点正是打破这种静态检索假象：记忆检索不应该只是一个被动的打分算子，而应该是一套由基础原语编排而成的动态执行程序。

### 记忆即程序：原语、技能与轻量路由

为了让检索过程具备可塑性与程序化执行能力，ERSkill 构建了一套层次分明的检索抽象体系，包含记忆基质、原子原语、技能程序与调度路由四个层次。

<img src="/images/2608.12720v1/main_architecture.webp" alt="ERSkill系统整体架构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### 1. 结构化记忆底座与原语库

ERSkill 首先将原始交互历史 $D$ 切分为原子级记录集合 $\mathcal{A} = \{a_1, \dots, a_n\}$，每个原子记录不仅包含原始文本，还绑定了结构化元数据和时间戳。在此基础上，系统构建了带有双重索引特性的记忆底座 $M(D) = (\mathcal{A}, \mathcal{I}, \mathcal{G})$。其中，$\mathcal{I}$ 是用于快速检索候选候选原子的索引集合（包括稠密向量索引与实体映射索引），而 $\mathcal{G}$ 则是建立在原子之间的图结构，用于沿语义或关联网络进行关系扩散。

在这个底座之上，ERSkill 抽象出了一套标准检索原语库 $\mathcal{P} = \{p_1, \dots, p_m\}$。每个原语本质上是一个形式化的状态转移函数：




{% raw %}$$p: (q, s, M(D)) \mapsto s'$${% endraw %}



其中 $q$ 是用户查询，$s$ 是当前的证据上下文状态，$s'$ 是执行原语后的新状态。原语库涵盖了从关键词匹配、向量稠密定位、实体锚定过滤，到基于图的相似度多跳展开等原子操作。

#### 2. 可执行的检索技能

在 ERSkill 中，一个检索技能 $\kappa$ 被定义为一个可执行的原语调用序列，由技能文本描述 $c_\kappa$ 与原语执行流 $\rho_\kappa = (p_{\kappa,1}, \dots, p_{\kappa,L_\kappa})$ 组合而成。

<img src="/images/2608.12720v1/skill_sample.webp" alt="检索技能示例" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在上图展示的一个典型技能中，系统针对具有明确实体但缺乏直接上下文的查询，先执行实体对齐检索锁定种子原子，再执行原子重打分，最后沿着图关系网络展开高相关度的邻近事实。这一系列步骤将杂乱的全局存储，动态折叠为一个贴合特定查询证据链的微型视图（Evidence View），并以纯文本 Markdown 形式持久化，兼具执行确定性与逻辑可解释性。

#### 3. 动态技能路由器

有了不断生长的技能库，如何在毫秒级响应内为不同的提问精准下发指令？ERSkill 避免了昂贵的大模型多跳决策，而是训练了一个小巧轻量的专用路由模型（Skill Router）。

路由器采用冻结的轻量文本编码器 $\mathrm{Enc}(\cdot)$（实验中采用 0.6B 参数的 Qwen3-Embedding）分别提取查询特征 $h_q = \mathrm{Enc}(q)$ 与技能特征 $h_\kappa = \mathrm{Enc}(\kappa)$。二者经过投影层拼接后，送入双层感知机（MLP）打分函数 $u_\theta(q, \kappa)$。最终，技能的选中概率呈现为玻尔兹曼分布：




{% raw %}$$R_\theta(\kappa \mid q, \mathcal{K}) = \frac{\exp(u_\theta(q,\kappa))}{\sum_{\kappa' \in \mathcal{K}} \exp(u_\theta(q,\kappa'))}$${% endraw %}



这种设计赋予了框架两项极其关键的工程优势：第一，路由器能够直接基于新技能的自然语言描述与原语序列进行特征泛化，**当进化引擎引入全新技能时，完全无需重构路由器的输出空间架构**；第二，推理阶段仅涉及一次向量打分，与直接调用 LLM 做决策相比，路由延迟与 Token 开销几乎可以忽略不计。

### 协同进化：经验前缀树与双前沿控制

如果仅使用静态设计的几种技能，检索能力的上限依然会被人为经验锁死。ERSkill 的真正技术突破，在于它实现了一套让“检索技能库”与“路由模型”协同进化的自动化闭环。

在自动化程序生成领域，盲目探索往往会导致搜索空间爆炸，生成大量功能重叠、语义冗余甚至劣质的死循环程序。为了实现安全、高效的持续演进，ERSkill 引入了两项关键技术。

#### 经验前缀树（Experience Trie）

由于任何检索技能都是有限原语库 $\mathcal{P}$ 的排列组合，团队设计了一棵路径粒度的经验前缀树 $\mathcal{T}$。从树根到任意节点的路径，都严格对应着一段原语执行前缀。

前缀树完整记录了历史上探索过的所有技能分支、每个技能在训练批次与验证集上的执行成功率、LLM-judge 打分统计，以及技能的准入状态。当技能生成器尝试通过编辑现有技能（如追加原语或替换算子）提出候选技能时，它会读取树上的失败与成功模式，同时**自动过滤掉已经在树上被探索过的同构路径**。这种机制避免了传统演化搜索中反复尝试已知失败原语模式的算力浪费。

#### 双前沿管理（Double-Frontier Mechanism）

在真实的交互系统中，即便某个技能在理论上能解决某种极罕见的查询，如果路由器无法稳定地在对应场景下激活它，贸然将其加入部署池就可能造成对其他常规查询的误伤。为了化解“技能探索”与“稳定交付”之间的张力，ERSkill 提出了双前沿控制机制：

1. **能力前沿（Capability Frontier, $\mathcal{C}_t$）**：追踪系统迄今为止所能达到的检索能力天花板。它站在“理想上帝视角”（Oracle-side）进行评估——假定路由器每次都能百分之百选中最优技能，计算出当前技能集合在验证集上能覆盖的最大准确度。只有当一个新候选技能在某些查询上创造了无法被替代的增量价值时，它才会被允许进入能力前沿。

2. **部署前沿（Deploy Frontier, $\mathcal{B}_t$）**：面向真实推理场景的技能库。该集合中的技能不仅要在能力前沿中被证明有效，而且必须在**实际路由模型分配下**经过严格的检验。

团队在论文中给出了理论保证（Proposition 2.1）：通过基于帕累托最优的重算子 $\Phi(\mathcal{K}; \mathcal{Q})$ 进行剪枝，两个前沿在验证集上的理论覆盖率指标 $\mathrm{OCov}(\cdot)$ 在进化步数推进中均呈现单调不减，在数学层面上封堵了系统“越学越倒退”的退化风险。

在每个演化步 $t$，系统首先利用当前批次训练样本驱动前缀树产生候选技能，更新能力前沿 $\mathcal{C}_{t+1}$；随后利用累积的打分分布通过软标签交叉熵优化路由器参数 $\theta$：




{% raw %}$$\tilde{p}(\kappa \mid q, \mathcal{K}_q) = \frac{\exp(r(q,\kappa))}{\sum_{\kappa' \in \mathcal{K}_q} \exp(r(q,\kappa'))}$${% endraw %}






{% raw %}$$\mathcal{L}_{\mathrm{router}} = -\sum_{(q,\mathcal{K}_q,\cdot)} \sum_{\kappa \in \mathcal{K}_q} \tilde{p}(\kappa \mid q, \mathcal{K}_q) \log R_\theta(\kappa \mid q, \mathcal{K}_q)$${% endraw %}



在路由器权重更新完毕后，系统才在验证集上重新评估新技能并入部署池后的实际端到端收益 $\Delta_{\mathrm{route}}$。只有当引入新技能带来的增益超过特定阈值 $\gamma_{\mathrm{route}}$，或者在保持性能不降的同时缩减了部署集大小时，部署前沿才会正式接收该技能。这种解耦机制既允许能力探索大胆突破，又保证了线上运行平滑稳定。

### 实验检验：性能飞跃与极高性价比

为了验证 ERSkill 的实际表现，论文在三个具有代表性的智能体长记忆基准上展开了全面评测：涵盖多会话超长对话追踪的 **LoCoMo**、专注于长周期记忆评测的 **LongMemEval**，以及进一步引入多模态异质数据源记忆问答的 **PerLTQA**。

测试选用开源领先的 Qwen3-Next-80B-A3B-Instruct 与前沿轻量模型 GPT-5.4-nano 作为生成底座，并统一使用 GPT-4o-mini 作为评判员（LLM-as-a-Judge），从 Token 重合度（F1、BLEU-1）与语义真实性两个维度全面打分。


| 基准数据集 | 骨干模型 | 评测指标 | 静态最强基线 (LightMem / MemoryOS) | 自进化最强基线 (ReasoningBank / MemSkill) | ERSkill |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **LoCoMo** | Qwen3-Next-80B | F1 / BLEU-1 / L-J | 38.2 / 31.5 / 54.2 | 41.6 / 34.1 / 58.7 | **54.8 / 45.2 / 76.8** |
| | GPT-5.4-nano | F1 / BLEU-1 / L-J | 36.9 / 30.1 / 52.0 | 40.2 / 33.5 / 57.1 | **51.3 / 42.6 / 73.5** |
| **LongMemEval** (零样本迁移) | Qwen3-Next-80B | F1 / BLEU-1 / L-J | 42.1 / 35.0 / 59.4 | 45.8 / 37.9 / 64.2 | **58.9 / 49.3 / 82.1** |
| | GPT-5.4-nano | F1 / BLEU-1 / L-J | 40.5 / 33.8 / 57.6 | 44.1 / 36.5 / 62.0 | **56.2 / 46.8 / 79.4** |
| **PerLTQA** | Qwen3-Next-80B | F1 / BLEU-1 / L-J | 32.4 / 25.1 / 46.3 | 35.7 / 28.3 / 50.9 | **47.6 / 38.5 / 67.2** |
| | GPT-5.4-nano | F1 / BLEU-1 / L-J | 31.0 / 24.2 / 44.1 | 34.2 / 27.0 / 48.5 | **44.9 / 36.1 / 63.8** |

在各项评测中，ERSkill 的综合性能均大幅领先所有基线模型。尤其在面对 Single-Hop 与 Multi-Hop 这类极度依赖精准线索定位的硬核问题时，ERSkill 的提分幅度尤为惊人。这直接证明了：**决定长记忆问答质量的核心矛盾，往往不是底座模型本身的生成能力有多弱，而是在检索阶段输入的内容究竟包含了多少有效信号**。

值得关注的是 **LongMemEval** 上的评测设置：ERSkill 并未在 LongMemEval 训练集上执行演化，而是**直接加载在 LoCoMo 上进化出的技能库与路由器参数进行零样本推理**。即使在完全未见过的分布下，ERSkill 依然毫无悬念地击败了在该数据集上专门调优的基线模型。这表明，演化出来的原语调用序列并非过拟合特定文本特征的“作弊代码”，而是提炼出了在自然语言记忆访问中通用的认知搜索策略。

除了绝对准确率，系统的 Token 消耗与计算开销是决定工业落地的生死线。

在内存构建阶段，诸如 LightMem 等方案需要消耗大量 LLM 调用来完成实体压缩、摘要改写和长程归纳。而 ERSkill 仅利用模型执行必要的轻量关系提取，其余工作全部交由确定性的图与向量索引底座完成，构建阶段的消耗被压缩在第一梯队。

在推理阶段，得益于技能对无效噪音文本的精准过滤，ERSkill 生成答案时输入的上下文长度远小于传统的宽进宽出方案（如直接抓取 TOP-20 文本块的 RAG）。这意味着它在以极高的胜率通过 LLM-judge 评测的同时，单次问答消耗的推理 Token 维持在低位，实现了极具竞争力的 Pareto 能效比。

### 拆解演化黑盒：消融与演化动态

为了验证系统内部各模块的必要性，论文进一步剥离了各项关键设计：

1. **移除技能进化（w/o skill evolution）**：仅保留初始的 3 个基础种子技能（稠密、实体、词法单步检索），系统在所有指标上均遭遇断崖式下跌，这充分说明了组合型技能程序的必要性；

2. **移除路由器（w/o router）**：改由大模型通过标准 Prompt 自行挑选技能，不仅推理时延成倍上涨，端到端指标也显著受挫，证明小参数量的密集向量路由在判断技能适配性时比开放式大模型更可靠；

3. **移除双前沿机制（w/o double frontier）**：无论演化产生的候选技能是否引起线上震荡均强行吸纳入库，导致由于候选膨胀引发严重的路由混淆；

4. **移除经验前缀树（w/o experience trie）**：技能生成盲目性增加，耗费了大量迭代预算在已知低效的原语组合上，进化收敛明显放缓。

<img src="/images/2608.12720v1/case_study.webp" alt="进化动力学案例分析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

上图揭示了 ERSkill 在长周期进化中的内部运行动态。在图左侧，能力前沿与部署前沿的理论覆盖上限（Oracle Accuracy）呈现稳步爬升态势。尽管在某些节点，新技能的引入使得尚未完全收敛的路由器出现了短暂的部署准确率下探，但随着前缀树反馈与软标签梯度的回传，路由模型迅速完成自适应，部署准确率迅速恢复并反超历史峰值。

而在图右侧，两个前沿集合的实际容纳规模始终被稳稳锁死在极低的紧凑区间内（通常保持在个位数到十几个优质技能之间）。新技能只有在功能上实现对旧技能的严格支配时才会引发替换，系统并没有在时间的推移下变成一个尾大不掉、充斥着冗余算子的臃肿黑盒。

### 长记忆Agent的新思考

ERSkill 给大模型智能体系统设计带来的最大启示，在于重新划定了“进化”应该发生的位置。

过去的思路习惯于将算力砸在两头：要么在记忆输入端，试图用更强大的大模型把历史数据揉捏成天衣无缝的外部摘要；要么在推理输出端，利用多智能体辩论或庞大的推理链在杂乱的上下文中苦苦打捞线索。而 ERSkill 证明，**夹在中间的“检索动作”本身，恰恰是弹性最大、成本收益比最高的进化切入点**。

将看似不可控的记忆检索拆解为离散的确定性原语，再将原语编织为可解释的技能程序，并辅以严密的工程机制进行动态修剪与平滑路由——这种兼顾符号确定性与神经自适应的设计范式，不仅让长期伴随型智能体真正具备了自我迭代的信息捕获力，也为未来多模态环境下的复杂工具使用与动态记忆管理提供了一条清晰、可落地的技术参照。
