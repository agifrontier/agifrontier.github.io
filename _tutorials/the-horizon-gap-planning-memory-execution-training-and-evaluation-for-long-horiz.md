---
layout: default
title: "长程Agent最新综述！六大生命周期+1547篇论文总结"
description: "前沿大语言模型已经能够在单次前向传播中解决极其复杂的推理问题，其表现足以媲美数年前的独立学术成果。然而，一旦将同一个模型置入智能体（Agent）循环中，要求它独立处理耗时数小时的软件工程或数据分析任务，系统往往会暴露出截然不同的脆弱性：遗忘数步之前的关键决策、在代码只修了一半时便草草宣称任务完成，或是在多轮工。"
arxiv_id: "2608.06663"
published_at: "2026-09-04T13:15:08.122237+08:00"
topics:
  - "知识系统"
  - "AI Agent"
tags:
  - "credit assignment"
  - "execution"
  - "horizon gap"
  - "long-context"
  - "long-horizon"
  - "long-term memory"
related_tutorials:
  - "agentic-memory-learning-unified-long-term-and-short-term-memory-management-for-l"
  - "dynamic-affective-memory-management-for-personalized-llm-agents"
  - "forgetful-but-faithful-a-cognitive-memory-architecture-and-benchmark-for-privacy"
  - "simplemem-efficient-lifelong-memory-for-llm-agents"
---

<p class="paper-original-title" lang="en">The Horizon Gap: Planning, Memory, Execution, Training, and Evaluation for Long-Horizon LLM Agents</p>

<img src="/images/2608.06663v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

前沿大语言模型已经能够在单次前向传播中解决极其复杂的推理问题，其表现足以媲美数年前的独立学术成果。然而，一旦将同一个模型置入智能体（Agent）循环中，要求它独立处理耗时数小时的软件工程或数据分析任务，系统往往会暴露出截然不同的脆弱性：遗忘数步之前的关键决策、在代码只修了一半时便草草宣称任务完成，或是在多轮工具调用中不知不觉偏离最初的目标。

> ArXiv URL：https://arxiv.org/abs/2608.06663v1

这种“单步推理能力极强”与“多步长程任务难以可靠交付”之间的鸿沟，被来自 AlphaAvatar 与 DeepGrounding 等机构的研究者定义为**视野差距**（Horizon Gap）。围绕这一核心瓶颈，研究者对 2024 至 2026 年间的 1,547 篇 arXiv 论文进行了系统性梳理，在剔除 26.8% 的无关文献后，构建出一套覆盖长程任务完整生命周期的六维分类框架。

这篇长达近百页的综述给出了一个贯穿全领域的清晰判断：随着任务视野的拉长，仅凭最终成败的**结果级信号**（Outcome-only signals）会迅速失去信息量，无论是训练阶段的过程奖励模型、信用分配机制，还是评测阶段的轨迹诊断，整个学术界的本质应对方案都是在长链条中强行“制造”更密集的**过程级信号**（Process-level signals）。

### 厘清三大概念：长任务、长上下文与长程记忆

在现存的大量讨论中，“长视野”（Long-horizon）、“长上下文”（Long-context）和“长程记忆”（Long-term memory）经常被混为一谈。这种术语层面的模糊直接导致了技术选型的混乱，很多研究试图用扩大上下文窗口去解决本属于状态管理的工程问题。该综述在开篇便对这三者做了正交切分：

1. **长视野**（Long-horizon）：属于**任务属性**，指完成目标所必须执行的交互或推理步数。它关注的是决策链路的深度与累积误差。

2. **长上下文**（Long-context）：属于**模型属性**，指 Transformer 注意力机制单次前向传播能有效吞吐的 Token 数量上限。

3. **长程记忆**（Long-term memory）：属于**系统属性**，指信息能否跨越单次推理步甚至跨越不同的运行会话（Sessions）稳定持久化。

三者在逻辑上完全独立。一个系统完全可以通过激进的摘要与外部工具，在极短的上下文窗口内驱动包含上百步调用的长视野任务；反之，一个拥有数百万 Token 上下文能力的模型，如果每次运行都从空白状态开始，系统层面依然不具备任何长程记忆。

<img src="/images/2608.06663v1/figure1_taxonomy_grid.webp" alt="图1：六大生命周期类别与视野承载位置交叉分类矩阵" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了看清技术方案的真实定位，该研究引入了双轴框架。第一轴沿着任务的**生命周期**展开，划分为规划与分解、记忆与上下文管理、执行控制与恢复、长程训练、评测度量、理论极限与安全六大板块；第二轴则是**视野由何处承载**（Where the horizon is carried）：

- **上下文内**（Within-context）：单次交互窗口即可容纳全部历史，瓶颈主要在于注意力衰减与利用率下降；

- **任务内跨上下文**（Within-task-beyond-context）：任务步数超过窗口上限，依赖 Harness 外壳（外部存储、滑动窗口、上下文压缩、子 Agent 交接）维持单次任务的连贯；

- **跨任务持久化**（Cross-task-persistent）：信息和技能跨越独立会话固化下来（如参数微调、静态经验库），依赖持续积累拓宽视野。

<img src="/images/2608.06663v1/map_landscape.webp" alt="图2：1,547 篇文献的语义密度图谱" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 规划与记忆：从预先承诺到外部工程

规划是长程任务被拆解为可操作步骤的起点。在对 162 篇规划分解类文献的分析中，研究者指出，规划策略的选择本质上取决于**环境的不确定性**与**动作空间的大小**，而非底层模型的智能程度。

<img src="/images/2608.06663v1/figure3_planning_spectrum.webp" alt="图3：规划策略光谱与环境不确定性、动作空间的关系" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在确定性高、反馈延迟低的小型环境中，经典的“先规划后执行”（Plan-then-execute）非常高效且具备可解释性；但随着任务走向开放环境，“在第 $t=0$ 步制定的计划能够撑到第 $t=k$ 步”的假设彻底破裂。早期做出的全局承诺不仅无法适应意外，反而会成为后续动作的桎梏。因此，研究重心已显著转向交替式规划（Interleaved planning）与显式搜索（Search / World Models）。交替式机制通过牺牲全局一致性换取局部鲁棒性，而搜索机制则通过增加推理期算力消耗，在不确定的环境中动态剪枝。

如果规划决定了下一步“做什么”，记忆机制则决定了这一决策依据“什么信息”做出。在这一维度上，技术方案呈现出清晰的保真度与持久性权衡（Persistence-Fidelity Trade-off）。

<img src="/images/2608.06663v1/figure4_memory_hierarchy.webp" alt="图4：记忆三层架构的保真度与持久性权衡" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

直接放在 Prompt 中的上下文保真度最高，但容量受制于物理窗口和注意力成本；固化在模型权重中的知识持久性最高，但更新成本高昂且几乎不可审计；位于中间的外部向量库或图存储，则成为当前工业界与学术界的最爱。在收录的记忆类论文中，外部存储方案（294 篇）的数量几乎是上下文管理方案（103 篇）的三倍。这表明学术界普遍倾向于将记忆问题转化为存储与检索工程，绕开昂贵的长上下文微调。然而外部存储并非银弹，随着运行时间延长，检索噪声、信息干扰和上下文陈旧化迅速显现，检索本身成了引入错误的新环节。

### 执行与训练：编排膨胀与过程信号的重构

在漫长的任务执行期，决定系统生死的往往不是模型单步输出有多精妙，而是外部外壳（Harness）能否感知错误并及时止损。执行控制类文献占据了整个语料库最大的份额（584 篇），但内部结构严重失衡：多 Agent 编排（Orchestration，338 篇）与恢复机制（Recovery / Self-correction，245 篇）占据了压倒性多数。

研究指出，当前社区投入了大量精力去“横向扩容”——通过堆叠更多的智能体角色、设立更繁琐的协同流程来拆解长任务；而在“纵向强化”单 Agent 纠错能力上的探索相对滞后。更关键的诊断研究表明，**内在自我纠错**（Intrinsic self-correction）在缺乏外部可靠反馈环境（如编译器、单元测试、断言验证）的前提下，极易退化为无意义的确认偏差，甚至越改越错。多 Agent 之间传递错误状态的现象，使得许多宣称的编排优势在严格消融实验中难以站稳脚跟。

在模型训练层面，长任务彻底击碎了强化学习传统的结果奖励范式。如果一个任务需要连续操作数百步才能得知最终是成功还是失败，那么单一标量奖励在反向传播时将遭遇极度的信息稀疏与方差爆炸。

为此，近年研究集中于两个方向：

- **信用分配**（Credit Assignment，130 篇）：在最终奖励固定的情况下，利用反事实基线或轨迹分析，推断究竟是哪一步操作导致了全局崩溃；

- **过程级监督**（Process Supervision，37 篇）：引入过程奖励模型（PRM），对推理和调用的每一个中间状态打分，将稀疏奖励转化为稠密信号。

尽管过程奖励能显著改善长程策略的学习动态，但它引入了新的风险：如果用于训练过程奖励模型的启发式规则本身带有偏差，模型就会学会“过程看起来极为合理、最终彻底偏离目标”的作弊行为。

### 评测危机与基础理论的缺位

长任务领域面临的最大隐忧在于**评测基准的失真**。评测类文献（114 篇）中，针对现有 Benchmark 的诊断与打假占据了很大比重。以软件工程基准 SWE-bench 为代表，多项独立复现研究揭示，大量看似优秀的解决率背后，充斥着测试用例过弱、无意识的数据污染以及模型利用偶然机制“碰巧通过”的假象。

<img src="/images/2608.06663v1/figure5_benchmark_duration.webp" alt="图5：主流长程任务基准的人类完成耗时分布（对数坐标）" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了摆脱简单的 Pass/Fail 评分掩盖的执行细节，评估体系正在向“自主运行时间”（Autonomy Time）迁移——即以人类工程师完成该任务所需的基准时间（从数分钟到数小时不等）作为度量轴，观察 Agent 在不同时间跨度下的可靠性衰减。此时，评估的核心单位不再是最终产物，而是整条**执行轨迹**（Trajectory）。

然而，关于长任务为何会崩溃，基础理论层面依然缺乏统一的数学描述。在“理论与安全”这一收录较少的领域（103 篇），现有观察呈现出分散的局部经验：

- 系统的一致性丧失往往不是平滑线性的，而是呈现双峰分布：系统在很长一段时间内维持高度理性，但在特定阈值后突发崩塌；

- 在崩溃真正发生前，轨迹中常常会出现细微的预警信号（如自发增加的不必要检索、冗余的工具查询），但现有的监督机制往往直接放行；

- 无人值守时间越长，对抗性提示注入与意外越轨的累积风险呈指数级放大，使得传统的单步对齐（Alignment）技术在长程自主运行下全面失效。

### 趋势研判与三大未决度量难题

纵观近三年的论文增长趋势，长任务 Agent 的研究重心发生了明显的结构性转移。

<img src="/images/2608.06663v1/growth_timeline.webp" alt="图6：2024-2026年各研究类别季度发表量与占比演进" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在 2024 至 2025 年间，执行与编排类论文牢牢占据半壁江山（占比 41%-54%）。这是由于调用多个模型搭建流水线的门槛最低、出成果最快。但从 2026 年第一季度开始，记忆机制的论文份额迅速飙升并实现反超，基础理论与极限探讨的发文占比在 2026 年达到了 77%。这种演进轨迹清晰地反映了技术落地过程中的阵痛：研究者早期专注于“让 Agent 跑起来并执行多个动作”，而当 Agent 真正被部署到复杂现实任务中时，因记忆混乱、状态污染导致的系统停摆才成为最刺眼的矛盾，从而倒逼研究注意力向深层机制回流。

结合全景文献分析，综述最终指出了三个阻碍该领域建立可预测科学体系的核心度量问题：

1. **能力归因的解耦难题**：当一个长任务系统跑通了复杂任务，这份长程能力究竟有多少来自底层大模型本身的单步推理与遵循指令素质，又有多少来自包裹在外的工程 Harness（如重试机制、硬编码规则、状态机切分）？目前的基准缺乏有效消融手段将二者严格剥离。

2. **训练与评测过程信号的同相关偏差**：由于终端结果过于稀疏，学术界在训练端用过程奖励模型引导模型，在评测端同样用过程诊断逻辑衡量好坏。若两端基于相同的演进假设，评估体系便会系统性放过那些符合“表面进度”却南辕北辙的严重故障。

3. **长程可靠性理论的缺失**：不同于预训练模型具有清晰的缩放定律（Scaling Laws），长任务自主运行下的错误累积尚未建立起可靠的预测模型。我们仍无法仅凭模型的单步错误率，直接推算出它在自主运行 4 小时后的无故障概率。

长视野智能体的突破，远不是把上下文长度推向无限大那么简单。它标志着大语言模型的研究范式，正在从关注静态文本生成的“认知快照”，彻底转向应对环境交互、状态漂移与自我恢复的“动态控制论”。在这个跨度数小时乃至数天的未知探索空间里，制造并度量出稠密、真实的中间过程信号，仍是横亘在通往通用自主智能体道路上最核心的技术高墙。
