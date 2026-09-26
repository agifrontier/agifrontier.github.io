---
layout: default
title: "MemArbiter：破解长程智能体“记住了却用不对”，成功率最高提升25.4个百分点"
description: "为了解决这一问题，研究团队提出了 MemArbiter 。它不再使用传统“混作一团”的扁平记忆流，而是在决策时刻引入动态仲裁机制：将交互历史拆解为原子记忆，按五种功能明确的 Memory Bank（记忆库）进行分流管理，并借助“焦点-环境”双粒度表达与时序门控。"
arxiv_id: "2608.02113"
paper_published: "2026-08-03"
published_at: "2026-09-26T13:15:07.411720+08:00"
topics:
  - "知识系统"
  - "AI Agent"
tags:
  - "知识系统"
  - "AI Agent"
  - "AI论文解读"
related_tutorials:
  - "long-horizon-embodied-decision-making-via-multimodal-memory-compression"
  - "coevokg-co-evolving-knowledge-graphs-with-self-evolving-search-agents"
  - "attrimem-attribution-guided-process-feedback-for-agent-memory-learning"
  - "v-mem-modality-routed-retrieval-for-long-term-multimodal-agentic-memory"
seo_title: "MemArbiter：破解长程智能体“记住了却用不对”，成功率最高提升25.4个百分点"
---

<p class="paper-original-title" lang="en">MemArbiter: Decision-Time Memory Arbitration for Long-Horizon LLM Agents</p>

在长时间运行的 LLM Agent（大模型智能体）开发中，一个令人头疼的现象屡见不鲜：当任务步数被拉长到几十步甚至上百步时，即使把之前探索过的环境信息、踩坑经验一股脑塞进 Context，或者通过向量数据库完整检出了相关片段，智能体依然会在关键节点“视而不见”，不断重复已经失败的动作，甚至陷入漫无目的的循环游荡。

> ArXiv URL：https://arxiv.org/abs/2608.02113

许多开发者下意识地将这种失败归咎于模型底层的长文本遗忘或检索算法不够准。但最新研究指出了一个更根本的盲区：信息在记忆库中**“能够被访问（Accessible）”**，绝不等于它在决策时刻**“能够发挥作用（Action-guiding）”**。论文将这种即使拥有相关信息却无法正确指导动作的现象，正式定义为**“记忆-动作鸿沟”（Memory-Action Gap）**。

<img src="/images/2608.02113/preliminary_new_hyt.webp" alt="记忆-动作鸿沟示意图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了解决这一问题，研究团队提出了 **MemArbiter**。它不再使用传统“混作一团”的扁平记忆流，而是在决策时刻引入动态仲裁机制：将交互历史拆解为原子记忆，按五种功能明确的 Memory Bank（记忆库）进行分流管理，并借助“焦点-环境”双粒度表达与时序门控，精准控制关键信息在有限 Prompt 预算下的显式呈现。在具身交互基准 ALFWorld 的评测中，MemArbiter 配合开源大模型在统一 Token 预算下取得了高达 92.5% 的成功率，比最强基线提升了 25.4 个百分点，彻底打破了长程任务中常见的死循环魔咒。

### 记住了，为什么还是做不对？

以往针对 Agent 记忆的研究，大部分精力都花在如何“存下来”和“找出来”上，例如外挂向量检索、维护跨会话日志、或是构建分层摘要。然而，传统机制大多采取扁平化管理（Flat Memory）：把所有的历史观察、动作反馈和环境事实当作同质的文本块拼在一起，或者直接按时间衰减保留最近几步（Flat Recency），或者用 BM25 检索最相似的几行句子（Flat Retrieval）。

这种扁平做法恰恰忽视了长程决策中信息异构的本质。在执行一个复杂任务时，智能体需要的信息其实具有截然不同的决策功能与生命周期：

- **目标（Goal）**：指引全局方向，需要长期保持高度显著；

- **任务状态（Task State）**：反映当前进度的最新快照，需要持续被新状态覆盖；

- **硬性约束（Constraint）**：限制哪些动作不可行，可能持续生效数步；

- **情境经验（Episodic）**：记录过去的试错痕迹与后果，主要是为了防踩坑和错误恢复；

- **参考事实（Reference）**：探索过程中发现的客观环境布局，仅在需要特定实体时才应被唤醒。

如果将包含上述所有要素的原始交互日志作为整块文本存入，系统就无法对其进行独立的更新、降级或排序。一旦遭遇扁平拼接，关键的“约束”或“刚失败过的教训”很容易被冗长的环境描述淹没。大语言模型在注意力分配受限的情况下，极易忽略那些真正影响下一步动作的关键事实，从而酿成 Memory-Action Gap。

### MemArbiter：给 Agent 打造功能感知的记忆仲裁器

MemArbiter 的核心逻辑在于：**不再把记忆当作被动的静态存储桶，而是在每一个决策步（Decision-Time）对其进行有针对性的功能仲裁与排版。**

<img src="/images/2608.02113/overview-hyt.webp" alt="MemArbiter架构图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整体架构作为一个外置模块无缝嵌入 ReAct 循环中，在每一步动作生成前执行五个严密的流水线操作：

#### 1. 原子拆解与分库写入

当智能体获得新的环境观察 $o_t$ 以及上一轮动作的结果 $(a_{t-1}, r_{t-1})$ 时，候选写入器（Candidate Writer $\mathcal{W}$）会结合全局目标、子目标与未决信息需求，将新的交互内容分解为独立的“原子记忆单元”，分别路由到五大功能记忆库（Goal、Task State、Constraint、Episodic、Reference）。

每个库拥有独立的更新策略 $\mathcal{U}_b$：Task State 自动用最新有效快照覆盖旧值；Constraint 会合并重复项并动态修订适用范围；Episodic 按时间顺序追加关键的“动作-结果”事件；Reference 则基于新证据更新互斥事实。

#### 2. 双粒度表达（Dual-Band Representation）

在上下文窗口受限时，若把所有记忆都以详尽细节塞进 Prompt，预算很快就会见底；但若完全删去，模型又会失去全局感知。

MemArbiter 巧妙地引入了正交的“功能库（Bank）”与“表达带（Band）”概念：

- **焦点表达（Focal）**：保留记忆单元的完整内容，提供最充沛的上下文，直接支撑具体动作的生成；

- **环境表达（Ambient）**：通过确定的模版提取骨干实体与关键关系，只占用极少 Token，作为低显式度的背景线索。

同一个记忆项可以在焦点态与环境态之间无缝切换，无需反复重新抽取或重新分类。

#### 3. 需求与相关性双层信号

在长程任务推进过程中，不同类型记忆的价值是不断波动的。MemArbiter 维护了两种互补的评估信号：

- **库级需求信号（$d_{b,t}$）**：在宏观上评估当前步对某一类记忆的紧迫度（例如刚遭遇失败时，对 Episodic 库的需求会激增）；

- **项级相关性信号（$\eta_{i,t}$）**：在微观上根据记忆项与当前子目标、实体匹配度打分，决定它在同类记忆中的排队位次。

#### 4. 时序呈现门控（Temporal Presentation Gate）

如果每一轮完全只看当下的瞬时打分，记忆就会出现剧烈的“频闪”现象——上一秒还是焦点，下一秒突然消失，导致决策不稳定。

时序门控把呈现控制建模为一个具备状态记忆的赋值过程：结合上一轮的呈现状态 $z_{i,t-1}$、当前相关度以及历史相关性轨迹，在“焦点（$F$）”、“环境（$A$）”和“隐藏（$H$）”三种状态间受控流转。同时，对于 Goal 和 Constraint 施加保护机制，除非被显式判定完成或失效，否则不会轻易被时间衰减抹去。

#### 5. 提示词装配（Prompt Assembly）

在固定 Token 预算约束下，装配器将门控输出的记忆内容组织成结构清晰的 Prompt 块：按功能分区排列，焦点项以完整形式排在前面，环境项以精炼摘要缀后，同粒度项按相关性排序，隐藏项彻底剔除。处理后的记忆块与当前观察拼装，交给底层模型生成下一步动作。

### 复杂长程任务上的突破性表现

论文在具身交互基准 ALFWorld 上进行了严格的对比测试。ALFWorld 包含大批多步骤的家庭日常任务（如寻找物品、加热、清洗、放置等），特别选用了未见过的测试集（eval_out_of_distribution split），要求模型在纯文本环境中通过交互自主完成闭环。

实验统一采用tiktoken的 o200k_base 编码，严格卡死每一步注入动作 Prompt 的记忆预算（分别测试 500 Token 和 750 Token），动作生成底层模型统一采用 Qwen3.6-27B-FP8。

<img src="/images/2608.02113/alfworld_sr_at_k.webp" alt="ALFWorld各步数成功率对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整体评测给出了极具说服力的数据：

在 500-Token 内存预算下，传统 Flat Recency（最近历史）的成功率仅为 59.70%，基于 BM25 的 Flat Retrieval 仅为 61.94%，而 MemArbiter 达到了 **82.84%**（提升 20.9 个百分点）；

当预算放宽至 750 Token 时，Flat Recency 提升到 67.16%，而 MemArbiter 更是冲到了 **92.54%**（大幅超越基线 25.4 个百分点）。

这种断层式领先在极其复杂的任务类型上表现得尤为夸张。在需要同时协调多个物体状态的 **Pick Two（拾取两件物品）** 任务中，由于历史轨迹极长、中间状态极多，Flat Recency 与 Flat Retrieval 在两个预算设定下全军覆没，**成功率均为 0%**；而 MemArbiter 在 500 和 750 Token 预算下分别拿下了 **58.82%** 和 **70.59%** 的高成功率。

从图 3 的累计步数成功率（$\mathrm{SR}@k$）曲线可以看出，扁平基线在 15-20 步之后就进入了平台期，即使给满 50 步也无法解决更多问题；而 MemArbiter 的曲线则随着交互步数的增加稳定攀升。这证实了 MemArbiter 的高成功率并不是靠“漫无目的地瞎试更多步”换来的，而是真正让智能体具备了在超长交互轨迹下有效利用前期探索信息的能力。

不仅如此，当动作生成模型切换为闭源旗舰 **GPT-5.4**（记忆抽取仍由 Qwen3.6 担任），MemArbiter 依然交出了 83.58% 的高分，稳稳领先两个基线 10 个百分点以上，证明了这套记忆仲裁机制的通用性。

### 为什么它能有效阻断“死循环”？

为了探究系统发挥作用的内在机理，研究者对错误应对行为展开了细致的归因诊断。

在智能体与环境的交互中，由于物理前置条件未满足（例如柜门没开就伸手拿东西），经常会遇到动作执行失败。在基线方案中，智能体面对执行报错往往反应迟钝：在 500 Token 设定下，Flat Retrieval 和 Flat Recency 在单步失败后，立即重试同一错误动作（Failed-action repetition）的比例高达 26.5% 和 24.9%，整段轨迹中出现相同“状态-动作”死循环（State-action recurrence）的概率更是超过了 20%。

MemArbiter 展现出了完全不同的应对模式：

- **单步恢复率（One-step recovery）**：遭遇执行失败后，下一步能立即给出合法动作的比例提高到了 **88.2%**（基线均在 74% 左右）；

- **失败动作重复率**：骤降到了 **9.7%**；

- **状态-动作循环率**：被压缩至 **6.9%**。

这一系列行为转变的根源，来自于其动态的**库级需求调度**。数据分析显示，在常规正常决策步中，Episodic（经验轨迹）占用的 Prompt Token 份额仅为 7.8%；但一旦某一步执行失败，在下一个决策步中，门控网络瞬间将 Episodic 的份额提升至 **23.1%**（在 750 Token 下甚至达到 28.6%）。系统迅速将“刚才失败的操作及其原因”推上焦点高位，强行引起 LLM 的注意力偏置，从而极高概率阻断了模型的下意识重复。

在消融实验中，如果将五大 Functional Bank 抹除、退化为单一存储池，最终的 $\mathrm{SR}@50$ 暴跌了 15.68 个百分点；如果去掉动态需求信号、采用固定比例切分，成功率更是直降 17.17 个百分点；而如果移除时序门控（Temporal Gate）、改用每步完全孤立的打分，长程表现也衰退了 8.96 个百分点。这三者相辅相成，缺一不可。

### 从单纯存储，走向决策时仲裁

MemArbiter 的实践为大模型时代的 Agent 记忆架构指出了一个清晰的新演进方向：**检索只是第一步，呈现才是胜负手。**

在过去很长一段时间里，社区把太多精力放在了“如何通过 RAG 召回更多历史片段”或者“如何借助超大 Context Window 吞下全部交互”上。然而 MemArbiter 的实验结果表明，在没有功能分类和显式仲裁的情况下，无脑堆砌上下文往往适得其反；相反，在极为克制的 500 到 750 Token 狭小预算下，仅仅通过理清记忆的功能角色、动态调配焦点粒度与展示门控，就能激发出远超长上下文扁平塞入的决策可靠度。

虽然该研究目前主要立足于 ALFWorld 这一具身环境，五类记忆库的角色定义仍依赖先验设计，但其展现出的仲裁逻辑完全具备通用价值。无论是多轮工具调用、复杂的代码重构，还是开放网页环境中的深度导航，如何让沉淀下来的历史信息在正确的时间、以正确的粒度和层级出现在模型的注意力焦点中，将是解决智能体长程任务瓶颈最值得深耕的底层基础设施。
