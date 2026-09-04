---
layout: default
title: "PlanPO：成功不该同等奖赏！给轨迹做长度归一化，多轮Agent性能提升27.2%"
description: "在多轮交互任务中训练大语言模型（LLM）智能体，强化学习（RL）正在成为不可替代的核心范式。从DeepSeek-R1走红以来，组相对策略优化（GroupRelativePolicyOptimization，简称GRPO）凭借其“不需要训练昂贵Critic价值网络、直接在同题采样的候选组内对比结果”的轻量化优势。"
arxiv_id: "2608.17289"
published_at: "2026-09-04T11:26:52.370742+08:00"
topics:
  - "AI Agent"
  - "推理"
tags:
  - "ALFWorld"
  - "GRPO"
  - "PlanPO"
  - "SciWorld"
  - "advantage collapse"
  - "agentic LLMs"
related_tutorials:
  - "a-survey-of-reasoning-and-agentic-systems-in-time-series-with-large-language-mod"
  - "agent0-unleashing-self-evolving-agents-from-zero-data-via-tool-integrated-reason"
  - "beyond-outcome-rewards-step-level-self-distilled-policy-optimization-for-deep-se"
  - "harnessx-a-composable-adaptive-and-evolvable-agent-harness-foundry"
---

<p class="paper-original-title" lang="en">PlanPO: Group Planning-Aware Policy Optimization for Multi-Turn Agentic LLMs</p>

<img src="/images/2608.17289v1/A__title.webp" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

在多轮交互任务中训练大语言模型（LLM）智能体，强化学习（RL）正在成为不可替代的核心范式。从 DeepSeek-R1 走红以来，组相对策略优化（Group Relative Policy Optimization，简称 GRPO）凭借其“不需要训练昂贵 Critic 价值网络、直接在同题采样的候选组内对比结果”的轻量化优势，被广泛迁移到工具调用、网页导航和具身智能交互中。

> ArXiv URL：https://arxiv.org/abs/2608.17289v1

然而，当把 GRPO 直接套用到多轮交互场景时，一个隐蔽却致命的问题随之浮现：**优势坍缩（Advantage Collapse）**。

在长程交互环境中，同一个任务往往存在截然不同的完成路径。有的 Agent 目标明确，几步操作干净利落完成任务；有的 Agent 则像无头苍蝇一样反复试探、在死胡同里绕圈、在思考链里产生事实幻觉，但在耗尽步数前侥幸触发了成功条件。在现有 GRPO 框架下，这两种轨迹都会拿到完全相同的最终奖励值（比如 $+1$ 或 $+10$）。由于同组内成功的轨迹奖励无差异，算法计算出的相对优势几乎为零，模型根本无法区分哪种交互策略更具通用规划能力，甚至会将冗长、低效的试错模式固化为行为习惯。

针对这一瓶颈，来自暨南大学、南洋理工大学、上海交通大学和厦门大学的研究团队提出了 **PlanPO（Group Planning-aware Policy Optimization，组规划感知策略优化）**。该方法没有引入复杂的辅助验证模型或人工设计的启发式奖励，而是敏锐地抓住了一个天然存在却被长期忽视的物理信号——**轨迹与回复的多尺度长度**。通过将长度惩罚严格限制在“任务成功”的子集内进行组内归一化，PlanPO 让模型学会了区分高效规划与侥幸成功。在 ALFWorld、WebShop 和 SciWorld 三大高难度多轮基准测试中，PlanPO 相比 GRPO 取得了平均 27.2% 的大幅性能提升，不仅使交互轮数减半，还让整体训练时间缩短了 12.5%。

<img src="/images/2608.17289v1/intro.webp" alt="成功轨迹的异质性与 PlanPO 在不同基准上的表现" style="width:85%; max-width:600px; margin:auto; display:block;">

### 为什么说“无差别的成功奖励”正在拖垮多轮 Agent？

要理解多轮交互 RL 的困难，首先要看现有方案为了赋予多轮轨迹信用分配（Credit Assignment）所付出的代价。

在传统的单轮数学或代码推理任务中，答案的对错通常由最终的静态结果决定。但在多轮环境中，Agent 的决策是一个有限时界的马尔可夫决策过程（MDP）。在每一个交互轮次 $t$，模型接收环境的观察状态 $\boldsymbol{s}_t$，输出一段包含思考过程与具体动作的文本回复 $\boldsymbol{a}_t$，环境随之转移到新状态并可能给出一个极其稀疏的标量反馈。由于中间过程往往没有可靠的即时打分，直接使用 PPO 算法需要训练一个庞大的 Value 模型或过程奖励模型（PRM）来预测单步价值。这不仅引入了巨大的显存开销，还经常因为价值估计偏差导致策略更新漂移。

近年来，社区尝试通过挖掘采样轨迹来构建单步奖励。例如 HiPER 尝试做子任务分解，GiGPO 试图通过锚点状态识别动作空间的优势，还有些工作通过失败反思（Reflection）或代码执行反馈提供信用。但这些方法普遍重度依赖特定任务的人工启发式设计，泛化到新场景极其困难。

更糟糕的是，如果完全依赖纯结果导向的组相对优化，成功轨迹内部的“异质性”（Heterogeneity）会被直接抹平。如上图所示，当面对同一个厨房任务时，优秀的策略可以直接走到柜台拿到土豆，而低效的轨迹可能会在微波炉、水槽之间反复巡检，甚至在思考链 `<think>` 标签中胡言乱语“我已经拿到了土豆”，最终在动作 `<action>` 标签里蒙对操作。这两种轨迹的最终结果都是成功，如果给予相同的相对优势，策略梯度就会同等程度地强化那些冗长、混乱的思维和动作模式。最终，模型学到的是某种特定任务下的冗余行为模板，而非可迁移的高阶规划能力。

### 核心机制：成功约束下的由粗到细相对优势

既然冗余的交互和啰嗦的思考是低质轨迹的典型表征，那么直接在奖励中扣除长度惩罚不行吗？答案是否定的。以往的研究早已证实，如果在整个动作空间上盲目做长度惩罚（Length Penalty），模型为了迎合奖励函数，往往会迅速退化成“直接输出结束符以尽早止损”，导致任务成功率断崖式下跌。

PlanPO 的核心洞察在于：**长度信号只有在“已达成任务目标”的前提下，才具有指代规划质量的正向意义。** 换句话说，短并不代表好，但在同样成功的解决方案中，更短、更紧凑的轨迹必然代表了更高密度的有效规划和更低的推理幻觉。

为此，PlanPO 在标准 GRPO 框架内设计了一套由粗到细（Coarse-to-Fine）的组相对优势计算机制。

<img src="/images/2608.17289v1/fw.webp" alt="PlanPO 算法架构总览" style="width:85%; max-width:600px; margin:auto; display:block;">

#### 1. 轨迹级长度归一化优势（粗粒度）

对于同一个任务输入 $\boldsymbol{x}$，策略模型采样出 $N$ 条完整交互轨迹构成的候选组 $\mathcal{G}_{\boldsymbol{x}} = \{\boldsymbol{\tau}_1, \boldsymbol{\tau}_2, \ldots, \boldsymbol{\tau}_N\}$。若某条轨迹最终成功（属于成功集合 $\mathcal{G}_{\boldsymbol{x}}^{\mathrm{U}}$），其获得的最终奖励 $R(\boldsymbol{\tau}_i)$ 会被其总交互轮数 $T_i$ 进行归一化；而对于未成功的轨迹，其分数保持为 0：




{% raw %}$$R^{\mathrm{E}}(\boldsymbol{\tau}_{i})=\mathds{1}[\boldsymbol{\tau}_{i}\in\mathcal{G}_{\boldsymbol{x}}^{\mathrm{U}}]\,\frac{R(\boldsymbol{\tau}_{i})}{T_{i}}$${% endraw %}



紧接着，算法在当前任务组的所有采样中计算轨迹级别的相对优势：




{% raw %}$$A^{\mathrm{E}}(\boldsymbol{\tau}_{i})=\frac{R^{\mathrm{E}}(\boldsymbol{\tau}_{i})-\operatorname{mean}\left(\left\{R^{\mathrm{E}}(\boldsymbol{\tau}_{j})\right\}_{j=1}^{N}\right)}{F_{\mathrm{norm}}\left(\left\{R^{\mathrm{E}}(\boldsymbol{\tau}_{j})\right\}_{j=1}^{N}\right)}$${% endraw %}



其中 $F_{\mathrm{norm}}(\cdot)$ 通常为组内分数的标准差 $\operatorname{std}(\cdot) + \epsilon$。这个优势值 $A^{\mathrm{E}}(\boldsymbol{\tau}_{i})$ 会广播（Broadcast）给该轨迹内的每一个交互轮次。如此一来，如果同组内有多条轨迹成功，交互轮数更少、路径更直击要害的轨迹将获得显著更高的正向优势，在环境探索层面形成了对冗余游荡行为的有效抑制。

#### 2. 单轮回复长度归一化优势（细粒度）

光有轨迹级别的轮数约束还不够。在具体某一步交互中，模型生成的思维链和动作描述同样可能存在长篇大论、逻辑脱节的问题。为了给单轮回复提供精细化的信用分配，PlanPO 将同一任务下所有成功轨迹产生的单步回复汇总到一个集合 $\mathcal{G}_{\boldsymbol{x},t}^{\mathrm{S}}$ 中。

对于成功轨迹中的第 $t$ 轮回复 $\boldsymbol{a}_{i,t}$，根据该轮生成的 token 总数 $L_{i,t}$ 计算其精细化得分，并计算组内相对优势：




{% raw %}$$R^{\mathrm{S}}(\boldsymbol{a}_{i,t})=\mathds{1}[\boldsymbol{\tau}_{i}\in\mathcal{G}_{\boldsymbol{x}}^{\mathrm{U}}]\frac{R(\boldsymbol{\tau}_{i})}{L_{i,t}}$${% endraw %}






{% raw %}$$A^{\mathrm{S}}(\boldsymbol{a}_{i,t})=\frac{R^{\mathrm{S}}(\boldsymbol{a}_{i,t})-\operatorname{mean}\left(\left\{R^{\mathrm{S}}(\boldsymbol{a}_{j,t})\mid\boldsymbol{a}_{j,t}\in\mathcal{G}_{\boldsymbol{x},t}^{\mathrm{S}}\right\}\right)}{F_{\mathrm{norm}}\left(\left\{R^{\mathrm{S}}(\boldsymbol{a}_{j,t})\mid\boldsymbol{a}_{j,t}\in\mathcal{G}_{\boldsymbol{x},t}^{\mathrm{S}}\right\}\right)}$${% endraw %}



这个优势值 $A^{\mathrm{S}}(\boldsymbol{a}_{i,t})$ 会直接作用于该轮回复生成的所有 token 上，迫使模型在确保当前动作正确的前提下，尽可能去除思维链中的胡言乱语与冗余废话。

#### 3. 动态衰减融合与偏差-方差权衡

得到粗粒度的轨迹优势 $A^{\mathrm{E}}$ 和细粒度的单轮优势 $A^{\mathrm{S}}$ 后，如何将二者有机结合？PlanPO 没有采取简单的静态求和，而是引入了随训练步数动态线性衰减的加权系数 $\alpha(k)$：




{% raw %}$$A^{\mathrm{PlanPO}}(\boldsymbol{a}_{i,t})=A^{\mathrm{E}}(\boldsymbol{\tau}_{i})+\alpha(k)A^{\mathrm{S}}(\boldsymbol{a}_{i,t})$${% endraw %}






{% raw %}$$\alpha(k)=\mathtt{LinearDecay}(k;\alpha_{\rm init},\alpha_{\rm final})$${% endraw %}



作者在论文中给出了严谨的理论证明（Theorem 1）。从理论上看，令 $A^{\mathrm{E}}$ 的方差为 $v_E$，$A^{\mathrm{S}}$ 的方差为 $v_S$：

- 当 $\alpha$ 较大时，估计器的偏差（Bias）较小，能充分保留单轮生成的细粒度引导；但由于跨轨迹、跨轮次的 token 长度方差较大，整体梯度的采样方差 $\operatorname{Var}(A_\alpha)$ 也会随之上升。

- 当 $\alpha$ 较小时，方差被大幅压缩至接近 $v_E$，有利于维持策略更新的整体平稳性，但会引入一定的收缩偏差。

因此，采用从大到小衰减的 $\alpha(k)$ 机制极为精妙：在训练初期，模型极度需要单步 token 级别的精细惩罚来迅速纠偏啰嗦、发散的语言模式；而到了训练中后期，策略生成趋于稳定，此时逐步降低 $\alpha$ 权重，让全局交互规划（轨迹轮数）占据主导地位，从而在保证策略稳定的同时实现渐进式收敛。

随后，组合后的优势值 $A^{\mathrm{PlanPO}}$ 被直接送入带有 PPO 截断项和参考模型 KL 散度约束的目标函数中，完成整套参数更新。

### 实验见证：全面超越强基线，交互更短且省显时

为了验证 PlanPO 的实际效果，研究团队在三大极具代表性的多轮交互基准上开展了严格测评：模拟家庭具身环境 ALFWorld、复杂网络购物决策 WebShop，以及极具探索挑战的科学实验环境 SciWorld。实验基座模型选用了开源领域极具竞争力的 Qwen2.5-1.5B-Instruct 和 Qwen2.5-7B-Instruct。

#### 1. 任务综合表现显著跃升

实验结果表明，经过 PlanPO 微调后的开源小模型展现出了惊人的任务解决能力，甚至在多项指标上击败了顶尖的闭源前沿模型。

在 ALFWorld 具身测试中，基于 Qwen2.5-1.5B 的 PlanPO 在全部 6 个交互类别上取得了高达 91.3% 的综合成功率。相比于原始 GRPO 的 72.8%，绝对提升达 18.5 个百分点（相对提升超过 25%）；对比此前专门针对智能体改进的强基线 GiGPO（86.5%）和 EMPG（81.8%），PlanPO 依然保持了显著优势。

在任务跨度更大的 WebShop 中，包含商品搜索、属性比对、页面跳转等复杂动作，PlanPO 在 1.5B 模型上达到了 78.4 的成功得分，超越了 GPT-4o（69.2）和 Gemini-2.5-Pro（74.6）等未微调的闭源旗舰模型，相比 GRPO 提高了 6.4 个得分点。

在最为开放且对容错率极低的科学环境 SciWorld 中，针对 GRPO 改进的 AgentGym-RL 基准得分为 50.5，而引入了规划感知优势的 PlanPO 直接将综合得分刷新至 68.46，展现出了在复杂长链条探索环境中强大的抗干扰与自主纠偏能力。综合三大基准，PlanPO 相比标准 GRPO 实现了平均 27.2% 的跨越式增长。

<img src="/images/2608.17289v1/abla.webp" alt="消融实验与训练耗时分析" style="width:90%; max-width:700px; margin:auto; display:block;">

#### 2. 消融分析：关键组件如何发挥作用？

为了探究收益的真实来源，作者对权重调度系数 $\alpha(k)$ 和不同层级的优势进行了深度消融（如上图所示）：

- 在加权衰减设置上，将初始权重设为 $\alpha_{\mathrm{init}}=0.1$ 能够取得最佳效果。如果将 $\alpha$ 设为 0（即完全不考虑单轮 token 长度），性能会出现明显下滑；而如果将 $\alpha_{\mathrm{init}}$ 设得过大（如 0.5），过高的方差反而会破坏训练稳定性，这与理论推导的偏差-方差权衡完全吻合。此外，动态衰减的调度方案明显优于固定常数的权重方案。

- 在层级拆解上，完全剔除轨迹级优势 $A^{\mathrm{E}}$（仅保留单轮 token 优势）会导致性能大幅暴跌；而只保留 $A^{\mathrm{E}}$、去除 $A^{\mathrm{S}}$，模型依然能取得不错的基准性能。这证明了**轨迹维度的长程规划才是多轮任务的立足之本**，单轮层面的 token 精简则扮演了画龙点睛的微调润色角色。

#### 3. 计算开销不增反降

以往很多尝试改善信用分配的方法，由于需要运行辅助价值网络或多次模型回溯，往往伴随着数倍的训练时间膨胀。然而从上图右侧的运行时耗分布（Runtime Breakdown）可以看出，PlanPO 引入的计算开销主要在组内奖励归一化和优势相加环节，这部分张量运算在四卡 A40 集群上耗时不足整体迭代的 1%，在计算图上完全可以忽略不计。

更令人惊喜的是，由于 PlanPO 极度鼓励高效率的规划路径，模型在训练采样时自主放弃了大量无谓的来回兜圈与死循环探索。在同样的步数设置下，**PlanPO 使整体任务的训练时延净缩减了 12.5%**。在不加显存包袱的同时实现更快的收敛，这在强化学习算法中尤为难能可贵。

### 深度剖析：学会的是真实规划，还是单纯变短了？

强化学习领域经常会遇到“刷指标”的作弊现象：模型可能只是把轨迹强行截断，利用评价指标的漏洞获利。PlanPO 究竟是学会了真正的环境规划，还是仅仅退化成了长度最小化？研究团队从泛化性验证和轨迹形态学两个维度给出了极具说服力的证据。

首先是**分布外（OOD）泛化测试**。如果模型只是死记硬背某些特定任务的捷径动作，在没见过的场景下必然原形毕露。在 ALFWorld 的未见任务分布测试中，PlanPO 的成功率依然稳定在 87.1%，相比原始 GRPO 的 70.1% 净胜 17.0 个百分点，相比 GiGPO 亦高出 4.7 个百分点。从域内（91.3%）到域外（87.1%），PlanPO 仅出现了 4.2 个百分点的温和衰退。这种极强的跨场景适应力表明，模型提取出的是对环境状态敏锐感知、快速定位核心依赖的通用规划元能力。

<img src="/images/2608.17289v1/len_norm_ana.webp" alt="ALFWorld 任务中的平均长度对比" style="width:85%; max-width:600px; margin:auto; display:block;">

其次是**交互行为与长度分布的变化趋势**。从上图的对比曲线可以清晰地看到两种完全不同的进化路径：

- 在交互轮次（Turns）上，标准 GRPO 在训练中后期平均需要 26.1 轮才能解决任务，甚至随着探索范围扩大有变冗长的趋势；而 PlanPO 训练出来的模型，平均交互轮次大幅腰斩至 13.8 轮。

- 在回复长度（Tokens）上，GRPO 生成的单步思维与动作极其拖沓，平均长度达 95.1 个 token；PlanPO 则自适应地将回复浓缩至 56.3 个 token，剔除了大量“自言自语”的幻觉声明。

至关重要的是，对比实验表明，如果不加“成功”作为前置条件、直接在所有采样上施加无条件长度惩罚，任务成功率会在几个 step 内迅速归零（模型选择立刻认输）。纯粹的长度缩减会扼杀推理，而**在成功轨迹集合内的相对长度比拼**，才构成了引导模型从混沌试错走向清晰规划的有效动力学。

### 总结与展望

在长思维链推理与多轮智能体快速演进的当下，PlanPO 为社区提供了一个非常干净且极富启发性的视角。

它清晰地指出了当前组相对策略优化在多轮交互中的结构性缺陷：仅仅以终局胜负论英雄，会严重低估低效轨迹对策略空间的污染。而解决这一问题并不必然需要堆砌庞大、脆弱的判别模型。在“同题采样”的天然对照组内，利用成功条件约束下的交互轮数和生成长度构建分层相对优势，就能以近乎零成本的算力开销，优雅地将冗余探索转化为强大的监督信号。

对于正在从事 Agentic RL、工具调用系统优化以及端到端多轮对话模型训练的研究者与工程师而言，PlanPO 的核心思想具有极强的即插即用价值：别让低质量的侥幸成功稀释了策略梯度，把效率作为优势函数的一等公民，大模型智能体完全能够在更短的交互里展现出更深邃的规划智慧。
