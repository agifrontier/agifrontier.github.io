---
layout: default
title: "TAPO：不靠额外数据，让大模型 Agent 学会“预判动作后果”"
description: "针对这一问题，本文提出了一种轻量且通用的后训练框架—— TAPO （Transition-Aware Policy Optimization）。该方案的核心突破在于： 不采集任何额外的专家数据，不增加任何多余的在线采样环境交互，也不在推理时引入额外的计算负担。"
arxiv_id: "2607.27973"
paper_published: "2026-07-30"
published_at: "2026-09-11T13:15:08.164389+08:00"
topics:
  - "AI Agent"
  - "模型优化"
tags:
  - "AI Agent"
  - "模型优化"
  - "AI论文解读"
related_tutorials:
  - "sana-video-20-hybrid-linear-attention-with-attention-residuals-for-efficient-vid"
  - "robostral-navigate"
  - "autonomous-repair-for-multi-agent-systems-via-monte-carlo-tree-search"
  - "fapo-flawed-aware-policy-optimization-for-efficient-and-reliable-reasoning"
---

<p class="paper-original-title" lang="en">TAPO: Transition-Aware Policy Optimization for LLM Agents</p>

大语言模型（LLM）驱动的智能体（Agent）在多轮交互任务中正展现出越来越强的潜力，但如何有效地对它们进行强化学习后训练（RL Post-training），依然是学术界和工业界共同面临的痛点。无论是通过 PPO 还是近期风靡的无 Critic 组内相对优势算法 GRPO，主流的强化学习范式本质上都在做同一件事：让 Agent 不断试错，然后依靠稀疏的任务奖励（Reward）来更新策略参数。

> ArXiv URL：https://arxiv.org/abs/2607.27973

这种“纯策略优化”（Policy-only Optimization）最大的盲区在于：它极度依赖最终的奖惩信号，却把交互过程中原本就存在的大量高密度物理反馈随手扔掉了。智能体在执行某个动作后，环境其实立刻给出了下一个状态反馈。因为缺乏对环境动态（Environment Dynamics）的建模，模型很难在内部建立起“如果我采取这个动作，环境究竟会变成什么样”的因果直觉。面对长链路、需要深思熟虑的决策时，纯靠奖励试错出来的策略往往脆弱不堪。

针对这一问题，本文提出了一种轻量且通用的后训练框架——**TAPO**（Transition-Aware Policy Optimization）。该方案的核心突破在于：**不采集任何额外的专家数据，不增加任何多余的在线采样环境交互，也不在推理时引入额外的计算负担，仅仅通过把强化学习 Rollout 阶段天然产生的状态转移元组重新利用起来，让同一个 Backbone 交替学习“如何做决策”和“动作会带来什么反馈”。**

在 WebShop 与 ALFWorld 等经典的长链路复杂 Agent 基准上，TAPO 在不同参数规模（1.5B 到 7B）以及不同强化学习基础算法（GRPO、GiGPO）上均取得了持续的性能提升，在 ALFWorld 上将 Qwen2.5-7B 的成功率推至 93.6%。

### 被浪费的密集信号：为什么纯 RL 不足以理解环境？

在强化学习的理论视角下，一个能够在多步目标导向任务中具备泛化能力的有限智能体，其策略网络内部必然隐式编码了对环境的预测模型（Predictive Model）。早期的无模型强化学习（Model-free RL）研究（如 UNREAL、DeepMDP、SPR）也早就证实过，在主任务之外增加对未来状态的辅助预测目标，能极大改善特征表征质量与决策稳定性。

然而，当前主流的 Agentic RL（如 RAGEN、GiGPO 等）在将强化学习迁移到长程多轮 Agent 时，关注点主要集中在如何细化优势函数计算（Advantage Assignment）、如何过滤低质轨迹，其学习信号依然主要来自标量奖励。

但请注意：智能体与环境的每一步交互，不仅产出了可能非常稀疏的奖励 $r_t$，还客观生成了一个状态转移：




{% raw %}$$(s_t, a_t) \rightarrow s_{t+1}$${% endraw %}



相比于终局才给出的稀疏标量，这个环境转移信号具备三个极为珍贵的属性：

1. **密集性（Dense）**：每走一步必定发生，不存在奖励稀疏问题；

2. **局部因果性（Action-conditioned）**：它极其明确地揭示了“在当前上下文 $s_t$ 下执行动作 $a_t$，环境会变成 $s_{t+1}$”；

3. **真实客观（Veridical）**：无论这条交互轨迹最终是成功拿到了高分还是中途失败暴毙，环境给出的状态转移反馈本身都是客观、真实的物理规则。

以往有些工作试图训练独立的世界模型（World Model）或者在 RL 之前先做一轮环境预测预训练，但这往往需要昂贵的额外数据收集流程，或者需要维护另一个参数独立的模型。TAPO 的出发点非常直接：既然每次 RL 采样（Rollout）已经把 $(s_t, a_t, s_{t+1})$ 跑出来了，为什么不直接拿它当监督信号，就地反哺给当前的模型？

### TAPO 核心机制：策略优化与状态转移预测的交替协同

TAPO 的架构并不追求复杂的设计堆砌，而是力求在极低工程侵入性的前提下实现双重学习目标的对齐。

<img src="/images/2607.27973/method.webp" alt="TAPO 框架总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整体流程如上图所示，TAPO 将训练过程划分为两个共享同一个骨干模型（Backbone）参数 $\theta$ 的交替环节：

#### 1. 策略优化阶段（Policy Optimization）

模型根据给定的任务描述 $x$ 和多轮历史观测生成动作。在当前主流的组内强化学习框架（如 GRPO 或 GiGPO）下，模型针对同一任务采样 $N$ 条轨迹构成轨迹组，利用组内平均奖励与方差计算相对优势：




{% raw %}$$A(\tau_i) = \frac{R(\tau_i) - \operatorname{mean}(\{R(\tau_j)\}_{j=1}^N)}{\operatorname{std}(\{R(\tau_j)\}_{j=1}^N)}$${% endraw %}



配合步级（Step-level）或轨迹级（Trajectory-level）的优势分配，通过 PPO 风格的截断目标函数 $\mathcal{J}_{\text{RL}}(\theta)$ 更新参数，专注于提升高回报动作的生成概率。

#### 2. 状态转移监督阶段（Transition Supervision）

与传统做法将轨迹用完即弃不同，TAPO 直接从这些在线交互轨迹中提取所有三元组：




{% raw %}$$\mathcal{D} = \{(s_t, a_t, s_{t+1})\}$${% endraw %}



将这些数据重新组织为一个动作条件下的下一观测预测任务（Action-conditioned Next-observation Prediction）。共享参数的模型需要以当前状态 $s_t$ 和执行的动作 $a_t$ 为条件，自回归预测下一个状态 $s_{t+1}$：




{% raw %}$$\mathcal{L}_{\text{TS}}(\theta) = -\mathbb{E}_{(s_t, a_t, s_{t+1})\sim\mathcal{D}} \left[\log p_\theta(s_{t+1} \mid s_t, a_t)\right]$${% endraw %}



在优化调度上，TAPO 引入了一个交替间隔超参数 $I$。模型正常进行策略优化梯度更新；每当强化学习迭代达到 $I$ 的整数倍时（即 $t \mathbfod I = 0$），便插入一次基于 $\mathcal{L}_{\text{TS}}$ 的监督梯度更新。

这种联合方式的关键收益在于：**策略生成与转移预测共享了完全一致的模型参数**。当模型被强制去学习预测动作所引发的客观环境变化时，其隐空间表征被深度注入了环境物理因果的约束；当模型回过头去进行策略规划与思维链（Chain-of-Thought）推演时，这种因果敏感度能天然阻止模型产生脱离环境物理规律的“幻觉动作”。

### 实验验证：跨环境、跨规模的稳健增益

为了验证 TAPO 的实际效果，研究团队在两大极具代表性的长链路 Agent 基准上进行了全面评测：

- **WebShop**：模拟真实电商购物网站的多步文本导航与商品检索任务，极度依赖对网页交互后反馈的理解；

- **ALFWorld**：具身模拟居家决策环境，包含大量复杂的物理交互状态转换。

实验采用了不同参数规模的开源基座（如 Qwen2.5-1.5B-Instruct、Qwen2.5-7B-Instruct），并分别与纯 GRPO 及 GiGPO 进行了横向对比。

数据表明，无论基座大小、无论底层搭配的是哪种 RL 策略梯度算法，TAPO 均带来了显著且一致的提升：

- 在 **ALFWorld** 环境下，对于 1.5B 小模型，TAPO 将 GRPO 的成功率从 72.8% 提升至 76.4%，将 GiGPO 的成功率推升至 88.4%；而在 7B 模型上，TAPO 同样展现了强大的增益，使得 GRPO 的成功率提升了 6.0 个百分点，GiGPO 提升了 2.8 个百分点。

- 在与现有的显式世界模型/状态转移增强方法（如 Early Experience、RWML）的对比中，使用 Qwen2.5-7B-Instruct 的 TAPO 在 ALFWorld 上斩获了 93.6% 的高成功率，优于 Early Experience 报告的 82.8% 和 RWML 的 90.1%。更重要的是，TAPO 完全不需要像这些先驱工作那样在前置阶段额外收集专门的模型训练数据。

### 深入剖析：模型真的理解动作后果了吗？

TAPO 的收益到底来自哪里？是单纯因为多做了一些自回归预测作为正则化，还是模型确实掌握了环境因果？研究团队通过多组消融与机理分析给出了答案。

首先是**交替频率 $I$ 的敏感性分析**。在 WebShop 环境下，使用 Qwen2.5-1.5B 搭配 GRPO 进行测试时，不同的交替间隔展现出了极具启发性的规律：

- 当 $I=4$ 时，智能体取得了 66.2% 的最高成功率；

- 无论是更高频的交替还是更低频的交替，其最终表现均稳定超越了完全不加转移监督的纯 GRPO 基准；

- 这一现象证明，引入环境反馈学习这一方向本身容错度很高，只要在“探索新策略”和“消化环境物理规律”之间维持一个合理的步调，模型就能稳定获益。

其次是**全流程交替训练的必要性**。研究者对比了一种对照实验：仅在训练初期将状态转移监督作为预热（Warm-up），随后切回纯 RL。结果显示，仅做前期预热虽然比完全不做略有提升，但最终性能明显逊色于全程保持交替训练的 TAPO。这表明，随着 RL 策略的演进，智能体探索到的状态空间分布是在动态变化的，将环境转移监督贯穿始终，才能持续为策略学习提供精准的物理锚点。

最后是**针对环境转移建模能力的定量与定性检验**。在保留的独立测试集上，团队测试了模型对下一状态的预测困惑度（Perplexity, PPL），并在实际 Case 中观察模型的生成逻辑。

分析发现，经过 TAPO 训练的模型不仅预测下一状态的 PPL 显著更低，而且在 WebShop 的实际决策思维链中，展现出了惊人的“前瞻模拟”（Lookahead Simulation）能力：在真正做出点击某个隐藏尺寸选项之前，模型在 `<think>` 标签内就已经预测到了该动作会暴露出的属性字段，甚至衍生出了条件分支推理——“如果点击后显示的尺寸符合要求则继续下单，如果不符合则返回重新搜索”。这种深层前瞻能力的自发涌现，强有力地证明了模型不再仅仅是在做生硬的文本模式匹配，而是确实内化了环境的动态逻辑。

### 总结与展望

TAPO 提供了一个兼具学术洞见与工业落地价值的 Agent 后训练范式。它敏锐地指出了当前 LLM 强化学习中一个习以为常的资源浪费：**智能体与环境交互留下的每一次“动作-新状态”转换，都是极其宝贵且完全免费的高密度监督样本。**

通过将纯策略梯度优化与轻量级的下一观测预测交替结合，TAPO 在零额外数据、零额外采样成本、零推理时延迟的前提下，显著增强了大模型 Agent 对行动后果的感知力与长链路决策鲁棒性。这种将世界模型思想内化为策略网络“辅助训练目标”的思路，为构建更可信、更善于深思熟虑的自主智能体提供了极具通用性的实践路径。
