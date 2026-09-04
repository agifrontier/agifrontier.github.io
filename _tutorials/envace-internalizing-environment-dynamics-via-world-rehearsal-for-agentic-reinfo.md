---
layout: default
title: "EnvACE：不是死磕外部沙箱，而是自我排演让Agent表现提升4.2分"
description: "在面向大语言模型（LLM）的智能体强化学习（AgenticRL）研究中，环境交互一直是一道难以逾越的高墙。要让模型学会在复杂的现实任务中连续调用工具、处理报错并达成目标，通常需要让它在成千上万次交互中碰壁摸索。"
arxiv_id: "2608.06197"
published_at: "2026-09-04T13:15:08.122237+08:00"
topics:
  - "AI Agent"
  - "强化学习"
tags:
  - "Agentic RL"
  - "BFCL-v4"
  - "EnvACE"
  - "Internalized World Model"
  - "LLM Agents"
  - "Policy-Environment Co-optimization"
related_tutorials:
  - "online-process-reward-leanring-for-agentic-reinforcement-learning"
  - "the-landscape-of-agentic-reinforcement-learning-for-llms-a-survey"
  - "training-task-reasoning-llm-agents-for-multi-turn-task-planning-via-single-turn-"
  - "a-practitioners-guide-to-multi-turn-agentic-reinforcement-learning"
---

<p class="paper-original-title" lang="en">EnvACE: Internalizing Environment Dynamics via World Rehearsal for Agentic Reinforcement Learning</p>

<img src="/images/2608.06197v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在面向大语言模型（LLM）的智能体强化学习（Agentic RL）研究中，环境交互一直是一道难以逾越的高墙。要让模型学会在复杂的现实任务中连续调用工具、处理报错并达成目标，通常需要让它在成千上万次交互中碰壁摸索。然而，构建可执行的真实沙箱环境成本极高，容易崩溃且难以大规模并发；纯代码合成的环境逻辑脆弱，覆盖面极其狭窄；而依赖另一个独立的语言模型充当外部模拟器，又往往面临虚假幻觉、状态漂移以及缺乏真实接地（Grounding）等难题。

> ArXiv URL：https://arxiv.org/abs/2608.06197v1

腾讯、新加坡国立大学、上海交大、中南大学、香港中文大学和清华浙大等机构的研究团队提出了名为 **EnvACE** 的新框架。这项研究带来了一个根本性的范式转变：智能体在强化学习过程中，并不一定非要向外索取环境反馈。相反，模型完全可以依靠同一套参数在内部“分饰两角”——既做发出工具调用的行动者（Actor），又做模拟返回观察结果的环境扮演者（Rehearsal），把整套环境动力学（Environment Dynamics）直接内化（Internalize）进模型的自身权重中。

实验表明，这种被称为“世界排演”（World Rehearsal）的训练机制在 BFCL-v4、$\tau^{2}$-Bench、VitaBench 和 FinMCP-Bench 四个主流智能体基准上，全面超越了依赖环境合成与外部模拟的基线模型。尤其在复杂的测试阶段，模型能够在不与真实世界发生任何副作用调用的前提下进行“私下推演”，使综合表现直接提升 4.2 个百分点。

<img src="/images/2608.06197v1/envace.webp" alt="三种智能体交互生成范式的对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 从依赖外部到内化环境：打破 POMDP 的传统边界

在传统的交互式智能体建模中，任务通常被形式化为有限视界的部分可观测马尔可夫决策过程（POMDP），其数学元组由状态空间 $\mathcal{S}$、动作空间 $\mathcal{A}$、观察空间 $\mathcal{O}$、状态转移函数 $P$ 以及奖励函数 $\mathcal{R}$ 构成。在这一经典范式下，策略模型与环境之间有着极其严格的分工界限：




{% raw %}$$ a_{t} \sim \pi_{\theta}(\cdot \mid h_{t}), \qquad o_{t} \sim P(\cdot \mid h_{t}, a_{t}) $${% endraw %}



在这里，策略网络 $\pi_{\theta}$ 仅负责根据当前历史上下文 $h_{t}$ 吐出下一步的工具调用或文字动作 $a_{t}$；而由这一动作引发的环境状态迁移以及伴随返回的观察结果 $o_{t}$，则完全被推给了外部不可控的转移概率分布 $P$。不管是调用真实的数据库、真实的第三方 API，还是调用一个独立的模拟器，策略网络自身始终只是被动接收外部数据。这种切分带来的直接后果是：智能体只学会了“见招拆招”，却并没有在自身的神经元连接里建立起对“我的行为会导致何种世界变化”的因果预判。

EnvACE 的核心改动，正是把环境转移的职责直接回收到了策略网络内部。它将原本单向的交互回路重构为一种交替推演序列：




{% raw %}$$ a_{t} \sim \pi_{\theta}(\cdot \mid h_{t}, \textsc{Act}) $${% endraw %}






{% raw %}$$ \hat{o}_{t} \sim \pi_{\theta}(\cdot \mid h_{t}, a_{t}, \textsc{Rehearse}) $${% endraw %}



在生成动作 $a_{t}$ 后，模型不发起外部网络请求，而是立即切换角色提示（Prompt），以 $\textsc{Rehearse}$ 身份预测该工具被执行后应该返回的观察响应 $\hat{o}_{t}$。随后，历史上下文更新为 $h_{t+1} = h_{t} \oplus (a_{t}, \hat{o}_{t})$，动作角色在此基础上继续决定下一步行动 $a_{t+1}$。通过这种轮番交替，整条轨迹不需要任何外部环境或者第二套模型参与，就能在单个 LLM 的自回归生成中完整铺展开来。

这种机制彻底改变了强化学习的采样开销。过去受限于沙箱并发吞吐、网络延迟和数据库回滚的训练瓶颈被完全打破，模型在参数内部进行着一场纯粹的“思维实验”与“沙盘推演”。

<img src="/images/2608.06197v1/method2.webp" alt="EnvACE 架构与训练推演流程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 角色分离的 GRPO：如何防范“左右互搏”的作弊行为

既然轨迹中的行动和环境反馈都是模型自己生成的，一个显而易见的风险随之浮现：模型会不会“自欺欺人”？比如，为了更容易拿到奖励，环境扮演角色会不会故意生成极度简化、甚至完全顺从行动角色的荒谬反馈？

为了确保推演不仅能自洽展开，而且能精准锚定真实世界的任务逻辑，EnvACE 引入了基于分组相对策略优化（Group Relative Policy Optimization, GRPO）的角色级优化算法。对于每一个输入任务指令 $x$，模型自采样生成包含 $K$ 条排演轨迹的分组 $\{\tau_{i}\}_{i=1}^{K}$。整条轨迹执行完毕后，由可验证的任务结果评估器或基于清单检查项的大模型裁判（LLM Judge）计算出最终的轨迹级任务奖励 $R_{i} = R(\tau_{i})$。

在反向传播与优势估计阶段，EnvACE 没有粗暴地将所有生成的 Token 混为一谈，而是设计了角色隔离的分组基准（Role-wise Baseline）。对于一次展开中的所有生成内容 $\mathcal{Y}_{i} = \{y_{i, m}\}_{m=1}^{M_{i}}$，每个输出单元都会被赋予角色标签 $r_{i, m} \in \{\textsc{Act}, \textsc{Rehearse}\}$。此时，针对不同角色的基准值是分别统计的：




{% raw %}$$ \mathcal{G}_{x, r} = \left\{ y_{i, m} \mid i = 1, \dots, K, \; r_{i, m} = r \right\} $${% endraw %}






{% raw %}$$ \mu_{x, r} = \frac{1}{|\mathcal{G}_{x, r}|} \sum_{y_{j, n} \in \mathcal{G}_{x, r}} R_{j}, \qquad A_{i, m} = R_{i} - \mu_{x, r_{i, m}} $${% endraw %}



在计算优势值 $A_{i, m}$ 时，Act 角色的产出只与这 $K$ 条轨迹中所有 Act 角色的平均表现作比较；Rehearse 角色的产出也只与同组内所有 Rehearse 角色的基准对比。这一巧妙的分离隔离了两个角色的基准偏差，避免了其中一方因为基线过高或过低而导致的梯度失真。

然而，在参数更新层面，两个角色却共享同一套权重 $\theta$。GRPO 的裁剪损失函数直接作用于所有 Token：




{% raw %}$$ \max_{\theta} \ \mathbb{E}_{x, i, m, \ell} \left[ \min\left( \rho_{i, m, \ell}(\theta) A_{i, m}, \operatorname{clip}\left(\rho_{i, m, \ell}(\theta), 1-\epsilon, 1+\epsilon\right) A_{i, m} \right) \right] $${% endraw %}



这种“基准分别计算、梯度汇聚一身”的设计带来了至关重要的耦合效应：如果 Rehearse 角色胡乱捏造环境反馈，导致后续 Act 角色做出了错误决断并最终使得任务失败，$R_i$ 就会跌入低谷，不仅 Act 角色受到惩罚，制造出虚假反馈的 Rehearse 角色也会同步承受负优势反冲。为了让最终的任务收益最大化，两个角色必须在参数共享的隐空间中形成协同：Rehearse 必须尽可能模拟出真实、严苛且符合因果逻辑的环境状态变化，而 Act 则必须学会适应这些反馈并规划出通向成功的有效工具链。

### 推理时缩放：在没有副作用的私有沙盒中推演

将环境动力学融进模型参数之后，EnvACE 在推理部署阶段展现出极其独特的优势。在传统智能体流程中，大模型面对现实系统（如转账、下单、修改线上数据库）时往往是“盲目且冒险”的：每发出一次工具调用，都会立刻对物理世界产生不可逆的副作用；一旦中间某步判断失误，整个任务便彻底崩溃。

拥有了内部世界模型的 EnvACE 能够在真正执行任何一个外部 API 之前，进行测试期计算扩展（Test-Time Scaling, TTS）。给定任务 $x$ 时，模型首先在受限的私有推演空间内独立进行 $N$ 次完整的“思维推演”。这一过程支持两种模式：

1. **并行推演模式（Parallel Rehearsal）**：模型针对同一任务独立采样出 $N$ 条自展开轨迹 $\tilde{\tau}^{(n)} \sim \Pi_{\theta}(\cdot \mid x)$，每条轨迹代表模型预演的一种执行路径，随后模型自我反思生成评估与修改建议 $f^{(n)}$。

2. **串行推演模式（Sequential Rehearsal）**：推演轨迹以自回归形式向前演进，后续的演练将之前失败或存疑的推演轨迹与反思纳入上下文，即 $\tilde{\tau}^{(n)} \sim \Pi_{\theta}\left(\cdot \mid x, \{(\tilde{\tau}^{(j)}, f^{(j)})\}_{j<n}\right)$，实现多轮纠错。

推演结束后，系统会将这些想象中的成功路径与踩坑教训高度提炼为一份精简的“排演记忆”（Rehearsal Memory, $m_x$）。直到这一步，智能体才真正走向生产环境。此时，Act 角色挂载着 $m_x$ 提供的预演先验，与外部物理环境展开唯一一次、也是胸有成竹的真实交互。所有的试错与纠偏全部在内部参数计算中完成，既保护了外部系统的安全性，又大幅降低了外部接口被死循环调用的网络成本。

<img src="/images/2608.06197v1/share8b_tau2_avg_styled.webp" alt="训练动态与基线对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 四大基准全方位检验：多轮工具交互的全面跃升

为了验证世界排演机制的通用性，研究团队在涵盖代码函数调用、多轮复杂服务调度、生活服务以及金融领域的四大权威基准上进行了全量评测，分别为 BFCL-v4、$\tau^{2}$-Bench、VitaBench 和 FinMCP-Bench。对比对象不仅覆盖了 Qwen3 原始模型与标准的 GRPO 强化学习方案，还包含了如 EnvScaler-8B、AWM-8B/14B、ScaleEnv-8B 等依赖程序化合成可执行环境的顶尖方案，以及采用独立大模型充当外部环境的 Simulator-8B。

在以复杂多轮、有状态服务交互著称的 $\tau^{2}$-Bench 上，8B 规模的 EnvACE 斩获了 36.7% 的平均得分，而使用相同主干模型进行标准 GRPO 训练的得分仅为 31.2%，性能净增长达到了 5.5 个百分点。在更加侧重真实物理世界工具链的 FinMCP-Bench 测试中，EnvACE 取得了 46.78% 的最优工具 F1 得分（TF1），相比依赖复杂环境代码合成的 EnvScaler-8B（43.68%）和 AWM-8B（42.50%）分别高出了 3.10% 和 4.28%，同时取得了 54.04% 的最高工具调用精确率。

这一组数据打破了学术界过去对于“无环境强化学习必然导致严重幻觉”的担忧。实验证明，在端到端任务成功奖励的强约束下，大模型不仅没有退化成胡言乱语的幻觉发生器，反而通过在参数内部消化工具的输入输出规范，培养出了比单纯依赖外部模拟器更为稳健的泛化决策能力。

### 为什么必须参数共享？消融实验拆解核心归因

为了探明 EnvACE 优异表现的真正来源，研究团队进行了一系列极其严格的控制变量与消融实验。其中最关键的一个问题是：**“环境扮演”和“行动决策”必须写进同一个模型的同一套参数里吗？**

团队构建了一个名为 **Per-role Policy** 的对照变体。在这个变体中，依然存在世界排演的交互回路，但 Act 角色和 Rehearse 角色被拆分给两个完全独立的模型分别负责，彼此之间在训练过程中参数完全隔离。实验结果显示，在 $\tau^{2}$-Bench 基准上，分立模型的 Per-role 策略得分为 35.5%，而参数完全共享的 EnvACE 则达到了 36.7%，取得了 1.2% 的稳固领先。

这一对比深刻揭示了世界模型“内化”的本质意义。如果环境模拟器与决策模型相互独立，环境动力学的知识就仅仅停留在外部生成文本的表层；而当两组行为共享一套权重时，模型在学习“如何精准预测一个 API 报错”时所积累的表征，会直接作用于其做决策的注意力机制中。换言之，**模型在学会当好一个“考官”的同时，自然而然地变成了一个更具前瞻性的“答题者”**。

此外，从模型规模的伸缩性来看，EnvACE 表现出了显著的 Scaling 特性。将模型底座从 1.7B 扩大到 8B 时，BFCL-v4 的均分从 31.81% 大幅跳升到 46.04%（净增 14.23%），$\tau^{2}$-Bench 更是从 15.3% 暴涨至 36.7%（净增 21.4%）。更重要的是，在 1.7B 和 8B 两个量级上，EnvACE 相较于标准 GRPO 的优势非但没有被大参数摊平，反而在 8B 模型上拉开了更大的差距。这表明模型自身的容量越大，其参数内部所能容纳的世界因果动力学就越丰富、越逼真，进而为动作决策提供的先验引导也更为强大。

<img src="/images/2608.06197v1/bfcltts.webp" alt="测试期推演预算Scaling曲线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 测试期推演的收益边界与边际递减

在具备了内化世界模型后，测试期的“私下排演”到底能释放多大的潜力？研究深入测试了推演预算 $N$（排演次数）对最终交付表现的定量影响。

在固定推演预算为 $N=2$ 的设定下，EnvACE 展现出了惊人的爆发力。在 $\tau^{2}$-Bench 和 BFCL Multi-Turn 的综合评估中，EnvACE 通过并行排演机制，将不进行推演（Non-TTS）的基础得分 36.7% 一举推高至 40.9%，取得了 4.2 个百分点的绝对增益。

值得玩味的是对照组的惨淡表现：如果直接拿未经世界排演训练的原始基座模型（Base Model）来强行进行这种推演和自我修正，其测试期扩展几乎完全失效。在并行模式下，基座模型的得分仅仅出现微弱波动；而在串行多轮推演中，基座模型的表现甚至跌破了原本不推演时的基准线。这有力地证实了一个论断：**测试期推理计算的有效性，绝非单纯堆砌 Token 或单纯调用自反思 Prompt 就能免费获取的，它极度依赖于模型权重中是否真正沉淀了高精度的环境动力学知识。**

然而，推演预算并不是越大越好。在针对推演预算 $N$ 从 1 到 3 的动态分析中，曲线呈现出清晰的倒 U 型走向：

- 当 $N$ 从 1 提升到 2 时，无论并行还是串行模式，智能体在真实环境中的成功率均呈现出显著的单调上升，表明适度的内部预演能够有效过滤潜在的决策盲区；

- 当 $N$ 继续增加到 3 时，模型表现却出现了轻微的冲高回落。

论文作者对此给出了切中要害的技术分析：过多的内部推演轨迹与反思内容会急剧膨胀上下文长度。长序列带来的注意力稀释（Attention Dilution）以及逼近或超出模型有效上下文窗口的边界效应，开始反噬模型的全局推理能力。这提醒了后续的应用落地研究：在 Agent 推理扩展的设计中，必须为私下推演设定适度的计算预算与更加高效的上下文记忆压缩机制。

### 走向更自洽的智能体强化学习

长期以来，强化学习在语言智能体领域的落地一直被深深锁死在由代码沙箱构筑的“人工鸟笼”之中。开发者为了让模型学会订机票、查账单，不得不手工编写成百上千行模拟数据库逻辑和虚拟 API，费时费力且极难迁移到开放域领域。

EnvACE 提供了一种极富启发性的全新解题思路：大模型自身经过海量互联网预训练后，其隐空间内本就潜藏着关于物理逻辑、软件工具和人类社会运转的大量隐式世界知识。与其劳师动众地在外部搭建脆弱的模拟环境，不如设计一套严谨的博弈回路，让模型在自己的参数内把这些环境动力学“演”出来，并用任务的最终成败倒逼这些动力学收敛至客观真实。

这种“让环境向内塌缩、将世界装进参数”的范式，极大地降低了 Agentic RL 的训练门槛与基础设施复杂度。当一个大语言模型在没有与外界产生任何一次网络握手的前提下，仅仅依靠自回归生成与内在的多角色推演，就能在复杂的现实交互基准上大幅提分，我们或许正在见证下一代自主智能体构建范式的起点。它不再只是一个听话执行动作的机械臂，而是一个在下笔前就已在脑海中完成千百次棋局推演的“深思熟虑者”。
