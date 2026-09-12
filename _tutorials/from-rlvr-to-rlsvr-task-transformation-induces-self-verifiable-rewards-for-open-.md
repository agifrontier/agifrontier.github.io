---
layout: default
title: "RLSVR：不是死磕外部裁判，而是用“谁是卧底”重构大模型自进化"
description: "研究团队借鉴自监督学习（Self-Supervised Learning, SSL）“从无监督数据中构造辅助代理任务（Pretext Task）以获取标签”的核心思想，提出了 RLSVR（Reinforcement Learning with Self-Verifiable Rewards） 框架。"
arxiv_id: "2607.23802"
paper_published: "2026-07-26"
published_at: "2026-09-12T13:15:08.779692+08:00"
topics:
  - "强化学习"
tags:
  - "LLM"
  - "RL"
  - "RLSVR"
  - "RLVR"
  - "SpyRL"
  - "multi-agent self-play"
related_tutorials:
  - "voyager-an-open-ended-embodied-agent-with-large-language-models"
  - "shrinking-the-variance-shrinkage-baselines-for-reinforcement-learning-with-verif"
  - "internalizing-world-models-via-self-play-finetuning-for-agentic-rl"
  - "language-self-play-for-data-free-training"
---

<p class="paper-original-title" lang="en">From RLVR to RLSVR: Task Transformation Induces Self-Verifiable Rewards for Open-Ended LLM Self-Improvement</p>

<img src="/images/2607.23802v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

大语言模型（LLM）基于可验证奖励的强化学习（Reinforcement Learning with Verifiable Rewards，简称 RLVR）在数学推导与代码编写等领域展现出了惊人的规模化扩展能力。OpenAI o1 与 DeepSeek-R1 的突破，很大程度上依赖于一个关键假设：环境存在能够通过规则精确判别正误的验证器（Verifier）。然而，当模型面对摘要生成、创意写作或开放式问答等开放领域（Open-ended Tasks）时，客观真值（Ground Truth）荡然无存。

> ArXiv URL：https://arxiv.org/abs/2607.23802v1

为了解决这一死局，过往的做法大多退回传统路线：要么依赖人工偏好构建奖励模型（Reward Model），要么借助更强大的外部大模型充当裁判（LLM-as-a-Judge）。然而，模型裁判不仅会引入固有偏见、产生评价上限瓶颈，还会带来高昂的推理调用开销。来自 Adobe、杜克大学、新加坡国立大学与宾夕法尼亚州立大学等多家机构的研究者们，在最新研究中提出了一个颠覆性的视角：**可验证性不一定必须是任务与生俱来的内在属性，而是可以通过“任务转换”（Task Transformation）人为构造出来的。**

研究团队借鉴自监督学习（Self-Supervised Learning, SSL）“从无监督数据中构造辅助代理任务（Pretext Task）以获取标签”的核心思想，提出了 **RLSVR（Reinforcement Learning with Self-Verifiable Rewards）** 框架。同时，他们基于信息不对称博弈机制设计了首个具体实现——**SpyRL**。它将开放式文本生成转换成多智能体“谁是卧底”推断游戏，让模型在完全没有外部裁判评分的情况下，仅凭环境预设的隐藏身份与投票机制实现闭环自进化。在 Qwen3-8B 模型上，SpyRL 在文本摘要与创意写作任务中斩获了 75.4% 与 77.3% 的胜率，并在数学推理上同样实现了多基准的稳定增长。

<img src="/images/2607.23802v1/rlvr.webp" alt="RLVR、自监督学习与 RLSVR 的设计范式对比" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

### 从 SSL 到 RLSVR：可验证奖励的人工重构

要理解 RLSVR，必须先厘清自监督学习与标准 RLVR 的映射关系。传统的 RLVR 需要环境对生成内容 $y$ 给出二元判定 $V(x,y) \in \{0, 1\}$；而在开放式任务中，质量评价函数 $Q(x,y)$ 属于无法精确计算的黑盒。

在计算机视觉或预训练语言模型中，自监督学习通过人为掩码（Mask）或数据增强构造出伪标签（如填空、对比视图判别），虽然模型训练的目标是代理任务，但学到的表征却能无缝迁移到下游真实任务中。RLSVR 将这一哲学完整搬到了强化学习领域。

<img src="/images/2607.23802v1/rlsvr.webp" alt="RLSVR 核心范式图示" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

在 RLSVR 范式下，系统不再死磕如何去精确评估一段开放式文本“写得好不好”，而是将原始任务指令转换为一个代理博弈环境 $\mathcal{E}$。在环境初始化时，系统预先植入一个潜变量 $z$（例如某个参与者的隐藏身份）。随后，智能体之间展开多轮交互，生成公开的输出内容与推断结果。**由于潜变量 $z$ 是由环境直接采样并记录的，任何针对 $z$ 的推断都可以被无条件、确定性地验证。** 只要环境的交互规则确保“输出质量越高的个体，暴露潜变量的风险越低（或推断正确率越高）”，那么基于规则的内部验证结果，就能完美替代不可计算的外部主观质量函数 $Q$。

### SpyRL：信息不对称下的“谁是卧底”推断博弈

基于 RLSVR 思想构建的落地框架 SpyRL，巧妙地将开放式任务包装成多智能体社交推理游戏“谁是卧底”。整个训练轮次被划分为紧密耦合的两个阶段：**执行阶段（Performing Stage）** 与 **侦测阶段（Detection Stage）**。

<img src="/images/2607.23802v1/framework.webp" alt="SpyRL 多智能体推断博弈框架图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### 1. 执行阶段：信息不对称下的能力呈现

假设环境中有 $n$ 个智能体玩家，系统在开局时随机指定一名玩家为“卧底”（Spy，记为 $u$），其余 $n-1$ 名玩家为“平民”（Civilians，记为 $\mathcal{C}$）。系统为它们分配不对称的信息输入：

- 平民玩家观察到完整、无损的任务背景文档 $x$；

- 卧底玩家仅能观察到经过掩码破坏或降级的输入 $g(x)$。

在获取各自信息后，所有玩家根据统一的任务指令模板 $\tau$，调用执行策略 $\pi^P_{\theta}$ 生成公开的交付物 $y_i$。在整个过程中，玩家使用的只是廉价的无标注文档信息，完全不需要人工标注的问答对。对于平民而言，必须尽可能详实、严密地利用完整信息完成任务；对于卧底而言，虽然信息缺失，但也必须竭尽全力基于上下文进行推断弥补，避免因明显的逻辑漏洞或信息断层暴露身份。

#### 2. 侦测阶段：完全可验证的投票裁判

所有玩家的生成内容公开后，进入侦测阶段。每位玩家基于自己最初观察到的信息以及所有人的公开产出 $Y=\{y_1, \dots, y_n\}$，调用侦测策略 $\pi^D_{\phi}$ 进行推理并投票：




{% raw %}$$v_i \sim \pi^D_{\phi}(\cdot \mid (o_i, Y)), \quad v_i \in \{1, \dots, n\}$${% endraw %}



该投票旨在指出谁是隐藏在其中的卧底。这里的绝妙之处在于：**侦测阶段的奖励 $r_i^D$ 是完全由规则确定的。** 因为卧底是谁在环境建立时就是已知的先验事实，因此只需直接判定 $r_i^D = \mathbb{I}[v_i = u]$ 即可。这一奖励机制无需任何人工介入，也没有大模型裁判的偏见干扰，天然属于纯粹的 RLVR。

<img src="/images/2607.23802v1/example.webp" alt="任务转换生成示例" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 两阶段耦合优化：零外部裁判的闭环自进化

SpyRL 之所以能够持续提升模型的真实生成质量，核心在于执行阶段与侦测阶段形成的博弈闭环。侦测阶段产生的可验证投票结果，反向构成了执行阶段的奖励信号：

- 卧底受到的惩罚与它获得的怀疑票数成正比：$r_u^P = -\beta (m_u - \bar{m}_c)$；

- 平民受到的奖励与卧底被成功指认的程度正相关，同时如果某个平民自身表现过差导致被同伴误投，也会承受个体惩罚：$r_{c_j}^P = \frac{\beta}{n_c}(m_u - \bar{m}_c) - \lambda(m_{c_j} - \bar{m}_c)$。

在这一机制下，智能体如果想要少得票、多拿奖励，唯一的路径就是在执行阶段拿出超越同组其他玩家的生成水准——更具条理的文本摘要、更富新意的故事续写、更严密的数学公式推导。

两阶段策略均采用类似 GRPO 的组内相对优势（Group-based Advantage）进行归一化计算，并施加策略裁剪（PPO-clip）与 KL 散度约束，以确保策略在对抗进化过程中的稳定性。整个系统没有任何昂贵的外部评价者，执行与侦测两项能力在博弈中相互促进、共同上升。

### 实验印证：全面超越自进化基线，逼近昂贵闭源裁判

研究人员在三大不同性质的任务上系统评测了 SpyRL：GovReport 文本摘要（平民完整文档，卧底掩码 20%）、WritingPrompts 创意写作（卧底掩码 20%）以及 Nemotron-CC-Math-v1 数学推理（卧底掩码 40%）。基线模型涵盖了当前主流的生成-求解（Proposer-Solver）自对弈框架，如 R-Zero 与 Absolute Zero。

实验数据表明，相较于已有自进化方法在开放式任务上的乏力，SpyRL 展现出了绝对优势。在 Qwen3-8B 上，SpyRL 在文本摘要和创意写作任务中分别实现了针对基础模型的 75.4% 与 77.3% 的胜率。而在原本就具备明确真值的数学推理任务上，SpyRL 依然展现出了通用的泛化增强能力，在跨越 7 项数学基准的综合测试中，分别将 Qwen3-4B 与 Qwen3-8B 的数学解题能力提升了 8.97% 和 6.16%。这证明了多智能体信息推断带来的批判性思维训练，对于严密逻辑推理同样大有裨益。

不仅如此，研究团队还将 SpyRL 与业内常用的“基于规则细则奖励”（Rubric-as-Reward, RaR）方法进行了正面硬碰硬测试。在创意写作任务中，RaR 借助新颖度、情感张力、连贯性与一致性 4 项细则，分别调用 Qwen3.5-27B 与商业闭源旗舰 GPT-4o 充当外部评分执行者。

实验结果极具说服力：

1. **击败百亿级专用裁判**：在 WritingPrompts 与 WritingBench 上，无外部裁判的 SpyRL 面对调用 Qwen3.5-27B 充当奖励模型的基线，分别取得了 59.3% 与 56.2% 的正向胜率；

2. **直逼 GPT-4o 评分基线**：面对耗资巨大的 GPT-4o-RaR，SpyRL 展现出极具竞争力的水平，甚至在文本新颖度与情感充沛度维度取得了反超；

3. **推断成本归零**：在训练周期内，Qwen3.5-27B-RaR 与 GPT-4o-RaR 仅在裁判评分阶段就分别额外消耗了约 200 美元与 900 美元的推理账单，而 SpyRL 的外部验证成本为 0 美元。

在多智能体交互规模的消融研究中，团队进一步测试了玩家组大小 $n$ 对模型能力提升的影响。

从消融曲线上可以看出，将每局玩家数量从 3 人扩展到 5 人时，模型推理基准的平均增益从 5.5 分大幅跳升至 9.3 分，边际收益最显著；而进一步增加至 6 人或 8 人时，性能提升逐渐平缓。这表明 5 人的博弈规模既能提供足够复杂的集体决策对抗环境，又兼顾了训练吞吐效率。

### 打开开放式自进化的全新路径

长久以来，学术界与工业界普遍存在一种思维定势：强化学习在数学、代码上的成功是因为这些任务天生拥有验证器；而对于开放式生成任务，如果不依赖昂贵的大模型裁判（甚至人类标注），模型自学习就必然陷入模式崩塌或偏见放大的泥潭。

RLSVR 与 SpyRL 的出现击碎了这一假设。它清晰地表明：**只要对任务形式做出巧妙重构，开放式任务的质量评价完全可以映射为环境预设潜变量的推断问题。** 这一范式不仅摆脱了固定外部裁判设定的能力天花板，更将自监督学习“利用数据自身结构自动诱导监督信号”的威力带入了强化学习。对于未来探索更通用、无需人工持续喂养的大模型后训练自进化体系而言，这种环境转换驱动的思路展现出了巨大的延展潜力。
