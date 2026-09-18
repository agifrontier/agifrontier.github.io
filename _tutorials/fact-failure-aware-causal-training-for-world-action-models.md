---
layout: default
title: "FACT：颠倒动作与未来生成顺序，把失败轨迹变成因果监督"
description: "他们提出了 FACT（Failure-Aware Causal Training）框架，打破了“先脑补未来、再倒推动作”的定势，颠倒为“先决策动作、再因果推演未来与任务进展”。"
arxiv_id: "2608.10232"
paper_published: "2026-08-10"
published_at: "2026-09-18T13:15:07.871050+08:00"
topics:
  - "模型训练"
tags:
  - "FACT"
  - "WAMs"
  - "action-conditioned video prediction"
  - "bimanual manipulation"
  - "causal training"
  - "failure rollouts"
related_tutorials:
  - "mobilewam-bridging-world-action-models-to-mobile-manipulation-with-chain-of-fore"
  - "learning-fine-grained-bimanual-manipulation-with-low-cost-hardware"
  - "gamewam-a-world-action-model-for-video-games"
  - "world-tokens-enhancing-embodied-policies-with-training-time-world-modeling"
---

<p class="paper-original-title" lang="en">FACT: Failure-Aware Causal Training for World-Action Models</p>

<img src="/images/2608.10232v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在具身智能与机器人策略学习中，世界动作模型（World-Action Models，简称 WAMs）正逐渐成为一股核心力量。与单纯从图像映射到动作的传统 VLA 模型相比，WAM 最大的魅力在于引入了对物理世界演变规律的预判能力：模型不仅要决定当前机械臂往哪里挪，还要在脑海中预演未来几秒内场景会发生什么变化。

> ArXiv URL：https://arxiv.org/abs/2608.10232v1

然而，现有的世界模型在落地时普遍遭遇了一个尴尬的理论悖论。目前的绝大多数方案要么先生成未来的视频轨迹，再借助逆动力学模型（Inverse Dynamics Model）反解动作；要么先预测未来的目标图像特征，以此作为条件引导动作生成。这两种范式几乎都建立在人类专家的“成功演示数据”之上。这带来了一个隐蔽却致命的问题：模型从未见过失败，因此它根本不知道一个糟糕的动作究竟会导致何种糟糕的后果。在真机部署中，一旦机械臂执行了一个走形或失误的动作，世界模型往往依然会沉浸在“盲目乐观”的幻觉中，预测出一个任务圆满完成的美好未来。这种成功偏置（Success-biased hallucination）让世界模型的推演能力在关键时刻失去了纠偏价值。

来自加州大学圣迭戈分校（UC San Diego）的研究团队在论文《FACT: Failure-Aware Causal Training for World-Action Models》中给出了一个直击本质的解法。他们提出了 FACT（Failure-Aware Causal Training）框架，打破了“先脑补未来、再倒推动作”的定势，颠倒为“先决策动作、再因果推演未来与任务进展”。更关键的是，FACT 建立了一套巧妙的因果掩码机制，使得海量的真实失误与回滚轨迹（Failure rollouts）能够被全量回收——模型既不会把错误的动作当成模仿学习的目标，又能把错误动作引发的灾难性后果作为世界模型的真实监督信号。在真实双臂操作任务中，这一设计将策略成功率推升至 92%，同时显著消除了模型面对坏动作时的虚假乐观。

<img src="/images/2608.10232v1/x1.webp" alt="FACT 核心机制" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么现有世界模型总是“盲目乐观”？

要理解 FACT 的突破，必须先拆解当前具身策略学习在利用负面数据时的两难境地。

在强化学习或在线微调中，机器人与环境交互不可避免地会产生大量失败轨迹，比如夹爪滑脱、撞击桌面或抓取落空。对于传统的行为克隆（Behavioral Cloning, BC）而言，这些数据属于必须丢弃的“毒药”，因为直接计算模仿损失函数会导致机械臂学会各种奇形怪状的失误动作。但对于真正理解物理规律的世界模型而言，失败数据恰恰是极高价值的因果样本：它明确地告诉系统，“如果你在这个角度松开夹爪，物体就会掉在地上”。

现有主流 WAM 无法利用这层监督的核心原因在于时序与因果关系的倒错。当模型被训练成根据当前帧直接预测未来的成功画面时，未来帧就变成了一个单向的“任务目标”。只要任务指令不变，模型脑海中生成的永远是那个最终成功的视频。随后，动作解码器必须顺应这个成功的视频去脑补动作。一旦机器人在实际执行中手滑，这套推演链路就会发生断裂：动作已经不可挽回地偏离了轨道，但下游的视频生成器却由于缺乏针对“当前坏动作”的因果条件输入，依旧固执地渲染出任务成功的下一帧。

这种脱节意味着，现存模型虽然名为“世界模型”，本质上却只是“成功演示的记忆播放器”。它们没有建立起因果意义上的转移概率 $P(s' \mid s, a)$，而只是记住了联合分布 $P(s', a \mid s, \text{success})$。要摆脱这一困境，动作必须从未来的附属品，转变为决定未来演变的显式原因。

### 因果时序与双向解耦：FACT 的架构机制

FACT 颠覆了这一拓扑结构，提出了 Action-conditioned（动作作为条件）的前向推演架构。在 FACT 的逻辑中，输入当前的多视角 RGB 图像与本体感知状态后，模型首先输出一个动作块（Action Chunk）$a_{t:t+H}$；紧接着，模型以当前状态和刚刚生成的动作块为联合输入，去预测未来一段时间内的多视角视频片段 $o'_{t:t+K}$，以及一个标量形式的任务进展值（Task-Progress Value）$v_t$。

<img src="/images/2608.10232v1/x2.webp" alt="FACT 模型架构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了在一个统一的大模型内实现这一复杂的因果依赖，FACT 采用了一个共享权重的视频扩散 Transformer（基于开源的 WAN2.2-5B 视频骨干网络）作为核心底座，并挂载轻量级的 Action Adapter。如图所示，整个序列的 Token 布局被切分为五个功能区块：

1. 参考与当前观测前缀 $z^P$（包含文本指令、历史帧与机器人当前位姿）；

2. 预测动作 Token $z^A$（模型自身去噪生成的动作块）；

3. 真实动作条件 Token $z^G$（用于世界模型前向推演的标准因果条件）；

4. 任务进展值 Token $z^V$；

5. 未来视频潜码 Token $z^I$。

这一设计最精妙的构想体现在训练期的**教师强制因果掩码（Teacher-Forced Causal Mask）**中。

<img src="/images/2608.10232v1/x3.webp" alt="因果注意力掩码" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

如果简单地把预测出来的动作直接传给下游的未来预测模块，训练初期动作模块的巨大噪声会彻底摧毁未来视频的重建质量；反之，如果让动作预测模块直接看到环境里的标准动作，动作生成又会发生严重的因果信息泄露。

FACT 的注意力掩码矩阵漂亮地解耦了两个分支：动作预测区块 $A$ 只能看到前缀条件 $P$，以纯粹的策略网络形态运行；而负责预测未来视频的区块 $I$ 与价值评估区块 $V$，则被强制注意力对齐到“干净的真实动作条件” $G$ 上，完全对预测动作 $A$ 视而不见。由于 $A$ 与 $G$ 之间互不可见，两个学习目标在同一个自注意力矩阵里相安无事。

这就为消化失败数据扫清了最后的制度障碍。研究人员将训练样本划分为成功演示集 $\mathcal{D}_s$ 与失败轨迹集 $\mathcal{D}_f$。当输入成功样本时，模型全面计算动作模仿损失、未来视频重建损失以及任务进展回归损失：




{% raw %}$$ \mathcal{L}_{\mathcal{D}_s} = w_a \mathcal{L}_a + w_v \mathcal{L}_v + w_I \mathcal{L}_I $${% endraw %}



而当输入失败样本时，由于因果掩码的存在，研究人员可以极其自然地将动作模仿的损失权重归零（即 $w_a = 0$），但完整保留未来视频与任务进展的预测监督：




{% raw %}$$ \mathcal{L}_{\mathcal{D}_f} = w_v \mathcal{L}_v + w_I \mathcal{L}_I $${% endraw %}



在失败数据中，机械臂实际执行的操作往往很差，但这套坏动作到底引发了怎样的视觉后果（例如杯子被碰倒、物体滑落），正是世界模型必须啃下的硬骨头。与此同时，任务进展标签 $v_t$ 会在判定发生失误的瞬间被严厉扣分。模型由此被明确告知：执行这个特定动作，画面中将出现物体脱落的混乱场景，而任务完成度也将骤降为零。

### 两阶段推断与即插即用的动作打分

在训练完成之后，FACT 在推理部署阶段展现出极高的灵活性。由于世界模型的去噪分支高度独立，FACT 允许机器人在“快思考”与“慢思考”之间自由切换。

在标准推断流程中，系统仅需激活第一阶段（Stage 1）。此时模型只需在因果前缀下，通过 20 步 Flow-Euler 积分对动作 Token 进行单向去噪，即可直接输出长度为 $H=48$ 的连续动作序列。这一阶段完全省去了重型视频扩散步骤，借助预先计算并缓存的前缀 KV Cache，机械臂能够以极高的控制频率响应环境。

但在面对复杂、长程或容错率极低的操作场景时，FACT 允许无缝开启第二阶段（Stage 2）——**基于进展价值的候选动作重排序（Candidate Scoring）**。

由于 FACT 的任务进展预测头是在真实的成功与失败后果上联合打磨出来的，它具备了普通模仿学习模型完全不具备的鉴别力。传统策略模型的价值头因为只见过高分动作，面对偏离分布的随机动作也会盲目给出高分；而 FACT 的价值头对动作质量极其敏感。在 Stage 1 中，策略可以并行采样出 $N$ 组动作候选 $\{a^{(k)}\}_{k=1}^N$；在 Stage 2 中，这些动作分别注入动作插槽 $G$，价值头并行推演这批动作未来的进展得分，最终选出评分最高的那一个：




{% raw %}$$ a^{\star} = \arg\max_{a \in a^{(1:N)}} V_{\theta}(o_t, \ell, a) $${% endraw %}



整个过程完全依托模型内部自洽的因果头完成，不需要额外引入复杂的独立 Critic 网络或强化学习价值基准。

### 实验评测：仿真与真实双臂操作全景检验

为了检验这一套“动作先行、后果跟进”机制的真实含金量，作者团队在包含 50 个复杂任务的 RoboTwin 仿真平台，以及极具挑战性的真实世界双臂协同操作平台上展开了全面压测。

<img src="/images/2608.10232v1/x4.webp" alt="真实双臂实验任务" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

真实世界任务涵盖了双臂交接方块（Handover Block）、协同开启微波炉（Open Microwave）、双手抓放瓶子（Put Bottles）以及倾倒垃圾桶（Dustbin）等 5 个训练期见过的复杂技能，同时设置了 3 个在颜色、几何外观及指令表述上完全零样本泛化的未知变体（Unseen Tasks）。对比基线包括前沿的机器人基础模型 $\pi_0$、$\pi_{0.5}$，以及以视频生成为核心的 WAM 代表模型 Motus。

从实验数据来看，FACT 展现出了阶梯式的性能演进逻辑：

首先，在基础策略训练中引入未来视频预测作为辅助监督（Video Co-training），展现出了强大的表征正则化作用。在 RoboTwin 仿真测试中，未引入视频预测的纯策略模型平均成功率为 81.8%，而引入视频预测后迅速跃升至 85.6%；在真机测试中，缺失视频预测导致见识过的任务成功率从 82% 暴跌至 58%。这证明让网络同时学习物理世界如何推演，对于底层的连续动作生成具有极强的物理先验锚定效果。

其次，失败数据的因果注入带来了决定性的胜势。在注入由模型自身回滚产生的失败样本后，RoboTwin 任务成功率进一步刷新至 87.5%，逼平了计算成本极高的 Motus（87.8%），但在实际控制延迟上比后者快了近 3 倍；在真机见过的任务上，失败感知训练直接让成功率从 82% 攀升至 89%。如果进一步开启 $N=4$ 的候选动作价值打分（Candidate Scoring），成功率更是达到了惊人的 92%。

更为关键的是在面对未见过的物体和场景时，因果理解展现出的泛化韧性。在零样本泛化测试中，失败感知训练将基线成功率从 67% 提高到 77%，叠加动作打分后达到 82%。这一成绩极为逼近耗费了数万小时海量多机器人数据预训练的 $\pi_{0.5}$（85%），而 FACT 仅仅是在小规模任务演示与失败回滚上完成了端到端训练。

消融实验揭示了另一个反常识的关键细节：如果把收集到的失败动作强行当成负样本去模仿（即不关闭失败样本的动作损失），策略成功率会瞬间崩盘至 63%；而如果不在训练期引入失败轨迹，仅凭纯成功数据训练出来的价值头去进行 Candidate Scoring，测试成功率反而会从 82% 倒退至 79%。这无可辩驳地表明，**只有当模型真正见识过失败动作会导致何种恶果时，它的打分函数才具有挑选优质动作的能力，否则所谓的价值选择只是在未知区域的盲目自信**。

### 告别“幻觉未来”与自我纠错能力

研究团队针对世界模型最核心的病灶——“面对错误动作时的未来幻觉”，进行了深入的定性与定量分析。

<img src="/images/2608.10232v1/x5.webp" alt="未来预测对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在针对 512 个保留测试样本（严格按照 1:1 分布包含成功操作与失败操作）的图像重建测试中，纯成功数据训练的世界模型在面对失败动作时，图像 PSNR 仅有 16.71。如图所示，当真实机械臂执行了一个完全抓空的动作时，没有见过失败的模型在推演未来时，竟硬生生在画面里“幻觉”出了机械臂精准夹起物体的虚假图像。而在引入 FACT 的失败感知因果训练后，测试 PSNR 直接跳升至 20.89。面对同样那个失误的动作，FACT 准确预测出了夹爪落空、物体依然停留在原处的物理现实。

这种对因果关系的敏锐洞察，让模型在遭遇偶发扰动时表现出了非凡的鲁棒性。

<img src="/images/2608.10232v1/x7.webp" alt="任务进展曲线演变" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在记录机械臂抓取方块的实时轨迹中，可以清晰看到 FACT 内部预测的任务进展值（Progress Value）的波折起伏：当机械臂顺利逼近方块时，预测值平稳上扬；随后发生了一次意外的抓取滑脱，价值头立即敏锐地感知到了局势恶化，预测曲线瞬间发生断崖式下跌；然而策略并未因此瘫痪，机械臂迅速根据视觉反馈微调姿态并执行二次抓取，随着重新稳固握持物体，价值评分再次拉升。这种伴随动作因果动态修正世界预期的能力，正是机器人走向全自主鲁棒作业的核心基石。

同时，关于失败数据规模的消融实验（如下图所示）也给出了明晰的工程启示：在只注入 50% 失败数据时，各个任务的成功率就已经表现出了明显的正向爬坡，而在全量吸收失败回滚数据后，所有任务均达到了性能极值。这意味着现实中廉价、庞大且往往被直接废弃的失败日志，终于在世界模型时代找到了最完美的归宿。

<img src="/images/2608.10232v1/x6.webp" alt="失败数据比例扩展实验" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 走向具有真实因果知觉的具身基座

回顾 FACT 的设计哲学，其真正的贡献在于纠正了当前世界动作模型领域一种本末倒置的工程倾向：把视频生成当成动作生成的唯一驱动力，却忽视了动作本身才是改变物理世界的真正动因。

通过构建“先动作后未来”的因果时序，并利用教师强制掩码将动作模仿与后果预演彻底分轨，FACT 成功向学术界和工业界展示了一种极具扩展潜力的范式：

- 失败不仅不应该被丢弃，反而是构建无偏世界模型最稀缺的因果养料；

- 世界模型的价值不仅仅在于展示一个美好的未来，更在于客观预警一个危险或错误的未来；

- 动作空间的搜索与重排序，完全可以通过内在因果头以极小的代价完成自洽闭环。

尽管当前 FACT 依然受限于双阶段推断在启用多候选打分时带来的计算开销，但这种以因果推演为核心的世界动作架构，无疑为未来融合在线强化学习、DAgger 自主纠错以及大规模无监督负反馈探索，推开了一扇充满想象力的大门。
