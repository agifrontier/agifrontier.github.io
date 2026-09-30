---
layout: default
title: "ADRS：阿里提出自蒸馏奖励重塑，破解Agent长程强化学习信用分配难题"
description: "来自阿里巴巴和中国科学技术大学的研究团队在最新论文中提出了 ADRS（Agentic Reinforcement Learning with Self-Distilled Reward Shaping） 。"
arxiv_id: "2608.03223"
paper_published: "2026-08-04"
published_at: "2026-09-30T13:15:08.320273+08:00"
topics:
  - "AI Agent"
  - "强化学习"
tags:
  - "ADRS"
  - "Agentic Reinforcement Learning"
  - "Privileged skills"
  - "Reward-to-advantage construction"
  - "Self-Distilled Reward Shaping"
  - "Skill-free rollouts"
related_tutorials:
  - "online-process-reward-leanring-for-agentic-reinforcement-learning"
  - "reinforcement-learning"
  - "simpo-simple-preference-optimization-with-a-reference-free-reward"
  - "retrospective-harness-optimization-improving-llm-agents-via-self-preference-over-trajectory-roll"
seo_title: "ADRS：阿里提出自蒸馏奖励重塑，破解Agent长程强化学习信用分配难题"
---

<p class="paper-original-title" lang="en">Agentic Reinforcement Learning with Self-Distilled Reward Shaping</p>

<img src="/images/2608.03223v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在让大语言模型（LLM）学会像智能体（Agent）一样自主完成复杂任务的过程中，强化学习（RL）是当前最核心的技术引擎。无论是调用多步工具搜集证据、在电商网站中精准比对并下单，还是在具身控制环境中通过一系列动作探索解法，模型都需要在与环境的交互中不断试错与进化。

> ArXiv URL：https://arxiv.org/abs/2608.03223v1

然而，现阶段的多轮智能体强化学习普遍面临一个极其严苛的痛点：**稀疏的环境反馈与模糊的信用分配（Credit Assignment）**。在多数长程任务中，环境只能在整条轨迹结束时给出一个二元结果——成功或失败。面对数十甚至上百步交互、成千上万个 Token 的长序列，像 GRPO 或 GiGPO 这类基于组内相对优势的强化学习算法，虽然摆脱了对复杂 Critic 模型的依赖，却只能把最终的成败信号平均粗暴地分摊给轨迹中的每一个动作和 Token。模型知道最终成了或者挂了，却根本不知道究竟是第几步的哪一次决策立了大功，抑或是哪个隐蔽的操作埋下了祸根。

来自阿里巴巴和中国科学技术大学的研究团队在最新论文中提出了 **ADRS（Agentic Reinforcement Learning with Self-Distilled Reward Shaping）**。这项工作既不需要在推理时外挂复杂的系统提示，也不引入体积更庞大的独立验证模型（Process Reward Model, PRM），而是巧妙地通过训练期的“特权技能重塑（Privileged Self-Distillation）”，让模型自身充当打分老师，为采样出的无技能轨迹赋予稠密的 Token 级奖励。更关键的是，ADRS 解决了现有自蒸馏方法的三大底层结构缺陷：步间分数不具可比性、盲目自信与实际收益脱节、以及外挂辅助损失脱离强化学习主线。

<img src="/images/2608.03223v1/ADRS2.jpg" alt="ADRS 训练流程总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 从特权信息到信用分配：现有方案的症结何在

利用训练期独有的“特权信息”（Privileged Information）为无特权学生提供更稠密的监督，是机器学习中的经典思想。在多轮交互 Agent 的设定下，我们可以通过外挂面向特定任务的规程性技能（Procedural Skills，例如解题指南、子目标拆解或操作规范），指导策略模型的行为。

此前的一些代表性探索，如 OPSD 以及结合了强化学习的 SDAR、Skill-SD 等，尝试让同一个策略模型的冻结快照在输入包含特权技能文本的前提下，对学生模型在无特权条件下生成的轨迹 Token 进行重新打分（Rescoring）。这本质上是用“开卷”条件下的置信度去监督“闭卷”时的表现，同时保证推理部署阶段无需输入任何特权技能文本。

思路固然美妙，但在真正落地到复杂交互任务的强化学习中时，已有的特权蒸馏架构普遍遭遇了三道难以逾越的技术障碍：

其一是**打分刻度失准（Score Calibration）**。直接获取的教师 Log 概率（Log-probability）对上下文长度、交互所处的绝对位置以及动作词汇的先验分布高度敏感。比如在交互早期执行一个常见动作，其原生 Log 概率可能天然偏高；而在交互后期处理复杂长上下文时，即使动作完全正确，其数值也可能偏低。这就导致教师给出的原始打分在不同交互步骤之间根本无法直接横向对比。

其二是**盲目信任与收益脱钩（Reliability Estimation）**。老师打高分的决策，就一定能带来好的环境结果吗？答案是否定的。大语言模型容易对“语法顺畅、语言模式标准”的惯性操作给出极高的置信度，但这些看似规范的操作可能在具体环境下毫无进展，甚至直接导致任务失败；相反，某些探索性的非标准操作虽然老师给出的原生概率较低，最终却成功拿到了环境奖励。如果无条件信任老师的置信度，RL 训练很容易被带入歧途。

其三是**信用传导脱轨（Credit Integration）**。绝大多数现有方法都将特权蒸馏信号设计为一个独立的辅助损失函数（Auxiliary Loss），将其简单粗暴地与 GRPO 或 PPO 的策略梯度损失相加。这种外挂设计让特权监督绕开了强化学习本身的优势计算管道（Reward-to-Advantage Path），导致 Token 级别的局部更新与 GRPO/GiGPO 所构建的组内轨迹相对优势产生冲突，破坏了算法原本的优化方向。

ADRS 正是针对这三个核心冲突给出了统一的数学表述与模块化解决方案。

### 模块一：步内零和校准，剥离位置与长度偏差

为了让特权打分具备真实的相对指导价值，ADRS 改变了看待教师 Log 概率的视角：不再追求跨交互步骤的绝对打分高低，而是专注于在单次交互决策步骤内部，确定哪些 Token 相对更重要。

在具体实现中，学生模型在没有技能提示词的普通环境下完成采样并收集环境反馈。随后，同一个策略模型的冻结快照在拼接了任务匹配技能 $\rho(x)$ 的特权提示词下，对已固定的 Token 序列计算条件对数概率 $\ell^{T}_{s,j}$。为了消除跨步骤的全局偏移和方差波动，ADRS 在交互步骤 $s$ 的有效 Token 集合内执行均值中心化与尺度归一化：




{% raw %}$$b_{s} = \frac{1}{\lvert \mathcal{V}_{s} \rvert}\sum_{j\in\mathcal{V}_{s}}\ell^{T}_{s,j}, \quad q_{s,j} = \ell^{T}_{s,j} - b_{s}, \quad \widehat{q}_{s,j} = \frac{q_{s,j}}{\sigma_{q} + \epsilon}$${% endraw %}



这一操作直接带来了一个极其优美的数学性质：步内零和性（Stepwise Zero-Sum），即 $\sum_{j\in\mathcal{V}_{s}}\widehat{q}_{s,j} = 0$。

这意味着，特权打分只负责在步骤内部重新分配权重——那些高出平均水平的关键动作词元（例如关键的动词、工具调用参数）获得正向相对支持，而通用填充词元则被自然抑制。更重要的是，步内零和性确保了当我们将所有 Token 奖励累加回整条轨迹时，特权奖励不会改变轨迹总回报的原生排序，从而在根源上避免了打分偏差对原生环境回报的篡改。

### 模块二：TVA 可靠性门控，以真实回报约束特权置信度

在得到了校准后的步内相对偏好 $\widehat{q}_{s,j}$ 之后，ADRS 需要回答第二个问题：在当前的交互样本中，这个开卷打分的老师到底值不值得信任？

ADRS 提出了一个基于统计关联的门控机制——教师价值优势门控（Teacher Value Advantage, TVA）。系统首先计算每个样本单元中老师的平均置信度 $\bar{\ell}^{T}_{u}$，并通过 Sigmoid 函数将其转化为软划分权重 $\alpha_{u}$，代表该样本在组内更偏向高置信度还是低置信度。随后，TVA 统计两类样本对应的实际环境回报均值：




{% raw %}$$\mu_{g}^{+} = \frac{\sum_{u\in g}\alpha_{u}R_{u}}{\sum_{u\in g}\alpha_{u}+\epsilon}, \quad \mu_{g}^{-} = \frac{\sum_{u\in g}(1-\alpha_{u})R_{u}}{\sum_{u\in g}(1-\alpha_{u})+\epsilon}, \quad d_{g} = \mu_{g}^{+} - \mu_{g}^{-}$${% endraw %}



论文在理论上给出了严格的命题证明：差值 $d_{g}$ 在数学上等价于归一化的协方差：




{% raw %}$$d_{g} = \frac{\operatorname{Cov}_{g}(\alpha, R)}{\bar{\alpha}_{g}(1-\bar{\alpha}_{g})}$${% endraw %}



这一结论赋予了 TVA 极其清晰的物理意义：$d_{g}$ 的正负和大小，直接刻画了在当前样本组内，“老师觉得好的样本”与“环境实际给出高回报的样本”之间的统计一致性。当老师置信度高的轨迹确实更容易拿到高分时，$d_{g} > 0$，门控因子 $m_{u}$ 随之放大，允许更强的特权信号注入策略更新；反之，若老师高度偏好的样本在环境中频繁暴雷（例如出现了幻觉动作规范），协方差变为负值，门控则会主动将特权信号衰减甚至关停。

TVA 彻底改变了传统蒸馏算法对特权监督的盲信，让环境真实回报成为了调控教师权柄的最高准绳。

### 模块三：重塑先于优势，无缝融入原生 RL 管道

打分已校准，门控已就绪，特权信号应当以何种形式影响策略更新？这正是 ADRS 区别于 SDAR、OPSD 等先前方法最核心的工程与算法分水岭。

传统方法往往将蒸馏损失作为辅助项：$\mathcal{L}_{\mathrm{total}} = \mathcal{L}_{\mathrm{RL}} + \lambda \mathcal{L}_{\mathrm{distill}}$。但这种割裂的设计会导致策略在同一个 Token 处接收到来自强化学习优势和蒸馏梯度的双向拉扯，不仅难以调参，还频繁在复杂长程任务中破坏策略的单调提升。

ADRS 坚决主张“事前奖励重塑”（Pre-advantage Reward Shaping）。它先将门控调制后的特权信号转化为真实的 Token 奖励 $r^{T}_{s,j} = \eta m_{s,j}\widehat{q}_{s,j}$，并将其与底层环境奖励相加：




{% raw %}$$\widetilde{r}_{s,j} = r^{\mathrm{base}}_{s,j} + r^{T}_{s,j}$${% endraw %}



随后，ADRS 将整个计算完全交由后端的原生 RL 优势估计器（例如 GRPO 或 GiGPO）统一处理。优势的组内均值计算、方差归一化以及折扣机制，全部作用在重塑后的奖励之上。

论文通过一阶泰勒展开证明了这一设计的内在联系：在行为策略点附近且未发生截断时，ADRS 诱导的策略梯度与带有显式 Token 级辅助权重的梯度在局部等价，但其权重是严格在 GRPO/GiGPO 的组内归一化尺度下生成的。这种将外部特权信息降维为底层奖励输入、再统一升维至策略优势的设计，不仅维持了算法框架的整洁，更使得无论底层采用 GRPO 还是分步细粒度的 GiGPO，ADRS 都能即插即用，天然兼容。

### 复杂长程基准下的全面验证

为了检验 ADRS 分配信用的真实效果，作者在三类极具代表性的长程交互基准上展开了严格测试，涵盖文字具身控制任务 ALFWorld、多轮电商环境导航 WebShop，以及极度考验工具调用的长程检索问答 Search-based QA（基于 Search-R1 协议，覆盖 7 个 QA 数据集）。测试骨干模型包括 Qwen2.5-3B-Instruct、Qwen2.5-7B-Instruct 以及 Qwen3-1.7B-Instruct。

整个评估体系最为严苛的基线是对照了大量带“*”的特权基线——即允许在验证和推理阶段直接带入特权技能的强基线（如 Skill-Prompt、Skill-GRPO）。而 ADRS 在测试和实际部署时，**模型完全不包含任何技能提示词**。

在基准评测中，ADRS 展现出了极具说服力的提升轨迹：

- **长程决策性能显著领先**：在最具挑战性的具身任务 ALFWorld 上，纯环境驱动的 GRPO 和 GiGPO 经常由于多步动作的因果链条过长而陷入优化瓶颈；而引入了 ADRS 后，Qwen2.5-7B 在无特权推理的前提下，整体成功率大幅刷新，甚至全面压制了在验证期保留技能提示的强基线方法。

- **与先进蒸馏基线的对比优势**：对比此前将特权自蒸馏推至前沿的 SDAR 以及经典 OPSD，ADRS 在三个不同架构的模型规模下均取得了系统性优势。无论是在需要细致属性核对的 WebShop 上，还是在受限于 4 轮严格工具调用预算的 Search-based QA 中，ADRS 提炼出的 Token 级奖励都显著强于传统外挂辅助损失的方案。

除了通用的阶段性指标对比，消融实验更进一步揭示了 ADRS 为什么能走得更远：

首先是**长期训练的稳定性**。长程交互智能体在进行 300 步以上的强化学习更新时，极易因策略漂移或退化陷入崩溃。对比实验显示，外挂蒸馏损失的基线往往在 150 步左右达到虚假峰值后迅速恶化，而 ADRS 得益于步内零和约束与原生优势通道的融合，策略梯度在 300 步的长期优化中保持着稳定的爬坡曲线，最终检查点与平均胜率均保持在高位。

其次是**数据利用效率与泛化性**。在仅使用 20%、50% 交互训练数据的缩减设定下，ADRS 凭借稠密化的高质量 Token 信用分配，拉开了与纯稀疏奖励 RL 之间更大的差距；而在迁移至未见过的全新任务环境时，经由 ADRS 训练出的策略模型同样表现出远优于基线模型的行为鲁棒性。

### 智能体强化学习的范式思考

回顾大语言模型近两年的技术演进，强化学习正从最初简单的全序列人类偏好对齐（RLHF），迈向真正具有探索、规划与纠错能力的自主智能体训练。在这一过程中，稀疏的环境成败奖励早已成为制约模型泛化能力的深水区。

过去业内对于稠密奖励的探索，很大程度上押注在过程奖励模型（PRM）上。然而，为复杂环境专门训练一个高质量且不产生奖励作弊（Reward Hacking）的 PRM，其工程和数据成本往往极其高昂。

ADRS 的价值在于指明了另一条兼具理论优雅与工程落地的路径：**模型自身就蕴藏着自我纠偏的潜能**。在训练期，通过简单引入无需参数更新的过程化规程文本，让同一个模型充当“开卷裁判”；再通过零和校准剥离位置偏置、利用 TVA 门控以真实环境结果为裁判把关，最终将其收敛为最底层、最原生的 Token 奖励重塑。这种设计既规避了额外判别模型的开销，又让轻量级模型在推理时完全甩掉特权包袱，展现出更高的自主决策水准。

对于正在探索通过强化学习提升多轮 Agent 交互上限、深陷稀疏反馈泥潭的研究者与工程师而言，ADRS 提供了一套兼具严密数学证明与即插即用特性的极佳范式。随着开源社区在自主 Agent 探索领域的不断深入，这类将外部先验无痛内化为原生 RL 信用的技术方案，无疑值得被更多前沿系统纳入核心架构之中。
