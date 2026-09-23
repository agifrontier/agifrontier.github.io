---
layout: default
title: "FACTOR：不是全轨迹广播，而是先定动作信用再分Token"
description: "近期来自清华、北大、复旦、浙大等机构的研究团队提出了一种解耦机制 FACTOR （Factorizing Action Credit and Token Responsibility），直击当前多轮 Agent 强化的隐性缺陷。"
arxiv_id: "2608.07118"
paper_published: "2026-08-07"
published_at: "2026-09-15T13:15:08.126145+08:00"
topics:
  - "AI Agent"
  - "强化学习"
tags:
  - "Action-mean reduction"
  - "Action-to-Token Allocation"
  - "Checkpoint-calibrated TD residuals"
  - "FACTOR"
  - "Hindsight token allocation"
  - "Per-action normalization"
related_tutorials:
  - "a-practitioners-guide-to-multi-turn-agentic-reinforcement-learning"
  - "dler-doing-length-penalty-right-incentivizing-more-intelligence-per-token-via-re"
  - "training-task-reasoning-llm-agents-for-multi-turn-task-planning-via-single-turn-"
  - "ui-tars-2-technical-report-advancing-gui-agent-with-multi-turn-reinforcement-lea"
seo_title: "FACTOR：不是全轨迹广播，而是先定动作信用再分Token"
---

<p class="paper-original-title" lang="en">How Much, Then Where: Credit-Conserving Action-to-Token Allocation for Multi-Turn Agent Reinforcement Learning</p>

<img src="/images/2608.07118v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在多轮交互智能体（Agent）的强化学习训练中，一个根本性的矛盾始终存在：环境给出的反馈往往是极度稀疏的整条轨迹结果（例如任务成功或失败的标量奖励），但策略模型（LLM）的参数更新却必须逐个落实到每一个自回归生成的 Token 上。这种从“整条轨迹”到“单个动作”，再到“动作内部 Token”的巨大粒度错配，直接构成了多轮 Agent 信用分配（Credit Assignment）的核心挑战。

> ArXiv URL：https://arxiv.org/abs/2608.07118v1

主流的轨迹级强化学习算法（如 GRPO）选择用最简单粗暴的方式绕过这个难题：计算出一个全轨迹优势标量（Advantage），然后不加区分地广播给这趟轨迹里的每一个时间步和每一个 Token。近期来自清华、北大、复旦、浙大等机构的研究团队提出了一种解耦机制 **FACTOR**（Factorizing Action Credit and Token Responsibility），直击当前多轮 Agent 强化的隐性缺陷。该研究指出了现有微观信用分配方案中长期被忽视的两大失效机制——Token 长度引起的权重漂移与事后教师引导带来的尺度失真，并通过“先定动作信用、再分内部 Token”的严格守恒框架，在 ALFWorld、WebShop 和复杂的长程环境 ScienceWorld 上实现了稳定的性能跨越。

### 耦合的双重陷阱：Token 长度与教师漂移

要理解 FACTOR 的设计动机，必须先看清现有方法在试图细化信用分配时踩进了怎样的误区。目前学术界对这一问题的探索大致分化为两个割裂的流派。

一派聚焦于“时间步信用分配”（Temporal Credit Assignment），试图回答“每个动作到底值多少分”。这类工作通常借助状态分支、子轨迹 Rollout 或是蒙特卡洛值估计，为轨迹中的每个动作赋予一个独立的优势值。但这类研究通常止步于标量动作优势，一旦进入策略梯度反向传播，仍然要依赖某种平均化操作将动作标量打散到 Token 上。

另一派则聚焦于“动作内 Token 调节”（Intra-action Allocation），试图回答“信用应该落在动作内部的哪些关键 Token 上”。例如近期的 SERL 等工作，引入了一个拥有后验反馈（Hindsight Feedback）特权的教师模型，通过对比教师与学生在当前动作 Token 上的对数似然差距（Likelihood Gap），挑选出真正促成交互或导致失误的关键 Token 进行重点奖惩。

这两派方法看似互补，实则暗藏致命的尺度耦合。当我们在动作内部用后验教师对 Token 赋予不同的乘性权重时，如果这些权重在动作内部的均值不严格等于 1，就会悄然篡改整个动作的原本信用幅度。举例而言，当教师模型对某个动作的 Token 缺乏置信度时，所有 Token 的调整因子可能系统性偏小，导致这个动作在最终 Policy Loss 中的贡献被无意识地压缩；反之，若教师过度自信，动作的整体更新步长就会被成倍放大。动作在环境里实际发挥的客观价值，被教师模型的主观置信度严重污染。

更隐蔽的缺陷在于损失函数中的规约方式（Reduction）。在常见的全局 Token 平均（Token-Mean Reduction）下，动作对总损失的贡献与其包含的 Token 数量 $L_t$ 成正比。这意味着两个在环境语义上同等重要的动作，仅仅因为其中一个表述更冗长、生成的 Token 更多，就会在策略更新中占据数倍的梯度权重。环境交互的因果归因，最终不可避免地被语言表达的表层长度所绑架。

### 先定额度，再做分配：FACTOR 的双层架构

针对上述耦合失效，FACTOR 提出了清晰的原则：**先确定每个动作应当承担的整体信用额度，再决定这笔额度在动作内部各 Token 间如何二次分配，且分配过程必须严格遵守“信用守恒”，绝不篡改动作原有的总额与符号。**

<img src="/images/2608.07118v1/overview2.webp" alt="FACTOR 双层信用分配框架图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

如上图所示，FACTOR 将整个流水线解耦为三大核心组件与一个配套的损失函数形式：

第一步是**基于检查点校准的时序差分动作信用（TAC, TD Action Credit）**。为了解决单轨迹标量无法区分各个动作好坏的问题，FACTOR 从生成轨迹中恢复出稀疏的中间状态检查点，并在冻结的行为策略下采样少量无梯度的推理延续轨迹（Inference-only Continuations）。利用延续轨迹的蒙特卡洛回报训练轻量级价值头后，算法计算单步 TD 残差：




{% raw %}$$A_{i,t}^\star = r_{i,t} + \widetilde{V}_{\bar{\phi}}(x_{i,t+1}) - \widetilde{V}_{\bar{\phi}}(x_{i,t})$${% endraw %}



特别巧妙的是，FACTOR 通过将初始状态价值绑定为轨迹基线 $b_i$、终止状态价值设为 0，使得所有动作的 TD 信用沿时间展开求和后，恰好发生望远镜求和相消（Telescoping Sum）：




{% raw %}$$\sum_{t=0}^{T_i-1} A_{i,t}^\star = G_i - b_i = A_i^\mathrm{seq}$${% endraw %}



这意味着，所有动作分到的局部信用之和严格等于轨迹总优势，既实现了细粒度的时间步分解，又完全保留了原始优势的全局尺度。

第二步是**后验 Token 责任分配（HTA, Hindsight Token Allocation）**。确定了动作标量 $A_t^\star$ 后，模型需要找出动作内部哪些 Token 是核心。FACTOR 利用带有环境执行后观察结果的后验教师模型，计算其与当前策略在 Token $j$ 上的对数似然差距 $\Delta_{t,j}$。为了防止教师越权反转环境判断，FACTOR 将似然差距与动作信用的符号对齐：




{% raw %}$$s_{t,j} = \mathrm{sgn}(A_t^\star) \cdot \mathrm{clip}(\Delta_{t,j}, -d, d)$${% endraw %}



当动作被环境判定为正向贡献（$A_t^\star > 0$）时，教师认可度越高的 Token 分配到越多的奖励；当动作为负向贡献（$A_t^\star < 0$）时，教师认可度低的错因 Token 则承担更大的惩罚。

第三步是**动作均值守恒归一化（APM, Per-Action Mean Preservation）**。这是阻断教师漂移的核心关卡。FACTOR 将各 Token 的得分映射为一个非负分布 $\rho_{t,j}$，并通过退火权重与均匀分布平滑混合，最终定义每个 Token 的更新系数为：




{% raw %}$$C_{t,j}^\mathrm{FACTOR} = L_t \rho_{t,j} A_t^\star$${% endraw %}



由于 $\sum_j \rho_{t,j} = 1$，该动作内部所有 Token 系数的平均值严格等于 $\frac{1}{L_t} \sum_j C_{t,j} = A_t^\star$。后验教师的介入被严格限定在“调整动作内部各 Token 的权重相对比例”上，非负性保证了没有任何 Token 的正负号会被教师意外翻转，而均值守恒则保证了动作原本的信用总额毫发无损。

### 理论闭环：消除长度依赖的 Action-Mean 损失

微观分配上的均值守恒，只有在目标损失函数的配合下才能真正起效。如果下游优化依然采用全局除以 Token 总数的常规做法，不同长度动作的有效权重依然无法对齐。

为此，FACTOR 配套采用了**动作级平均规约（Action-Mean Reduction）**。其策略梯度损失在结构上先在每个动作内部对所有 Token 求平均，再在轨迹维度对所有动作求平均：




{% raw %}$$\mathcal{L}_\mathrm{RL}^\mathrm{act} = -\frac{1}{\sum_i T_i} \sum_{i,t} \frac{1}{L_{i,t}} \sum_j \min\Bigl( q_{i,t,j} C_{i,t,j}, \operatorname{clip}_\epsilon(q_{i,t,j}) C_{i,t,j} \Bigr)$${% endraw %}



在行为策略采样点处（即重要性采样比率 $q_{i,t,j} = 1$），在 PPO 截断生效之前，内层括号关于 Token 的平均项发生直接简化：




{% raw %}$$\frac{1}{L_{i,t}} \sum_j C_{i,t,j} = \frac{1}{L_{i,t}} \sum_j L_{i,t} \rho_{i,t,j} A_{i,t}^\star = A_{i,t}^\star \sum_j \rho_{i,t,j} = A_{i,t}^\star$${% endraw %}



这一推导揭示了一个极具价值的理论性质：在未截断状态下，任意动作对总梯度的有效标量贡献恰好完全等同于其 TD 动作信用 $A_{i,t}^\star$，与其表层文本长度 $L_{i,t}$ 彻底脱钩。一个由 3 个 Token 构成的精炼动作（如 `go east`）和一个包含 40 个 Token 的复杂调用，只要其环境 TD 信用相同，对模型参数更新的初始推动力度就完全对等。

### 实验评测：长视距与多模型的稳健泛化

评测实验在三个代表性的多轮交互基准上展开：涵盖具身家居指令的 ALFWorld（未见测试集 134 个任务）、涉及电商搜索与点击的 WebShop（1000 个留出指令），以及需要数十步物理推演的高难度长程基准 ScienceWorld（540 个评测 Episode）。

主干模型采用 Qwen2.5-7B-Instruct，所有方法在相同的计算框架下严格控制随机种子、数据切分与训练轮次。在严格对齐超参的受控对比中，FACTOR 在三个任务上全面领先强基线 GRPO 和经过对齐复现的 SERL（SERL-Repro）。最值得关注的规律在于**任务视距与提升幅度的正相关性**：在交互步数较短的 WebShop 上，FACTOR 相比最优基线获得了约 1.5 到 2 个百分点的稳定提升；而在视距最长、单任务交互上限达到 50 步、状态转换极其繁复的 ScienceWorld 上，FACTOR 相比基线取得了最为显著的飞跃。

为了验证该机制是否过度依赖特定的参数微调，研究团队将 7B 模型上固定的超参数直接迁移至更大参数量的 Qwen2.5-14B，以及架构不同的 Llama-3.1-8B。跨模型族实验显示，在未做任何超参重新搜索的前提下，FACTOR 在两个新主干上的 ALFWorld、WebShop 和 ScienceWorld 中均全面压制对比基线。对于 Llama-3.1-8B，三个环境的绝对成功率分别提升了 2.1、1.9 和 2.9 个百分点；对于 14B 模型，在 ScienceWorld 上更是直接斩获了 2.7 个百分点的增益。这种在长程环境上的高度一致性表明，双层信用解耦所带来的增益并非特定架构下的局部优化，而是触及了多轮交互更新的本质规律。

### 机制诊断与消融：性能飞跃的真正来源

这种性能优势究竟来自合理的数学解耦，还是单纯因为中间采样引入了更多计算量？消融实验与细粒度诊断给出了直接的定量回答。

<img src="/images/2608.07118v1/mechanism_diagnostics.webp" alt="机制诊断与微观证据" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

首先，通过剥离各模块（Table 4 的数据结论）可以发现，**TAC（动作级 TD 信用）是提升的最核心引擎**。一旦去掉 TAC、退回全局轨迹优势，模型在三个环境上的成功率分别大幅下跌 2.1、1.9 和 3.5 个百分点，平均性能损耗达 2.5 个百分点。而去掉负责动作内微观细化的 HTA 时，性能在短程任务上保持平稳，但在长程与复杂语义环境（WebShop 和 ScienceWorld）中分别下跌 1.6 和 2.1 个百分点。这证明了动作间的时间归因是全局骨架，而动作内的 Token 聚焦则在复杂场景下提供了关键的补充增益。

更具说服力的是针对机制假设的直接探针实验，如上图所示：

在上图 Panel A 中，研究人员监控了训练期动作平均 Token 乘数与 1 的偏离程度。在缺乏均值守恒约束的 SERL 中，动作平均系数偏离中位数达到 6.7%，甚至有超过 28% 的动作偏离幅度超过 10%，证实了教师置信度漂移并不是偶发的边缘个案；而在引入 APM 的 FACTOR 中，所有测量动作的偏离幅度被严格控制在 $4 \times 10^{-7}$ 以下，从机制上彻底锁死了信用尺度的漂移。

在上图 Panel B 中，TAC 计算的标量信用 $A_t^\star$ 与高采样量蒙特卡洛真实估计进行了对齐检验，两者在正负号判断上达到了 84.2% 的一致性，并在训练后期上升至 89.2%，验证了轻量级价值头足以在训练过程中提供高质量的方向判断。

在上图 Panel C 的 Token 扰动因果测试中，研究人员对比了替换高分配权重 Token（高 $\rho$）与低分配权重 Token（低 $\rho$）对环境回报的实际影响。对于正向动作，破坏高权重 Token 导致的任务回报下降幅度比破坏低权重 Token 高出 13.7 个百分点；而对于负向动作，修正高权重 Token 带来的回报恢复比修正低权重 Token 高出 7.9 个百分点。这直接证明了 HTA 分配的高权重确实高度锚定了决定交互成败的核心 Token。

为了彻底排除“算力红利”的质疑，研究团队设计了显存与 GPU 耗时完全对齐的强基线 SERL-Extended。即使赋予 SERL-Extended 额外的优化步数和采样开销，FACTOR 依然在各环境上保持 1.6 到 3.6 个百分点的显著优势。同时，在延续采样预算的敏感度测试中（Table 5），仅使用每个检查点采样 4 条延续轨迹的轻量配置（$2 \times 4$）就达到了性能饱和拐点，以 60% 的交互成本捕获了 94% 的理论最大增益。

### 信用分配新视角的启示与代价

回顾过去一段时间的强化学习演进，业界的研究重心长期停留在“如何构造更强的价值模型”或“如何设计更复杂的奖励函数”上。FACTOR 带来的重要启示在于：**优化目标中的微观代数结构（Algebraic Structure）与信用守恒，其重要性丝毫不亚于外部奖励的质量**。如果不主动处理损失函数规约引入的隐式长度加权，模型在训练时就会在不知不觉中更倾向于“把一句话说得更长”，哪怕长表述并不能带来更多环境信息；如果不显式限制辅助教师的干预边界，原本由真实环境反馈决定的策略梯度就会被后验先验悄悄稀释。

当然，FACTOR 的机制并非毫无代价。正如作者在文中所指出的，TAC 的核心前提是环境必须具备**状态可恢复性（Restorable State）**，即系统能够回溯到中间某个特定交互节点并向后分支执行无梯度采样。在桌面仿真器或沙箱环境（如 ALFWorld、ScienceWorld）中，这种快照机制成本可控；但在一些真实的外部 API 或无法回滚的现实交互场景下，状态分支采样的实现门槛依然较高。

总体而言，FACTOR 确立了一种极具启发性的多轮 Agent 强化范式。它告诉我们，从稀疏奖励走向稠密更新，并不意味着要在混乱的经验空间里随意插值，而是应当在数学上把“宏观定额”与“微观分配”严谨地割裂开来。随着大模型在深层次工具调用与自主规划中的应用越来越深，这种追求尺度不变性与因果守恒的底层算法设计，正在成为 Agent 训练走向精密化的关键基石。
