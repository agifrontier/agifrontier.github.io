---
layout: default
title: "HyGAE：统一Critic兼顾Token与轮次，多轮VLM决策胜率达到91%"
description: "研究者从理论上严格证明了 Token 级与轮次级强化学习在策略梯度目标上的数学等价性，并在此基础上提出了全新的 HyGAE 框架。该框架不仅巧妙融合了两者的优势估计，还通过理论证明消解了训练两个独立价值网络的算力负担。"
arxiv_id: "2607.23605"
paper_published: "2026-07-26"
published_at: "2026-09-12T13:15:08.779692+08:00"
topics:
  - "AI Agent"
  - "强化学习"
tags:
  - "Actor-Critic"
  - "Agentic RL"
  - "HyGAE"
  - "Hybrid Advantage Estimation"
  - "Multi-turn decision-making"
  - "Token-wise optimization"
related_tutorials:
  - "agentgym-rl-training-llm-agents-for-long-horizon-decision-making-through-multi-t"
  - "didpo-diff-in-diff-policy-optimization-for-coding-agent-training"
  - "a-practitioners-guide-to-multi-turn-agentic-reinforcement-learning"
  - "natural-language-actor-critic-scalable-off-policy-learning-in-language-space"
---

<p class="paper-original-title" lang="en">Hybrid Advantage Estimation with Unified Critic for VLM Agentic Reinforcement Learning</p>

<img src="/images/2607.23605v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在多模态大模型（VLM）从“被动看图说话”走向“自主与环境交互”的进程中，多轮序列决策（Multi-turn Decision-Making）能力成了衡量智能体成熟度的核心分水岭。当前的视觉语言智能体虽然在单轮视觉问答或图文理解上表现优异，但在复杂、动态的多轮交互环境中，往往会陷入一种顽固的“近视状态”（Nearsightedness）：模型倾向于仅仅依赖当前最新一帧的局部观察进行推理，完全无视先前的操作轨迹和环境的失败反馈。这导致智能体在遭遇阻碍时，容易陷入机械重复无效动作的死循环，多轮决策退化成了互不相连的单轮零散生成，任务的最终完成也沦为撞大运式的随机尝试。

> ArXiv URL：https://arxiv.org/abs/2607.23605v1

这一缺陷的根本根源，在于现有训练范式与多轮交互本质之间的严重错配。为了让智能体学会在交互中根据反馈动态调整策略，学术界开始转向端到端的多轮强化学习（Agentic RL）。但在建模这种多轮轨迹时，算法设计者长期面临两难权衡：究竟该把强化学习建立在“Token 级别”还是“轮次级别（Turn-wise）”？前者将生成的每一个 Token 视为独立动作，为自回归语言生成提供了极细粒度的微观引导，却往往由于稀疏奖励而难以把环境反馈准确传导给宏观动作；后者将模型在整轮对话中的输出视为一个统一动作，能够充分利用轮次间的中间反馈进行宏观信用分配（Credit Assignment），却又粗暴地忽略了单轮内部语言推理链条中不同 Token 的贡献差异。

来自阿卜杜拉国王科技大学（KAUST）的研究团队在论文《Hybrid Advantage Estimation with Unified Critic for VLM Agentic Reinforcement Learning》中给出了破解之道。研究者从理论上严格证明了 Token 级与轮次级强化学习在策略梯度目标上的数学等价性，并在此基础上提出了全新的 **HyGAE** 框架。该框架不仅巧妙融合了两者的优势估计，还通过理论证明消解了训练两个独立价值网络的算力负担，仅凭单一统一 Critic（Unified Critic）就实现了对 Token 级与轮次级价值的兼顾估计。实验显示，HyGAE 在 5 个多轮视觉决策基准任务上取得了高达 91% 的平均成功率，较现有主流方法提升了超过 10 个百分点，彻底扭转了小参数开源 VLM 在多轮环境中的失灵困境。

<img src="/images/2607.23605v1/overview.webp" alt="HyGAE 框架概览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 撕裂的两端：微观生成与宏观决策的建模鸿沟

要理解 HyGAE 的创新，首先需要厘清现有 VLM 多轮强化学习在建模层面的底层矛盾。在多轮智能体环境中，交互过程通常被表述为部分可观测马尔可夫决策过程（POMDP）。智能体在每一轮接收到视觉与文本组合的观测，随后自回归生成一段包含思考链推理和环境指令的 Token 序列，环境执行该指令后返回下一步的观测与即时奖励。

这一交互过程天然具备多尺度特性。如果采用 Token 级建模，系统将自回归生成的每一个 Token $a_t^i$ 都视作一个动作。其好处在于，它完全贴合自回归语言模型的因果解码机制，能细致约束生成概率并优化语言流畅度；但其致命弱点在于奖励极其稀疏。在绝大多数环境中，只有当一整轮文本完全生成且由环境执行后，才会得到一个环境奖励 $\boldsymbol{r}_t$，单轮内部的前序推理 Token 获得的奖励全为零。这种微观建模在面对长序列时极易产生巨大的累积方差，导致中间推理过程的信用分配失焦。

反之，若采用轮次级建模，系统将整轮生成的完整 Token 序列 $\boldsymbol{a}_t$ 打包成一个宏观动作。此时，环境反馈直接作用于该宏观动作，使得跨轮次的全局规划和中间反馈利用变得极其直接且低方差。然而，这种抽象却以牺牲微观表征为代价：轮次级优化默认将外部反馈平均摊派给本轮的所有 Token，完全无法区分“关键决策 Token”与“冗余过渡 Token”，极大地削弱了模型在细粒度语言逻辑上的纠错能力。

以往的研究要么非此即彼地倒向某一端，要么采用结构极其复杂的层次化强化学习（Hierarchical RL）来割裂处理两层规划。后一种做法不仅带来了双倍甚至数倍的参数显存开销与采样负担，更缺乏对两个层级之间数学联系的本质挖掘。这引发了一个核心问题：Token 级与轮次级强化学习，在数学上真的水火不容吗？

### 理论统一：策略梯度的一致性与统一Critic的诞生

研究团队通过深入的理论推导给出了否定回答。他们首先证明，尽管 Token 级与轮次级优化在形式上尺度不同，但只要满足特定的折现约束，二者在策略梯度层面具有内在的一致性。

在单轮动作 $\boldsymbol{a}_t$ 内部，整轮序列的对数概率生成天然等于各个 Token 局部条件概率对数之和：




{% raw %}$$ \log\pi_{\theta}(\boldsymbol{a}_t\vert{}\boldsymbol{\tau}_t) = \sum_{i=0}^{I_t-1}\log\pi_{\theta}\left(a_t^i\vert{}\tau_t^i\right) $${% endraw %}



基于这一自回归性质，本文构建了一个将轮次级优势函数 $\boldsymbol{A}^{\pi_{\theta}}(\boldsymbol{\tau}_t,\boldsymbol{a}_t)$ 与 Token 级优势函数 $A^{\pi_{\theta}}(\tau_t^i,a_t^i)$ 线性插值的混合替代目标函数：




{% raw %}$$ L^{\mathrm{hybrid}}(\theta)=\mathbb{E}\left[\sum_{t=0}^{T-1}\sum_{i=0}^{I_t-1}\log\pi_{\theta}\left(a_t^i\vert{}\tau_t^i\right)\left(\alpha\boldsymbol{A}^{\pi_{\theta}}(\boldsymbol{\tau}_t,\boldsymbol{a}_t)+(1-\alpha)A^{\pi_{\theta}}\left(\tau_t^i,a_t^i\right)\right)\right] $${% endraw %}



论文的 **Theorem 1** 明确指出：只要令轮次级折现因子与 Token 级折现因子满足 $\boldsymbol{\gamma}_t = \gamma^{I_t}$（其中 $I_t$ 为第 $t$ 轮生成的 Token 总长度），那么对于任意混合系数 $\alpha \in [0, 1]$，混合目标函数的策略梯度与原始强化学习目标 $J(\theta)$、纯轮次级目标 $L^{\mathrm{turn}}(\theta)$ 以及纯 Token 级目标 $L^{\mathrm{token}}(\theta)$ 完全恒等：




{% raw %}$$ \nabla J(\theta) = \nabla L^{\mathrm{hybrid}}(\theta) = \nabla L^{\mathrm{turn}}(\theta) = \nabla L^{\mathrm{token}}(\theta) $${% endraw %}



这一等价性定理从理论上扫清了混合优化的障碍，表明混合估计并没有改变原始优化问题的最优解方向，而是提供了一个在有限样本采样下调整估计特性的自由度。

然而，将这一理论付诸实践的最大拦路虎是价值函数（Critic）的计算。在传统的 Actor-Critic 架构下，如果要分别计算轮次级和 Token 级的优势值，通常需要分别维护一个针对轮次状态的评判网络 $\boldsymbol{\widehat{V}}_{\psi}$ 和一个针对每个 Token 前缀的评判网络 $\widehat{V}_{\psi}$。对于体量庞大的多模态模型而言，维护两套独立的 Critic 网络会直接使显存占用和训练反向传播开销成倍膨胀，这在工程上极不划算。

<img src="/images/2607.23605v1/algorithm.webp" alt="HyGAE 算法流程与统一价值训练架构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了解决该瓶颈，本文提出了 **Theorem 2**，揭示了价值函数的一致性规律：在同样的折现约束 $\boldsymbol{\gamma}_t = \gamma^{I_t}$ 下，轮次状态的真实价值与该轮最后一个 Token 处的真实 Token 级状态价值完全相等，即：




{% raw %}$$ \boldsymbol{V}^{\pi}(\boldsymbol{\tau}_t) = V^{\pi}(\tau_t^{I_t}) $${% endraw %}



这意味着，我们完全不需要两套 Critic 模型，只需训练一个精细到 Token 级别的统一价值网络 $\widehat{V}_{\psi}$。在计算轮次级优势时，智能体只需将对应轮次末尾的状态价值取出来充当轮次价值；而在计算中间 Token 的宏观价值时，则可以通过折现奖励的反向广播来复用这一统一 Critic 的预测。这一设计不仅大幅简化了系统架构，还将原本复杂的层次化多模态 Critic 开销压缩至单一网络，显著提升了训练效率。

### 偏差与方差的数学平衡：为什么混合优势更好？

统一了梯度目标并精简了 Critic 结构后，随之而来的关键疑问是：既然纯 Token 级与纯轮次级的梯度期望相同，为何非要费力将它们混合在一起？答案在于有限样本估计下的**偏差与方差权衡（Bias-Variance Tradeoff）**。

在实际使用广义优势估计（GAE）时，估算出的优势值并非真实值，而是受到价值网络逼近误差与采样随机性双重影响的估计量。论文在理论分析部分对两类估计器的性质进行了严格刻画：

1. **偏差对比（Theorem 3）**：假定价值网络与真实价值的最大逼近误差为 $e_{\max}$，轮次级优势估计的偏差上界由 $b_{\mathrm{turn}}$ 约束，Token 级优势估计的偏差上界由 $b_{\mathrm{token}}$ 约束。推导表明 $b_{\mathrm{turn}} \leq b_{\mathrm{token}}$。这是因为轮次级估计跨越了多个 Token 的真实转移步长，有效折现因子实际上是更小的 $\gamma^I$，使得来自 Critic 价值模型的逼近误差被更快地衰减。换言之，**轮次级估计具有更低的偏差**。

2. **方差对比（Theorem 4）**：在考察环境即时奖励与时序差分误差的方差时，轮次级优势估计的方差上界 $v_{\mathrm{turn}}$ 显著大于 Token 级估计的方差上界 $v_{\mathrm{token}}$。其数学本质在于，轮次级优势需要跨越更长的时间步，环境随机性和长链条生成的波动在单轮内层层累积，放大了方差。也就是说，**Token 级估计具有更低的方差**。

数学分析揭示了一组精妙的对立统一：轮次级估计“偏差更小、但方差偏大”，而 Token 级估计“方差极低、但受限于 Critic 误差导致的系统性偏差更大”。HyGAE 通过凸组合参数 $\alpha$ 将二者线性融合：




{% raw %}$$ \widehat{\mathsf{A}}_t^i = \alpha\widehat{\boldsymbol{A}}_t + (1-\alpha)\widehat{A}_t^i $${% endraw %}



通过控制混合权重 $\alpha \in [0, 1]$，算法得以在低偏差与低方差之间取得优雅的折中，在保持长期宏观因果感知的同时，利用局部 Token 梯度的低方差特性稳住语言模型的更新轨迹。

### 精确的回报构造：算法稳定的生命线

在深度强化学习中，理论形式的优雅并不总能自动转化为训练的平稳收敛。研究人员在深入剖析 HyGAE 时发现，将混合优势落地到 PPO 训练中，最关键的决定性因素在于**混合回报（Mixed Return）的解析形式必须保持数学上的极致严谨**。

为了更新统一的 Critic 模型，需要构造时序差分的目标回报 $\widehat{\mathsf{G}}_t^i$。研究团队定义：




{% raw %}$$ \widehat{G}_t^i = \widehat{A}_t^i + \widehat{V}_{\psi}(\tau_t^i) $${% endraw %}






{% raw %}$$ \widehat{\boldsymbol{G}}_t^i = \widehat{\boldsymbol{A}}_t + \sum_{k=i+1}^{I_t}\gamma^{k-i}r_k + \widehat{V}_{\psi}(\boldsymbol{\tau}_t) $${% endraw %}






{% raw %}$$ \widehat{\mathsf{G}}_t^i = \alpha\widehat{\boldsymbol{G}}_t^i + (1-\alpha)\widehat{G}_t^i $${% endraw %}



这一公式中的细节极为关键。在轮次级回报向内部 Token 广播时，必须严格补偿从当前 Token $i$ 到轮末这一段区间内已发生的折现即时奖励 $\sum_{k=i+1}^{I_t}\gamma^{k-i}r_k$。

<img src="/images/2607.23605v1/training_curves.webp" alt="HyGAE 关键设计因子的消融实验" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

如上图左侧曲线所示，如果忽略这一严格推导，直接采用粗暴的启发式做法——例如将轮次末尾的回报均匀广播给所有 Token，或者像部分简化算法那样彻底丢弃 Critic、退化为无基准线（REINFORCE）更新——训练曲线会迅速陷入崩溃，胜率断崖式归零。这是因为在语言模型强化学习中，Critic 本身就极难训练；如果目标回报中包含与当前 Token 状态不匹配的时序漂移，不仅无法降低方差，反而会向 Critic 注入剧烈的伪梯度，瞬间摧毁原本脆弱的价值基准线。

此外，消融实验还澄清了工程实现中的另外两个关键疑问：

* 首先，对于 KL 散度惩罚等微小的 Token 级细粒度奖励，严格来说应当求和累加至轮次奖励中。但实验表明，是否将这些微小项累加进轮次级奖励，对整体训练效果的影响微乎其微。这极大简化了多奖励组件下的数据处理管线。

* 其次，在为广播后的轮次级优势选取基准价值时，实验证实使用整轮初始的轮次级价值作为常数基线最为稳健，相比使用单 Token 预测价值能更好地规避中间步的预测噪声。

在偏差与方差的数值对比实验中，传统 PPO 在 Sokoban 任务上的成功率呈现出 $0.5797 \pm 0.2105$ 的剧烈波动，而 HyGAE 则达到了 $0.8222 \pm 0.0711$。方差缩小近三分之二的实测数据，直接印证了理论推导对实际训练稳定性的强大支撑。

### 实验评测：从小模型折戟到 91% 胜率跃升

为了验证 HyGAE 在真实多轮环境中的通用效能，论文在包含视觉推箱子（Sokoban）、冰面寻路（FrozenLake）、基础技能操作（Primitive Skill）等在内的 5 个主流决策基准上展开了系统性评测。这些环境要求模型不仅要读懂视觉观测图像，还要根据多轮连续交互反馈生成结构化或自然语言形式的动作指令。

评测结果揭示了一个引人注目的技术鸿沟：

* 未经多轮 RL 针对性优化的开源基座模型（如 Qwen2.5-VL-3B、Qwen3-VL-4B 等）在这些任务中表现惨淡，多轮交互的平均成功率普遍难以逾越 10% 的阈值；即使是参数更新、架构迭代带来的细微提升，也往往在不同任务间顾此失彼。

* 相比之下，具备海量系统提示与闭源对齐经验的商用模型（如 GPT-4o、Claude-3.5-Sonnet 等）平均胜率能够达到 50% 以上，但在面对极其严格的因果逻辑和长序列反馈时，依然偶有失手。

当引入 HyGAE 对开源小模型进行端到端多轮强化学习训练后，局势发生了根本性逆转。HyGAE 在 5 个评测环境中横扫所有基线方案，最终夺得了 **0.91（91%）的平均成功率**，相比现有的单尺度多轮强化学习基线实现了 **10 个百分点以上的显著提升**。

在针对混合权重 $\alpha$ 的消融研究中，数据显示当 $\alpha = 0.5$（即轮次优势与 Token 优势各取一半）时，模型在绝大多数环境下都能取得最优异且稳健的泛化表现；而将 $\alpha$ 推向纯轮次（$\alpha = 1.0$）或偏向纯 Token 的极端，性能均会出现可观测的回落。这从实验侧再次验证了微观与宏观信用联合分配的必要性。

从定性分析的动作轨迹来看，HyGAE 展现出了传统 VLM 极其罕见的“逆境自省”能力。在基线或冻结参数模型中，当智能体发出向右移动却遭遇障碍物阻拦、环境明确返回“无位移/失败”反馈时，模型往往视若无睹，在下一轮继续机械发出完全相同的向右指令，陷入无休止的无效循环。而经过 HyGAE 训练后的模型在接收到相同的失败反馈后，能迅速在思维链中识别出上一步决策的失效，并在下一轮果断改换策略（例如转而采取向左绕行的动作），展现出了类似人类在面对挫折时的敏捷应变逻辑。

### 总结与启示

长久以来，多模态智能体在多轮决策领域的探索，始终在语言模型的生成本能与强化学习的规划需求之间艰难摇摆。Token 级视角的局限让模型只见树木不见森林，轮次级视角的粗放又让模型知其然而不知其所以然。

HyGAE 的核心价值不仅在于刷新了 5 个基准测试的跑分纪录，更在于它用清晰严密的数学语言，为困扰社区的“微观生成 vs 宏观决策”建模之争给出了一个收敛解。它表明：Token 级与轮次级强化学习从来不是非此即彼的割裂范式，只要理清二者在 POMDP 框架下的状态映射与折现对应关系，单一 Critic 网络完全足以串联起跨轮次的全局反馈与轮内的细粒度因果归因。这种兼顾理论严谨性、算力友好度与优化稳定性的设计，为下一阶段多模态大模型在真实具身场景、复杂 GUI 操作等长周期任务中的落地，铺平了一条坚实而优雅的技术路径。
