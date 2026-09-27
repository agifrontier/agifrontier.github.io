---
layout: default
title: "310万次Rollout证实：大模型自主发现系统不存在“万能框架”，动态调度胜过固定方案"
description: "然而，来自 IBM、MIT 和 UC Berkeley 的联合研究团队在最新论文中提出了一个尖锐的问题：那些宣称性能卓越的复杂框架，究竟是在通用机制上真正带来了质的飞跃，还是仅仅在极少数偶然的“高光运行”（Lucky runs）中撞上了好运气？"
arxiv_id: "2607.18235"
paper_published: "2026-07-20"
published_at: "2026-09-27T13:15:07.328975+08:00"
topics:
  - "基础模型"
tags:
  - "LLM rollouts"
  - "OpenEvolve"
  - "TTT-Discover"
  - "adaptive allocation"
  - "budget allocation"
  - "discovery harness"
related_tutorials:
  - "learning-to-discover-at-test-time"
  - "towards-execution-grounded-automated-ai-research"
  - "alpharesearch-accelerating-new-algorithm-discovery-with-language-models"
  - "unifying-data-memory-and-compute-efficiency-in-llm-training-a-survey"
seo_title: "310万次Rollout证实：大模型自主发现系统不存在“万能框架”，动态调度胜过固定方案"
---

<p class="paper-original-title" lang="en">Automated Discovery Has No Universally Superior Harness</p>

<img src="/images/2607.18235v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

利用大模型进行科学与数学算法的“自主发现”（Autonomous Discovery），已成为当前 AI 领域最火热的前沿之一。从 DeepMind 引起轰动的 FunSearch，到后续的 AlphaEvolve、OpenEvolve 以及 TTT-Discover，这类系统的基本运作范式高度一致：语言模型生成候选代码，外部沙盒环境运行并打分，搜索框架（Harness）则负责决定接下来挑哪些历史代码进行变异、杂交或进一步扩展。

> ArXiv URL：https://arxiv.org/abs/2607.18235v1

社区近年来涌现出大量愈发复杂的框架设计。从基于 MAP-Elites 的多岛屿（Multi-Island）演化，到融入 UCT/PUCT 的蒙特卡洛树搜索（MCTS），各种复杂的技巧层出不穷，各类论文也竞相汇报各自在基准测试上的新突破。

然而，来自 IBM、MIT 和 UC Berkeley 的联合研究团队在最新论文中提出了一个尖锐的问题：那些宣称性能卓越的复杂框架，究竟是在通用机制上真正带来了质的飞跃，还是仅仅在极少数偶然的“高光运行”（Lucky runs）中撞上了好运气？

为了彻底厘清这一问题，研究团队完成了迄今为止最具统计控制力的严苛实验：**拆解两大代表性框架（OpenEvolve 与 TTT-Discover），在统一的计算预算下对 30 种框架变体进行超过 310 万次 LLM Rollout 的大规模对比实验，覆盖了从 3B 到 120B 参数量的 12 组“模型-问题”对。** 结果颠覆了许多既有认知：在自主发现领域，根本不存在所谓的“万用最优 Harness”；那些堆叠了海量复杂模块的方案，其泛化能力不仅没有提升，反而经常落后于极简基线。

<img src="/images/2607.18235v1/paper_elite_explore_2x3.webp" alt="框架拆解与跨任务泛化分析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 自主发现系统的“泛化危机”

过往的自主发现系统研究受限于推理成本，通常只进行 3 到 5 次独立实验。然而，大模型在探索未知代码空间时天然伴随极强的随机性，样本过少极易导致将单纯的运行方差误判为方法论层面的创新。

研究团队首先将常见的复杂框架逐步解构为单一变量。以最基础的贪心贪心策略“序列化最佳采样”（Sequential Best-of-$N$, 简称 Sequential BoN）为共同根节点，向两个方向渐进式叠加设计：

1. **演化算法路线（OpenEvolve 风格）**：Sequential BoN $\to$ Top-$K$ 存档 $\to$ 引入历史全域探索（$\epsilon$-greedy） $\to$ 减少单步并发 $N$ 并增加搜索步数 $T$ $\to$ 引入 MAP-Elites 灵感采样 $\to$ 多岛屿并发演化。

2. **树搜索路线（TTT-Discover 搜索内核）**：Sequential BoN $\to$ 子树价值预估（Subtree Value Estimate） $\to$ 引入基于访问计数的 UCT 探索奖励 $\to$ 引入带先验的 PUCT 机制 $\to$ 单步扩展多个父节点。

研究人员在圆堆积（Circle Packing）、Heilbronn 三角形问题以及第二自相关不等式等典型发现任务上，分别使用 Qwen2.5-3B-Instruct、Qwen3-4B-Instruct-2507、GPT-OSS-20B 和 GPT-OSS-120B 四种不同量级的大模型展开对比。

结果令人警醒：**没有一种固定的框架能够在全部测试的“模型-问题”配对中稳定占据优势。** 很多时候，一个框架在模型 A 解决问题 X 时表现拔群，但在换到问题 Y 或模型 B 时，性能便会急转直下。所谓的通用算法框架，本质上更像是一个高度依赖特定任务和基座模型的“超参数”，而非可以放之四海而皆准的万能配方。

<img src="/images/2607.18235v1/paper_openevolve_progression_3tasks_no_gp.webp" alt="OpenEvolve演化路径的递进消融实验" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 复杂堆砌并非解药：OpenEvolve 与树搜索的逐级拆解

研究团队在严格控制计算预算（严格对齐 Rollout 数量）的前提下，对两大主流技术流派进行了逐级剥离实验。

在 OpenEvolve 的递进消融中（如上图所示），结果出乎意料地没有呈现出“组件越多、性能越强”的单调上升趋势。最基础的轻量级探索改动（例如仅仅引入一个极小概率 $\epsilon$ 去历史池中回踩采样）就足以在特定任务上带来显著增益；然而，一旦继续叠加 MAP-Elites 特征存档机制、双亲灵感采样甚至多岛屿隔离演化等一整套完整工具链后，在绝大多数配置下，最终得分反而显著回落。完整的 OpenEvolve 方案在多个问题上的表现，甚至系统性地弱于被剔除了大量花哨功能的中间变体。

<img src="/images/2607.18235v1/paper_puct_uct_bp_summary_3tasks.webp" alt="TTT-Discover风格的树搜索消融对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在树搜索（TTT-Discover 风格）分支中，情况呈现出类似的特征。将贪心父节点替换为经典的 UCT 或带有先验项的 PUCT，确实在部分几何优化任务（如 Heilbronn 三角形问题）中带来了肉眼可见的突破。但是，随后为了匹配完整 TTT-Discover 而进一步调整搜索树的深度与单步多父节点并发采样时，这些额外的改动并未持续推高上限，反而经常将前期靠 UCT 积累的收益彻底抹平。

这说明当前许多研究把精力放在了设计层层嵌套的复杂元启发式机制上，但在严苛对齐总算力的前提下，这些复杂模块反而过早分散了模型的注意力，降低了在有效路径上的开采深度。

### 前期进展即信号：从固定设计转向动态早停

既然无法在实验前预知哪种框架适合当下的任务，那么把全部计算资源盲目押注在单一框架上显然是低效的。

这引发了一个实际权衡：面对一个全新的未知问题，我们到底应该把预算花在反复运行同一种固定框架去搏“单次峰值”，还是应该探索多种不同的框架？

研究团队发现了一个极具价值的经验规律：**探索前期的进展与最终的发现质量呈现极强的正相关性。**

<img src="/images/2607.18235v1/predictive_core_stats_by_checkpoint.webp" alt="早熟相关性与跨任务胜率统计" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

如上图 Panel (b) 所示，在搜索刚推进到总预算的 10%、25% 和 50% 时，当前阶段取得的最好成绩与最终全局成绩之间的 Spearman 秩相关系数随着步数迅速走高。换言之，一个如果能在终点取得卓越发现的框架，通常在任务的前 25% 阶段就会展现出领先的苗头；而前中期持续低迷的运行路径，在后期逆袭的概率微乎其微。

基于这一发现，研究团队构建了一种预算对齐的**在线动态资源分配方案（Adaptive Allocation）**。该方法借鉴了超参数优化领域的 Successive Halving 与 Hyperband 思想：

- 在初始阶段，系统同时启动多个差异化的候选 Harness 并发试跑；

- 当消耗到设定阶段点（如 $q_1, q_2$）时，外部评估器直接读取当前局部的真实得分；

- 算法果断剪枝淘汰排名靠后的弱势框架，将其未消耗的剩余 Token 和计算预算全部重新倾斜给跑分领先的优胜者，让更匹配当下环境的框架去跑深、跑透。

实验证明，在相同总计算开销下，这种动态剪枝调度策略的平均表现不仅全面超越了“盲选单一固定框架”的基线，也显著击败了“平均平分资源、互不淘汰”的朴素集成方案（Ensemble）。

### 自主探索领域需要重构评估范式

这篇论文给当前火热的大模型自主发现与自动编程研究敲响了警钟。

长期以来，该领域常常沉迷于设计越来越花哨的框架外壳，并将个别基准上的偶然成功归结为某种“普遍性方法突破”。而 IBM、MIT 与 Berkeley 的这项工作用 310 万次严格的 Rollout 证实：**Harness 没有银弹，强行寻找普适通用的固定框架是徒劳的。**

这一结论指明了两个关键转向：在学术研究侧，探索算法的有效性验证必须建立在足够的重复试验与空分布基线上，为此论文团队公开了全部 310 万次运行的完整轨迹与打分记录，为社区提供了统一的评估基准；在工业应用侧，与其费尽心机调优一个庞大脆弱的固定架构，不如建立敏捷的在线动态淘汰机制——用廉价的前期抽样探测问题特性，让算力顺应反馈动态流向最有希望的路径。在大模型奔赴未知前沿的路上，自适应的弹性往往比精巧的定式走得更远。
