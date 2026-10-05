---
layout: default
title: "A-SR：让大模型自进化找公式，科学符号回归准确率从25.8%跃升至48.3%"
description: "为此，研究团队提出了自进化智能体符号回归框架 A-SR （Agentic Symbolic Regression），将控制的核心从“公式编辑算子”转移到“角色-记忆视图对”，在权威基准测试 LLM-SRBench 上，将 Llama3.1-8B 的科学公式发现准确率从 25.79% 提升到了 48.30%。"
arxiv_id: "2608.04872"
paper_published: "2026-08-05"
published_at: "2026-10-05T13:15:07.965576+08:00"
topics:
  - "AI Agent"
tags:
  - "A-SR"
  - "Agentic LLMs"
  - "Evaluator-Reward Role Policy"
  - "Hierarchical Coordination"
  - "LLM-SRBench"
  - "Role-conditioned Evidence Views"
related_tutorials:
  - "retaining-by-doing-the-role-of-on-policy-data-in-mitigating-forgetting"
  - "morse-task-oriented-multi-agent-system-with-mixture-of-role-subtask-experts"
  - "gaussmemory-task-driven-3d-gaussian-scene-memory-for-long-horizon-robotic-manipu"
  - "skillhex-improving-agent-skills-via-hypothesis-driven-autonomous-exploration-and"
seo_title: "A-SR: Self-Evolving Agentic LLMs for Symbolic Regression via Hierarchical Coordination"
---

<p class="paper-original-title" lang="en">A-SR: Self-Evolving Agentic LLMs for Symbolic Regression via Hierarchical Coordination</p>

<img src="/images/2608.04872v2/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在科学发现的历史上，人类最顶尖的智力成果往往凝结为极其精炼的数学公式。从牛顿的万有引力定律到爱因斯坦的质能方程，科学家总是试图从有限的观测数据中提炼出结构紧凑、具有因果外推能力的闭式解析表达式（closed-form equations）。这一任务在机器学习中被称为符号回归（Symbolic Regression, SR）。与黑盒神经网络单纯追求在分布内的数值拟合精度不同，符号回归追求的是真实规律的数学骨架，不仅要求可解释性，更要求在从未见过的极端测试区间（即分布外外推，OOD）依旧成立。

> ArXiv URL：https://arxiv.org/abs/2608.04872v2

近年来，将大型语言模型（LLM）引入符号回归成为了前沿热点。研究者发现，预训练模型中蕴含着丰富的科学常识、量纲直觉与编程语法先验。将公式表示为可执行代码后，LLM 负责生成公式骨架，数值优化器（如 BFGS 或非线性最小二乘）负责拟合连续参数，二者结合展现出超越传统遗传编程算法的潜力。然而，现有的 LLM 引导方法存在一个致命瓶颈：它们大多沿用“统一提议-评估循环”（unified proposal-evaluation loop）。在这种简化的工作流中，大模型写出一个候选公式，评估器计算出一个均方误差（MSE），随后这个标量分数被塞回下一轮 Prompt 中让大模型“反思”并重新采样。

来自上海人工智能实验室、同济大学与加州大学洛杉矶分校（UCLA）的研究团队指出，这种将千差万别的失败模式全部压缩成单一标量分数的做法，是导致模型陷入搜索死循环的根本原因。真实的科学探索过程中，代码报语法错误、参数优化发散、出现大量无法抵消的冗余项、或者分布内过拟合导致外推崩塌，属于本质完全不同的失败类型。面对这些错误，研究系统需要的不是同一套含糊的重试指令，而是精准调配不同的推理角色与针对性的过程证据。为此，研究团队提出了自进化智能体符号回归框架 **A-SR**（Agentic Symbolic Regression），将控制的核心从“公式编辑算子”转移到“角色-记忆视图对”，在权威基准测试 LLM-SRBench 上，将 Llama3.1-8B 的科学公式发现准确率从 25.79% 提升到了 48.30%，在真实物理与材料实验数据上也刷新了多项指标。

<img src="/images/2608.04872v2/fig3_coordination_mechanism.webp" alt="A-SR反馈驱动的智能体协同机制" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 从“怎么修改表达式”到“谁该看到什么证据”

在经典的符号回归与演化算法中，学术界探索过以“编辑算子”（Edit Operators）为核心的自适应策略。例如近期的代表性工作 Deliberate Evolution（DE），尝试通过大模型自适应选择变异、交叉、重生成等算子，并利用工具分析残差。然而研究团队深入分析搜索日志后发现，单纯决定“对公式执行什么编辑”依然不够透彻。

问题的根源在于信息不对称：同一个标称的“变异”或“微调”操作，如果给大模型展示的是精英公式的骨干结构，它会倾向于保留守恒项；但如果给它展示的是数值震荡的失败轨迹，它就会倾向于重构分母。换言之，**决定搜索行为本质的，不是调用了哪个编辑动词，而是当前负责推理的角色究竟看到了哪一部分过程证据**。以往的统一提议循环往往在这个环节错配证据：当公式反复出现参数不可导、除以零等稳定性问题时，Prompt 却在催促模型去寻找更宏大的新颖结构；当公式已经捕捉到了主导物理机制、急需消除冗余常数项时，系统却依然向模型灌入大量的历史错误代码。

如图 1 所示，A-SR 彻底打破了单 Prompt 循环的思路，将控制单元重构为“角色-记忆视图对”（role–memory-view pair）。系统不仅要把执行器的数值评估反馈拿来算分，更要把错误类型解析为结构化的诊断信号，进而动态决定：下一轮推理该由哪一个专门角色出场？该角色应该获取什么样的定向记忆？

为了承担不同的搜索职能，A-SR 在同一个底层大模型的基础上，通过特定的角色提示词切分出四个互补的智能体角色：

- **生成者（Generator）**：专注于提出全新的方程骨架，从物理量纲和全局函数族出发，开辟新的搜索空间；

- **分析者（Analyst）**：专门排查残差与缺失项，对比当前候选公式与观测数据的系统性偏差，指出应引入周期项还是指数衰减项；

- **化简者（Simplifier）**：致力于奥卡姆剃刀原则，负责压缩公式、消除抵消项与无用参数，把复杂的代数表达式规约到紧凑稳健的形式；

- **审查者（Reviewer）**：充当守门人，严格检查程序可执行性、参数边界合理性、数值稳定性和外推发散风险。值得注意的是，Reviewer 在搜索过程中绝不接触保留的分布外标签，其全部预警信号均来自可执行环境的内在反馈（例如高阶导数发散、参数逼近设定上下界等）。

这种职能解耦使得系统在遭遇具体困难时，不再指望单个通用 Prompt 突然灵光一闪，而是由对口的专家角色调用特定证据精准破局。

<img src="/images/2608.04872v2/framework.webp" alt="A-SR整体架构设计与自进化流程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### A-SR 的双时间尺度自进化体系

A-SR 的整体架构如图 2 所示，包含状态感知协同、协议选择、在线角色策略自适应以及状态路由过程记忆四大支柱。整套框架在时间维度上巧妙设计了两个尺度的自进化机制：**运行期内的测试时自进化（Within-run Evolution）**，以及**跨任务的离线轨迹蒸馏（Across-run Distillation）**。

#### 1. 运行期内的分层协调与非平稳策略更新

在单次求解特定科学数据集的搜索过程中，A-SR 并不修改底层大模型的权重参数，而是通过动态调整上下文环境与决策逻辑实现自适应。

在宏观层面上，协调器在搜索初期会先进行性能剖析（Profiling），根据前期采样的可执行可靠性 $\rho_{\mathrm{rel}}$（语法有效率、数值稳定率）与生产力 $\rho_{\mathrm{prod}}$（解的改进频率）两个维度，从预先提炼的四种协调协议中选择当前最匹配的基线策略：




{% raw %}$$ \pi^{\star} = \mathrm{SelectProtocol}(\rho_{\mathrm{rel}},\rho_{\mathrm{prod}}), \quad \Pi = \{\pi_{\mathrm{EGC}},\pi_{\mathrm{RGC}},\pi_{\mathrm{PGC}},\pi_{\mathrm{SGC}}\} $${% endraw %}



这四种协议分别映射了科学搜索中的四种典型状态：

- **探索导向协议（$\pi_{\mathrm{EGC}}$）**：当可靠性极高且持续有更优解产生时激活，大幅向 Generator 和 Analyst 倾斜资源，加快空间拓展；

- **可靠性导向协议（$\pi_{\mathrm{RGC}}$）**：当搜索充斥着无效程序或数值发散时启动，将控制权移交给 Reviewer 与 Simplifier 强制做排错与降维；

- **阶段导向协议（$\pi_{\mathrm{PGC}}$）**：模拟人类科研流程，前期重点探索新机制，后期平滑过渡到正则化与化简收敛；

- **稳定性导向协议（$\pi_{\mathrm{SGC}}$）**：当搜索陷入长期停滞时激进化探索，但在模型提议不稳时迅速收缩。

在微观层面上，确定了宏观协议后，协调器引入了一个轻量级的非平稳在线多臂老虎机（Bandit）策略。每当一个角色 $r_t \in \mathcal{R}$ 提议的公式经过物理执行器评估后，环境会生成即时标量奖励 $R_t$：




{% raw %}$$ R_t = \lambda_{v}\mathbb{I}_{\mathrm{valid}} + \lambda_{b}\mathbb{I}_{\mathrm{best}} - \lambda_{i}\mathbb{I}_{\mathrm{invalid}} - \lambda_{p}\mathbb{I}_{\mathrm{param}} $${% endraw %}



该奖励不仅奖励产生历史全局最佳公式（$\mathbb{I}_{\mathrm{best}}$），同时严肃惩罚无效代码（$\mathbb{I}_{\mathrm{invalid}}$）和参数发散（$\mathbb{I}_{\mathrm{param}}$）。随后，系统使用指数移动平均更新该角色的效用值 $U_t(r_t)$：




{% raw %}$$ U_t(r_t) = \mathrm{clip}\left((1-\eta)U_{t-1}(r_t) + \eta R_t, -1, 1\right) $${% endraw %}



最终，在第 $t$ 步挑选哪个角色登场，取决于宏观协议的先验偏置分 $S_{\pi^{\star}}(r)$ 与动态更新的效用乘积的综合排序：




{% raw %}$$ S_t(r) = S_{\pi^{\star}}(r) + \alpha_{\pi^{\star}}g_{\pi^{\star},t}(r)U_t(r), \quad r_t = \arg\max_{r\in\mathcal{R}}S_t(r) $${% endraw %}



不仅选角色，**记忆路由（State-Routed Process Memory）** 机制同时动作：协调器绝不给角色看堆砌的全量历史日志，而是定向分流。Generator 获得全局精英公式的结构 Motif；Analyst 获得当前最优公式在残差极大处的误差切片；Simplifier 获得符号展开式与过拟合项分析；Reviewer 则获得近期引发异常的堆栈轨迹与参数边界越界记录。角色只在专属的信息视图中推理，极大减轻了长上下文退化与注意力分散。

#### 2. 跨任务轨迹蒸馏：A-SR-LoRA

如果在多次搜索运行中记录下了成功的协调决策和提议轨迹，这些高质量的思维链路是否可以沉淀为大模型固有的科学推理直觉？

研究团队进一步构建了离线路径，将 A-SR 运行日志中各个角色在特定状态上下文 $C_t$ 下成功改进损失或修复缺陷的公式提议 $f_t$，整理为微调数据集。通过参数高效微调技术（LoRA），将以角色为条件的主动提议先验注入开源基座模型（如 Qwen3-4B-Instruct），构建出 **A-SR-LoRA**：




{% raw %}$$ p_{\phi}\left(f_t \mid P, r_t, z_t, C_t\right) $${% endraw %}



在推理时，A-SR-LoRA 依然运行在上述分层协调控制器、老虎机更新与记忆路由框架中，但其提议主干换成了具备科学角色先验的自研模型。这实现了测试时无梯度适应与跨生命周期参数进化的闭环融合。

<img src="/images/2608.04872v2/fig_ood_acc_qwen.webp" alt="Qwen3-4B在外推准确率上的对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 基准评测：全方位的鲁棒性与精度跃进

评测符号回归能力不能仅靠单一玩具数据集。本研究在两大硬核基准上进行了详尽测试：涵盖物理、材料、化学、生物四大领域的开源基准 **LLM-SRBench**，以及源自真实实验观测的 4 个复杂真实科学发现任务（包括非线性振荡器、大肠杆菌种群生长、以及 6061-T651 铝合金真实拉伸应力-应变曲线）。

评估的核心指标包括归一化均方误差（NMSE，越低越好）以及科学领域极其严苛的准确率指标 **Acc@0.01**（预测值与真实值相对误差在 1% 以内的样本比例，越高越好）。

在以 Llama3.1-8B-Instruct 为基座的 LLM-SRBench 主实验中，A-SR 展现出极其显著的性能优势。如表 1 所示，此前表现最优秀的基线模型 Deliberate Evolution（DE）在四大合成科学领域（LSR-Synth）的平均 Acc@0.01 仅为 25.79%，而完整的在线 A-SR 直接将其推高至 **48.30%**，绝对增幅达到 22.51 个百分点。在 Physics、Material、Chemistry、Biology 四个子领域中，A-SR 的 Acc@0.01 全部取得第一名。

一个有趣的现象是行为分工：在涉及代数变形的变换测试集（LSR-Transform）中，禁用了在线效用更新的静态协调版本（A-SR-Static）取得了最低的 NMSE；但在面对真正复杂的真实科学机理发现时，拥有在线 Bandit 角色策略自适应的 A-SR 则在任务解出率与数值鲁棒性上展现出压倒性优势。

而在针对较小参数规模模型 Qwen3-4B-Instruct 的实验中，轨迹蒸馏的价值得到了坚实验证。如图 3 所示，在具有极大外推难度的分布外测试（OOD Acc@0.01）上，原本原生 Qwen3-4B 基线的平均准确率仅有 24.58%，经过 A-SR 训练-推理协同加持后的 A-SR-LoRA 将其提升至 **38.29%**。尤其在 Material、Chemistry 和 Biology 领域，外推准确率均实现了阶梯式上升。唯一的例外出现在 Physics 领域，由于该领域的方程大量包含高频三角震荡与相位敏感项，任何参数的微小偏移都会在 OOD 区间引起剧烈发散，这印证了高度非线性动力学系统符号拟合的固有难度。

在 4 个真实世界科学任务的评测中（采用 GPT-3.5 统一测试以对齐历史基线），A-SR 同样保持了强悍的统治力。在报告的全部 8 个指标（4 项任务的分布内 ID 与分布外 OOD NMSE）中，A-SR 在 **7 个指标上取得了全场最优**。特别是在大肠杆菌非线性生长预测中，A-SR 的 OOD NMSE 较经典科学发现方法大幅削减；在复杂的铝合金应力-应变拉伸实验中，面对带有测量噪声的复杂真实回弹曲线，A-SR 的分布内拟合误差达到最低，外推表现也与当前最顶尖的专用系统不相上下。

### 为什么 A-SR 没有退化为“随机试错”？

多智能体系统在长程任务中最容易出现的病态现象是“角色坍缩”——某个在早期获得正反馈的角色迅速主导整个采样过程，导致系统重新退化为单一的提议流；或者是频繁生成无法编译的废代码，浪费昂贵的算力预算。

论文对搜索过程动力学的监控直接打消了这一疑虑。图 4 展示了化学动力学反应任务 CRK12 求解过程中的全景搜索轨迹。在整个迭代步内，A-SR 的历史最优 Loss 呈现出清晰的阶梯式平稳下行。尤为关键的是下面的过程监控曲线：

1. **有效公式率（Validity Rate）始终高位运行**：得益于 Reviewer 的敏锐拦截与可靠性协调协议的约束，系统几乎没有在语法错误或矩阵奇异上陷入长时间死循环；

2. **角色使用均衡活跃（Role Usage）**：Generator、Analyst、Simplifier 与 Reviewer 均在整场搜索中保持了健康的调用频次。系统往往在开局由 Generator 快速拓宽基底，中期频繁调度 Analyst 修正阶数，在发现有潜力的骨架后由 Simplifier 快速砍去过度拟合项， Reviewer 则在关键节点排除数值不稳定风险。四个角色没有一个被边缘化；

3. **记忆定向流转**：路由模块持续根据当前搜索所处的停滞或跃迁阶段，交替喂入骨干 Motif、残差切片或失败轨迹，推动搜索跨越局部极值点。

在消融实验中，研究人员在包含 45 个任务的测试子集上拆解了系统的各个组件。结果表明，如果去掉状态记忆路由，仅保留统一提示，模型的有效解率明显下跌；如果去掉协议动态选择，固定使用某一套交互逻辑，系统在面对不同物理领域的适应能力严重受挫；而如果采用无协调的固定轮询（Fixed Role Rotation，即四个角色机械地轮流说话），性能甚至会退化到接近原始单 Prompt 的水平。这充分说明，A-SR 的效能跃迁并非来自简单的“多套几个 Agent 面具”，而是源于“协议选择-效用更新-针对性证据装配”这一严密咬合的闭环反馈控制。

<img src="/images/2608.04872v2/figure6.webp" alt="A-SR在真实科学任务中发现的公式对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 走向机理级科学发现：从形似到神似

评估符号回归模型的高下，终极标准是看它找到的公式究竟是“勉强拟合数据的凑数多项式”，还是“契合真实物理法则的内在机理”。

图 5 对比了 A-SR 与其他弱基线在实际物理与生物任务中所发现的符号结构。在物理与化学变换方程的求解中，基线模型为了在训练集上降低误差，往往会贪婪地拼凑高阶辅助项或极其敏感的三角函数，这些公式在分布内虽然残差很小，但一旦进入外推区间便立刻由于多项式爆炸而飞出天际。而 A-SR 找到的公式，在化简和 Reviewer 的持续规约下，往往能够直接在代数层面上精准还原出物理守恒律的标准因式分解形式，不仅与真实标准方程完全等价，而且连续参数的收敛值极为稳定。

在更加复杂的生物学任务中，尽管真实系统的物理规律本身并不存在唯一标准形式，A-SR 所发现的数学结构却精准捕捉到了典型的“生长-饱和-衰减”（growth–saturation–decay）三阶段非线性动力学机制。公式中的分母阻尼项与分子驱动项之间形成了稳定的物理平衡，因而在脱离原始实验区间的极端状态下，依然能够保持优秀的预测包络。这种“机理层面的发现能力”，正是科学 AI 区别于传统黑盒回归的魅力所在。

### 总结与未来演进

A-SR 的成功给大模型驱动的复杂科学搜索提供了一个极具普适性的启示：在面对高难度、多失败模式的严谨推理任务时，将决策单元粗暴地停留在“让模型重试生成”或“给模型换一个编辑动作”是不够的。大模型本质上是一个高度依赖上下文线索的条件概率生成器，**让正确的角色在正确的时刻看到与当前失败模式对齐的特定证据**，才是撬动其推理先验的正确杠杆。

尽管当前的 A-SR 仍使用显式的启发式协调规则与轻量级在线 Bandit 算法，未采用端到端的可微控制器训练，但这恰恰赋予了它极高的透明度与架构解耦性，使得无论面对闭源顶尖大模型还是轻量化开源基座，它都能实现即插即用的外挂式增强。随着更大规模科学发现轨迹的积累，结合强化学习训练的元控制器（Meta-Controller）与更深层次的跨任务符号记忆库，有望进一步拓宽这类自进化智能体在天文、核聚变、量子力学等更为严苛的未知科学前沿中的探索边界。
