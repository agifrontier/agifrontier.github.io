---
layout: default
title: "LoongReflect：把反思做成内存控制，长程Agent检索基准提升12.6分"
description: "为此，研究团队提出了全新的训练框架 LoongReflect 。该方法跳出了将反思视为自由文本批判的传统思路，将反思形式化为一种显式的 内存控制策略（Memory-Control Policy） 。"
arxiv_id: "2608.11967"
paper_published: "2026-08-12"
published_at: "2026-09-13T13:15:08.876178+08:00"
topics:
  - "AI Agent"
  - "推理"
tags:
  - "GRPO"
  - "LoongReflect"
  - "extragradient-style coordination"
  - "long-horizon reflection"
  - "memory-control policy"
  - "multi-hop RAG"
related_tutorials:
  - "self-rag-learning-to-retrieve-generate-and-critique-through-self-reflection"
  - "abseeker-training-long-horizon-search-agents-via-answer-backtracked-credit-assig"
  - "evoharness-rl-learning-self-evolving-runtime-harness-for-long-horizon-llm-agents"
  - "harnessing-uncertainty-entropy-modulated-policy-gradients-for-long-horizon-llm-a"
---

<p class="paper-original-title" lang="en">LoongReflect: Boosting Long-Horizon Reflection in Search Agents via Global Perspective Distillation</p>

<img src="/images/2608.11967v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在面向开放域多步推理与复杂检索任务时，大语言模型驱动的智能体（Agent）正变得越来越依赖长程规划、工具调用与工作记忆的交织执行。在动辄数十步的推理链路中，决定一个智能体上限的往往不是单步动作有多精准，而是它能否在遭遇错误、检索到噪声信息或陷入逻辑死胡同后，及时停下来进行“反思与纠偏”。然而，当前大多数智能体的反思能力仍然停留在表面生成式的自纠错阶段，一旦交互步数拉长，上下文便极易受到污染，甚至出现越反思越偏离目标的恶性循环。

> ArXiv URL：https://arxiv.org/abs/2608.11967v1

来自北京大学与教育部的研究团队在最新论文中指出了这一困境的深层原因：**反思在动作空间上是“局部的”，但其价值评估却依赖“全局”**。单纯依靠最终任务成败给予的强化学习信号，不仅反馈极其稀疏、延迟，而且无法将功过准确归因到某一次局部的反思动作上；反之，若强行在中间阶段引入局部奖励，又极易诱发模型学会无实质意义的敷衍反思（Reward Hacking）。为此，研究团队提出了全新的训练框架 **LoongReflect**。该方法跳出了将反思视为自由文本批判的传统思路，将反思形式化为一种显式的**内存控制策略（Memory-Control Policy）**。

在这一架构下，智能体在可逆的树状轨迹空间中运行，通过结构化的 `<reflect>` 与 `<backtrack>` 动作管理工作记忆，实现错误状态的物理剔除与经验沉淀。为了高效训练这套控制策略，研究团队设计了融合特权教师局部蒸馏（快通道）与终端结果驱动强化学习（慢通道）的双通道前瞻协同优化算法。在七大检索增强问答（RAG）与多步数学推理基准上的实验表明，LoongReflect 在 Qwen2.5-3B 和 7B 模型上均展现出显著优势，多跳检索基准平均 F1 分数较强基线 AgenticRAG-R1 均获得了 12.6 分以上的显著提升。

<img src="/images/2608.11967v1/intro_reflect.webp" alt="长程智能体中反思决策的局部-全局视角落差" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 局部诊断与全局价值的错位：长程反思为何难以学习？

为了理解 LoongReflect 的设计动机，必须先剖析长程智能体反思失效的核心机制。在典型的多步检索增强生成任务中，智能体需要不断调用搜索引擎、筛选片段、聚合证据并推导出最终答案。在这一过程中，局部检索的偶发性偏差几乎无法避免，例如实体消歧错误、检索到表面相关但事实不符的干扰段落，或是过期的记忆写入。

上述错误一旦发生，若智能体仅具备线性的推理轨迹，受污染的信息就会直接进入上下文窗口。随着步数推移，后续的每一次推理都不得不将先前生成的错误作为已知事实，导致状态污染像滚雪球一样不断放大。因此，智能体必须拥有审视轨迹进展、识别缺失证据与局部风险、并决定是继续推进还是放弃当前分支的能力。

然而，训练这种反思能力面临两大根本挑战：

其一是**学习信号困境（Learning-signal dilemma）**。一次反思决策到底是否有效，往往需要经过后续数步工具调用与多轮推导才能见分晓。如果完全采用基于最终答案成败的结果监督强化学习（Outcome-based RL），反思决策分摊到的监督信号极其稀疏、滞后，且 credit assignment（信用分配）严重受阻；但如果像过程奖励模型（PRM）那样为中间的反思步骤引入显式奖励，模型很容易学会迎合局部评分，生成辞藻华丽却对解决最终任务毫无实质帮助的假反思。

其二是**局部与全局的视角落差（Local–global perspective gap）**。反思动作是在当前推理分支的局部上下文中触发的，智能体站在局部节点上，根本无法获知“继续深挖这一分支”还是“立即回溯放弃”会在全局意义上导向更优的最终结果。现有的在策略蒸馏（On-Policy Distillation）或单步验证方法，往往只能评估当前步骤的合理性，却无法从整棵搜索树的宏观走向来校准反思的真实效用。

### 核心机制：将反思建模为可逆轨迹树上的显式内存控制

LoongReflect 解决这一矛盾的切入点，是彻底重构智能体对反思的理解方式：**反思不应是一段松散的自我说理，而是一种精确操作工作记忆的控制行为**。

在 LoongReflect 中，智能体的状态不再是扁平的线性历史，而被形式化为一个可逆轨迹树（Reversible Trajectory Tree）。在任意时刻 $t$，智能体的完整状态包含四个部分：轨迹树 $\mathcal{T}_t=(\mathcal{V}_t, \mathcal{E}_t)$、当前处于活动状态的推理路径 $P_t$、压缩后的工作记忆 $\mathbf{m}_t$，以及被归档丢弃的无效分支集合 $\mathcal{B}_t$。传递给模型上下文窗口的序列 $c_t$ 仅由任务输入 $x$、活动路径 $P_t$ 和压缩记忆 $\mathbf{m}_t$ 序列化而成。

在此基础之上，研究团队在标准执行动作空间 $\mathcal{A}_{\mathrm{exec}}$ 之外，引入了两个专门的控制动作：$\mathcal{A}_{\mathrm{ctrl}} = \{\texttt{<reflect>}, \texttt{<backtrack>}\}$。

当触发 `<reflect>` 动作时，智能体并不会无目的地发散思维，而是输出一个四元组结构化诊断 $\mathbf{r}_t = (e_t^{\mathrm{ver}}, q_t^{\mathrm{risk}}, j_t^{\mathrm{ret}}, d_t^{\mathrm{ctrl}})$。其中 $e_t^{\mathrm{ver}}$ 记录当前已严格验证的事实，$q_t^{\mathrm{risk}}$ 明确指出当前推理链条缺失的证据或潜在逻辑漏洞，$j_t^{\mathrm{ret}}$ 为提议回溯的目标历史检查点，而 $d_t^{\mathrm{ctrl}} \in \{\mathrm{continue}, \mathrm{backtrack}\}$ 则给出分支控制决策。这一设计强迫模型将工作记忆聚焦于确凿事实与结构化风险上，拒绝空泛总结。

当决策为回溯时，`<backtrack>` 动作便正式介入。它将当前已被污染的路径后缀 $P_{j+1:t}$ 整体从当前上下文移除，并移动至归档集合 $\mathcal{B}_{t+1}$ 中；同时，上下文指针无缝回滚至先前经过验证的安全前缀 $P_j$。更为关键的是，模型并不会将失败历史彻底当作从未发生，而是从当前反思中蒸馏出一条精炼的纠错教训（Corrective Lesson）$u_{j:t}$，例如核心逻辑矛盾点、被证伪的假设或下一次探索必须满足的先验约束，并将 $u_{j:t}$ 注入回退后的上下文 $P_{t+1} = P_j \oplus u_{j:t}$。通过这种物理级别的上下文截断与补丁式纠偏，彻底切断了错误信息的延续污染。

<img src="/images/2608.11967v1/method_overview.webp" alt="LoongReflect 整体方法架构图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 双通道前瞻协同优化：快蒸馏与慢强化学习的梯度博弈

定义了动作空间之后，如何让模型学会何时该反思、何时该回溯，成为整个框架的核心技术难点。LoongReflect 提出了一个由“快通道”与“慢通道”构成的双通道训练体系，并通过一种前瞻外梯度（Look-ahead, Extragradient-style）机制化解二者之间的优化冲突。

**快通道：特权教师的局部掩蔽蒸馏（Fast Local Supervision）**。为了解决学习信号稀疏的问题，快通道引入了一个“拥有全局上帝视角”的特权教师模型。该特权教师能够纵览整个轨迹树的全局结构以及最终任务的成败结果，从而站在终局高度审视学生模型在当前局部节点上的表现。教师据此生成结构化的控制标签 $\mathbf{h}_t$，明确指导学生当前分支应保留、修复还是回退。

为了杜绝学生模型通过走捷径直接模仿最终答案，研究团队对教师反馈执行了严格的“答案掩蔽（Answer-Masking）”，监督信号完全限定在 `<reflect>` 和 `<backtrack>` 所对应的控制 Token 范围内。通过掩码矩阵 $m_{t,k}^{\mathrm{ref}} = \mathbf{I}[y_{t,k} \in \mathrm{Span}(\texttt{<reflect>}, \texttt{<backtrack>})]$，快通道损失函数 $\mathcal{L}_{\mathrm{fast}}$ 仅对反思与控制决策进行监督：




{% raw %}$$\mathcal{L}_{\mathrm{fast}} = \frac{1}{Z}\sum_{t,k} m_{t,k}^{\mathrm{ref}}\,\min\!\bigl(\exp(-\delta_{t,k})-1+\delta_{t,k},\;c\bigr)$${% endraw %}



式中 $\delta_{t,k} = \bar{\ell}_{t,k} - \ell_{t,k}$ 表示学生与特权教师对数似然的差异，截断常数 $c$ 用于抑制极端的梯度漂移。快通道利用全局视角为局部反思提供了极高密度的监督信号，直接化解了信号延迟与视角落差问题。

**慢通道：面向完整轨迹的 GRPO 强化学习（Slow Global Optimization）**。尽管快通道提供了密集的局部纠偏指导，但特权教师给出的单步建议未必能在动态环境中绝对保证整条长链路的最优。为此，慢通道基于群体相对策略优化（GRPO）算法，直接面向完整轨迹的最终成功率进行优化。

对于每个任务 $x$，策略模型采样 $G$ 条完整执行轨迹 $\tau_1, \dots, \tau_G$，获取终端任务奖励 $R_1, \dots, R_G$ 并计算相对优势 $\widehat{A}_i = (R_i - \operatorname{mean}_g(R_g)) / \operatorname{std}_g(R_g)$。慢通道的损失函数涵盖整个序列的有效 Token：




{% raw %}$$\mathcal{L}_{\mathrm{slow}}(\theta) = -\frac{1}{G}\sum_{i=1}^{G}\frac{1}{\lvert \xi_i \rvert}\sum_{k\in\xi_i}\min\!\left\{\rho_{i,k}(\theta)\widehat{A}_i,\;\bar{\rho}_{i,k}(\theta)\widehat{A}_i\right\} + \beta\,D_{\mathrm{KL}}(\pi_{\theta}\,\|\,\pi_{\mathrm{ref}})$${% endraw %}



慢通道不关心中间某一步写得多么逼真，只以成败论英雄，确保智能体的反思策略最终服务于长程任务的达成。

**前瞻梯度校准（Look-Ahead Coordination）**。快通道追求局部反思的规范与敏锐，慢通道追求全局成败的绝对收益，两者在优化步中极易产生梯度方向的相互干扰。若简单加权相加，往往会导致某一方目标被掩盖。LoongReflect 借鉴优化理论中的外梯度思想，设计了前瞻校准机制。

算法首先在当前参数 $\theta$ 上，仅沿着快通道连续推进 $K$ 步局部更新，得到临时策略 $\widetilde{\theta} = \mathcal{U}_{\mathrm{fast}}^K(\theta)$。此时，快通道累积的等效探索方向为 $g_f = (\theta - \widetilde{\theta}) / \alpha$。随后，慢通道并不在原点采样评估，而是在快通道“前瞻探路”达到的 $\widetilde{\theta}$ 处计算全局任务梯度的评估方向 $g_s = \nabla_{\widetilde{\theta}}\mathcal{L}_{\mathrm{slow}}(\widetilde{\theta})$。

如果两个方向的内积 $\langle g_f, g_s \rangle < 0$，说明快通道提议的局部修正方向在全局长程视角下会损害任务的最终胜率。此时，算法将 $g_f$ 在 $g_s$ 的反方向投影彻底剔除，只保留其与慢通道不发生冲突的正交分量：




{% raw %}$$g_f^{\mathrm{LA}} = \begin{cases} g_f - \dfrac{\langle g_f, g_s \rangle}{\|g_s\|_2^2}\,g_s, & \langle g_f, g_s \rangle < 0, \\ g_f, & \text{otherwise}. \end{cases}$${% endraw %}



最终，主参数更新采用 $\theta^{+} = \theta - \eta_s g_s - \eta_f g_f^{\mathrm{LA}}$ 完成融合提交。这一设计在数学上保证了密集的局部反思蒸馏永远受到全局任务目标的牵引与校验。

### 多跳检索与跨域泛化评测

为了全面验证 LoongReflect 的有效性，研究人员在七大检索增强问答基准上开展了严格评测。训练集基于过滤后的 HotpotQA 与 2WikiMultiHopQA 构建，这两项任务代表域内（In-domain）表现；而 Bamboogle、FRAMES、MuSiQue、Natural Questions（NQ）以及 TriviaQA 则作为域外（Out-of-domain）基准，专门考查模型面对不同跳数、组合结构与分布偏移时的泛化能力。

测试模型选取了在开源社区极具代表性的 Qwen2.5-3B 与 Qwen2.5-7B。对比基线不仅涵盖标准直接回答（No-RAG）、朴素检索增强（Naive-RAG），还囊括了具备栈式回溯记忆的 AgenticRAG-R1 以及基于强化学习的自我蒸馏框架 RLSD 等先进方法。

实验评测结果呈现出极其一致的趋势：

在 Qwen2.5-3B 上，LoongReflect 在所有七个 QA 基准上均刷新了最优性能，平均 F1 达到了 46.15%，相比此前表现最好的 AgenticRAG-R1（33.55%）大幅提升了 12.60 个百分点。在域内任务上，平均 F1 从 38.46% 跃升至 52.09%；更值得注意的是，在完全未参与训练的五个域外任务上，平均 F1 亦从 31.59% 提升至 43.77%。

在 Qwen2.5-7B 模型上，规律保持一致。LoongReflect 的平均 F1 达到 49.21%，相比 AgenticRAG-R1 的 36.60% 同样取得了 12.61 个百分点的净胜。域内平均 F1 达到 53.86%，域外平均 F1 达到 47.35%。即便是与单纯使用结果强化学习的模型相比，LoongReflect 的领先幅度也极其稳健。这充分证明，显式的可逆内存控制与双通道前瞻校准所带来的收益，是模型参数规模扩张所无法自发填补的。

除了检索密集型任务，研究团队还进一步探究了该反思机制是否能泛化至非检索的长程逻辑推理领域。在 GSM8K 与高难度的 MATH 数据集上，直接部署使用 Qwen2.5-3B 训练出的反思策略，LoongReflect 在 MATH 上取得了 56.0% 的 F1，在 GSM8K 上取得了 82.4% 的 F1，相较于 AgenticRAG-R1 分别提升了 1.2 和 1.8 个百分点，相较于 RLSD 提升了 2.4 和 1.7 个百分点。这一跨领域迁移结果表明，LoongReflect 所学到的状态风险诊断与回溯纠错逻辑，本质上是底层通用的元控制能力，并不局限于搜索算子的调用。

### 消融与敏感性分析：系统收益的真实来源

为了回答究竟是哪一项机制驱动了性能的大幅跃升，论文从阶段贡献、动作空间完整性以及协同超参数三个维度进行了详尽的消融实验。

**阶段贡献拆解**。在 Qwen2.5-3B 上，仅对原始指令模型进行反思格式的指令微调（SFT），平均 F1 从原始的 30.33% 提升至 34.76%（+4.43 分）；而在加入双通道强化学习训练后，平均 F1 进一步大幅提升至 46.15%（净增 11.39 分）。在 Qwen2.5-7B 上，SFT 带来了 3.54 分的提升，而后续双通道优化再次贡献了 7.94 分。这表明，虽然高质量的 SFT 为智能体提供了基础的语法先验与反思动作格式，但决定其能否在动态多步探索中做出正确取舍的关键，依然在于双通道强化学习所带来的全局策略校准。

**控制动作的必要性**。研究团队在 3B 模型上分别移除了 `<reflect>` 与 `<backtrack>` 动作：

- 当剥离 `<reflect>` 动作（即不进行细粒度状态诊断与事实归纳）时，模型平均 F1 遭遇崩塌，从 46.15% 暴跌至 30.84%（下降 15.31 分）；

- 当保留 `<reflect>` 但剥离 `<backtrack>`（即模型只能口头反思，无法物理回滚受污染上下文）时，平均 F1 跌至 33.09%（下降 13.06 分）。

这一组关键对比有力证明了论文的核心论点：**单纯的“嘴上反思”无法替代“内存清理”**。如果没有 `<backtrack>` 将脏数据切除，已产生的幻觉与噪音会继续在注意力层被后续 Token 引用；而若没有 `<reflect>` 提供有依据的诊断与纠错经验，智能体的回溯动作就会退化成盲目的随机游走。

**前瞻协调超参数的动态平衡**。针对前瞻协同机制中的局部内步数 $K$ 与快慢通道权重比值 $w = \eta_f / \eta_s$，实验显示性能呈现明显的倒 U 型曲线。当 $w=1$ 固定时，$K=1$ 的平均 F1 为 41.26%，此时快通道对局部的探索尚未充分展开；当步数增至 $K=3$ 时，性能达到峰值 46.15%；若进一步过度增加至 $K=4$，性能则轻微回落至 44.71%。同样，在 $K=3$ 设定下，无论是降低局部探索比重（$w=0.5$）还是过度夸大局部蒸馏（$w=2.0$），均会导致最终得分下降。这印证了前瞻校准机制的设计初衷：密集的局部纠错信号与长程的全局目标之间存在一个精细的动态平衡点，过度偏向任何一方都会破坏反思的有效性。

### 总结与技术启示

LoongReflect 的工作为大模型长程自主智能体的发展提供了一条极具说服力的技术路径。在很长一段时间里，学术界与工业界倾向于将自我纠错寄托于语言模型内在的涌现能力，即通过提示词引导模型“再想一想”或“批评刚才的回答”。但这类方案在面对长程复杂任务时，暴露出不可避免的脆弱性与上下文退化缺陷。

这篇研究表明，要让 Agent 获得真正可靠的长程执行力，必须完成两个思维转变：

首先是在**表征层**，将反思从一种松散的“思维链后缀”，升级为对内部工作记忆与执行路径实施显式干预的“控制协议”。引入包含物理状态截断的可逆轨迹树机制，是防御长程状态污染的必要工程与算法手段。

其次是在**优化层**，承认局部反思动作与全局成功率之间的视角鸿沟。既不能指望极其稀疏的结果信号教会模型精细的单步诊断，也不能任由局部反思脱离全局结果而自嗨。LoongReflect 采用特权教师局部掩码蒸馏配合结果导向 GRPO、并通过前瞻外梯度投影消除方向冲突的机制，为解决复杂长任务中局部与全局监督信号冲突的问题，提供了坚实且极具借鉴意义的系统范式。
