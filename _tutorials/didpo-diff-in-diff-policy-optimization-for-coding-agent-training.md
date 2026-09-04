---
layout: default
title: "DiDPO：把代码 Diff 拆解归因，Stanford 等提出代码 Agent 强化学习新框架"
description: "在大语言模型被赋予工具调用、代码执行和环境反馈能力之后，基于可验证奖励的强化学习（RLVR，ReinforcementLearningwithVerifiableReward）已经成为训练代码Agent（CodingAgent）的核心范式。"
arxiv_id: "2608.07147"
published_at: "2026-09-04T11:26:52.370742+08:00"
topics:
  - "AI Agent"
  - "强化学习"
tags:
  - "DiDPO"
  - "RLVR"
  - "advantage groups"
  - "code diffs"
  - "coding agents"
  - "critic-free RL"
related_tutorials:
  - "agentgym-rl-training-llm-agents-for-long-horizon-decision-making-through-multi-t"
  - "coda-coordinating-the-cerebrum-and-cerebellum-for-a-dual-brain-computer-use-agen"
  - "mixture-of-minds-multi-agent-reinforcement-learning-for-table-understanding"
  - "training-task-reasoning-llm-agents-for-multi-turn-task-planning-via-single-turn-"
---

<p class="paper-original-title" lang="en">DiDPO: Diff-in-Diff Policy Optimization for Coding Agent Training</p>

在大语言模型被赋予工具调用、代码执行和环境反馈能力之后，基于可验证奖励的强化学习（RLVR，Reinforcement Learning with Verifiable Reward）已经成为训练代码 Agent（Coding Agent）的核心范式。与传统的问答或纯数学推理不同，软件开发环境天生具备客观的判准：代码能否编译通过、单元测试能否跑通、报错信息是什么。这些由编译器和测试套件给出的确定性信号，为策略模型提供了最直接的优化依据。

> ArXiv URL：https://arxiv.org/abs/2608.07147v1

然而，现有的强化学习方法在迁移到复杂代码任务时，遇到了一个极其棘手的瓶颈——**细粒度信用分配（Credit Assignment）的严重失效**。

当前主流的无评论家（Critic-free）算法，如 GRPO、GSPO，以及针对多轮交互的 GiGPO，在很大程度上依赖轨迹级奖励（Outcome Reward）或状态级奖励（Step-level Reward）。这种设计假定环境交互是一步一步发生的原子动作（Atomic Action）。但是在软件工程场景中，Agent 单次提交的代码变更（Code Diff）往往非常庞大且复合：它可能同时在一个函数里修复了越界漏洞，在另一个函数里调整了返回值类型，还附带修改了几行无害但也无用的注释。

如果仅凭最终测试“通过”或“失败”，就把奖励平摊给这次变更里的每一个 Token，或者整条轨迹里的每一步，强化学习便无法辨别究竟是哪一行关键改动拯救了程序，又是哪一行冗余修改埋下了隐患。

为了打破这一困局，来自斯坦福大学（Stanford University）、同济大学和中国科学技术大学的研究团队提出了 **DiDPO（Diff-in-Diff Policy Optimization）**。该方法不再将单步代码提交视为黑盒，而是深入代码差异的内部结构，提出了一种动态挖掘复现子块（Sub-diff）并构建优势分组的无 Critic 算法。在保持极低训练开销（额外耗时仅约 2.3%）的前提下，DiDPO 在 Qwen2.5-7B-Coder 和 Qwen3.5-4B 上均大幅超越了现有前沿的 Agentic RL 基线，并在算法竞赛级基准 USACO 上实现了相较于 GRPO 超过一倍的准确率跃升。

<img src="/images/2608.07147v1/aaa.webp" alt="代码动作与传统 Agent 动作的结构差异" style="width:85%; max-width:450px; margin:auto; display:block;">

### 代码动作的复合性：传统 Agentic RL 为何水土不服？

在传统的具身智能、网页浏览或简单工具调用任务中，智能体的动作通常对应于状态空间中的单次状态转移：点击按钮、移动一步、调用某一个特定 API。每个动作相对独立，对状态的影响边界清晰。

但在代码编辑场景中，代码动作呈现出完全不同的物理特征：

首先是**动作的高密度打包与交错**。随着任务难度提升，模型往往倾向于在单个交互步骤中输出长篇代码。这些代码变更横跨多个文件或同一个文件的不同段落，不同修改区域之间可能执行着完全独立的逻辑功能。

其次是**功能子块（Sub-diff）的异质性**。一次代码提交生成的整体 Diff，实质上是由多个执行不同功能的局部子差异组成的。例如，某个子块是在初始化边界变量，另一个子块是在重构核心循环。如果把整个 Diff 作为单一状态动作对来进行分组对比，两个在核心逻辑上完全一致、仅仅因为变量命名或空行略有差异的轨迹，就无法被归为同类状态。

最后是**语法的刚性约束**。代码执行具有高度的敏感性，一段包含十处精妙重构的代码，可能仅因漏掉一个括号或拼错一个变量名而导致全局崩溃。

在强化学习中，直接比较整段 Diff 会带来严重的样本碎片化问题。如下图所示，研究团队在 APPS 数据集上进行统计发现：如果以完整的整段 Diff 为粒度聚类状态，绝大多数分组里只包含极少量的样本（Small-mass groups），因为多次 Rollout 很难碰巧写出完全一模一样的长代码；但如果将 Diff 进一步拆解为局部的 Sub-diff，高频复现的逻辑单元就会大量浮现，形成具有统计意义的大容量分组（Large-mass groups）。这证明了：**代码动作天生具备可分性（Divisible），且必须被拆开归因。**

<img src="/images/2608.07147v1/show.webp" alt="整段 Diff 分组与子 Diff 分组的样本规模分布对比" style="width:85%; max-width:600px; margin:auto; display:block;">

### DiDPO 的核心机制：拆解、评分与锚点聚合

DiDPO 并不引入庞大且难以收敛的价值网络（Value Critic），而是沿着多采样对比（Group-relative Policy Optimization）的路线，在轨迹级优势之上，原生地嵌入了“Diff 内部的局部优势”。

整个算法由三个紧密咬合的环节构成：子差异拆分与匹配、基于可成团性得分（Groupability Score）的锚点选择、以及双层优势反向投影。

<img src="/images/2608.07147v1/master.webp" alt="DiDPO 架构全览与优势计算流程" style="width:85%; max-width:600px; margin:auto; display:block;">

#### 1. 动态锚点与可成团性评分（Groupability Score）

如果盲目地把一段代码按行甚至按 Token 随意切碎，强化学习就会被海量的无意义字符（如括号、缩进、空行）所淹没。因此，子块切分必须同时满足两个约束：**语义要完整，跨轨迹要能复现**。

研究团队为此设计了量化评估子块质量的“可成团性得分”（Groupability Score, GS）。对于一个潜在的锚点候选 $\mathbf{c}$，其得分定义为：




{% raw %}$$ \mathrm{GS}(\mathbf{c}) = \big(1 - e^{-\bar{L}(\mathbf{c})}\big) \big(1 - e^{-(n(\mathbf{c}) - 1)}\big) $${% endraw %}



这一简明的公式精妙地平衡了两个矛盾的维度：

* **语义跨度（Semantic Scope）项**：$1 - e^{-\bar{L}(\mathbf{c})}$，其中 $\bar{L}(\mathbf{c})$ 表示该代码块的平均长度。当代码片段极短（例如只有单行括号或变量声明）时，得分被强烈惩罚；随着长度增加并逐渐构成一个功能性代码块，该项平滑趋近于 1。

* **分组体量（Group Mass）项**：$1 - e^{-(n(\mathbf{c}) - 1)}$，其中 $n(\mathbf{c})$ 表示该代码块在当前任务采样的多条轨迹中复现的频次。如果一个子块只在一条轨迹中独现（$n=1$），得分直接归零；只有当该逻辑片段在不同 Rollout 中反复出现、能够形成参照组时，才能获得高分。

在此基础上，从整段 Diff 集合中寻找最优锚点集合 $\mathcal{C}^{\star}$ 的过程，被形式化为一个带基数约束的设施选址型次模函数最大化问题（Submodular Maximization）：




{% raw %}$$ \mathcal{C}^{\star} = {\rm argmax}_{|\mathcal{C}| \leq K} \sum_{s \in \mathcal{I}(\mathcal{S})} \max_{\mathbf{c} \in \mathcal{C}} \mathbf{1}[s \in \mathcal{O}(\mathbf{c})] \mathrm{GS}(\mathbf{c}) $${% endraw %}



内部的最大化机制保证了重叠的候选子块不会被重复计入奖励，算法通过贪心搜索即可在线性时间内高效抽取出代表性最强的局部功能单元。

#### 2. 从局部子块到 Token 级的双层优势计算

一旦锚点选定，原本松散且复杂的全量代码 Diff 就会被自动切割：匹配上锚点的部分划归为对应的优势组，未匹配的部分则各自保留。

对于落入同一锚点优势组 $G(\mathcal{C}^{\star})$ 的动作，算法计算出它们的**局部差异级优势（Diff-level Advantage）** $A^D$：




{% raw %}$$ A^{D}(\mathbf{a}^{(i)}_{t,m}) = \frac{R_{t}^{(i)} - \mathrm{avg}\big(\{R_{t}^{(j)} \mid (\mathbf{a}_{t,m}^{(j)}, R_{t}^{(j)}) \in G(\mathcal{C}^{\star})\}\big)}{F_{norm}\big(\{R_{t}^{(j)} \mid (\mathbf{a}_{t,m}^{(j)}, R_{t}^{(j)}) \in G(\mathcal{C}^{\star})\}\big)} $${% endraw %}



这一局部优势衡量的是：在大家都不约而同修改了这一处逻辑的背景下，当前轨迹对该逻辑的具体实现究竟是好是坏。

最后，DiDPO 将传统的全局轨迹优势 $A^E$ 与差异级优势 $A^D$ 线性组合，形成最终投影到每个代码 Token 上的混合优势值：




{% raw %}$$ \hat{A}_{i,l} = A^{E}(\mathbf{\tau}^{(i)}) + \lambda \cdot A^{D}(\mathbf{a}^{(i)}_{l}) $${% endraw %}



通过这种设计，$A^E$ 保障了智能体朝着跑通整体测试用例的宏观方向演进，而 $A^D$ 则像一把精细的手术刀，在单条轨迹的内部对各个局部修改进行奖惩加权。

### 理论支撑：对齐误差与方差缩减的边界

论文从度量几何（Gromov-Hausdorff 视角）对该机制给出了严格的理论解释。两个子块 $s$ 和 $s'$ 之间的失真程度，可以用其归一化代码单元之间的结构距离和语义相似度来进行度量（记为 $\Delta(s, s')$）。

在局部奖励贡献满足 Lipschitz 连续性的假设下，论文推导出了两个核心定理：

1. **偏差控制（Theorem 4.1）**：若同一分组内的每个子块与锚点的对齐误差被控制在 $\epsilon$ 以内，则用锚点归类替代完全一致的代码匹配，所引入的局部信用偏差上界仅为 $O(L\epsilon)$。

2. **方差收敛（Theorem 4.2）**：当把子块聚合进包含 $m$ 个样本的优势组中计算均值时，其局部优势估计量的均方误差（MSE）满足：




{% raw %}$$ \mathrm{MSE}(\hat{A}_{i}^{D}) \leq O(L^{2}\epsilon^{2}) + O(\sigma_{\xi}^{2}/m) $${% endraw %}



其中 $\sigma_{\xi}^{2}$ 代表非因果的轨迹随机噪声。这一数学结论直观地说明了为什么 DiDPO 要追求“大容量分组”：**更大的 $m$ 能够以反比速度压制单步探索中的环境噪声，从而让策略梯度的更新更加平稳精准。**

### 实验评测：长程代码与算法竞赛的突破

为了全面检验 DiDPO 的能力，研究团队构建了涵盖 8 个标准及竞赛级代码基准的评测体系。评测基准既包括基础的函数生成任务（HumanEval、MBPP），也包含真实编程问题（APPS、LiveCodeBench），以及对多步算法推理要求极高的算法竞赛基准（LeetCode、USACO、OJBench、ICPC）。

实验以开源表现卓越的代码大模型 Qwen2.5-Coder-7B 和推理能力强劲的 Qwen3.5-4B 作为基础骨干。训练前首先经过轻量级的 Thought-Action 格式冷启动微调（SFT），随后分别在基线方法（PPO、GRPO、DAPO、GiGPO 等）与 DiDPO 上进行强化学习训练。

综合实验结果展现出显著的性能跃迁：

在全量基准测试中，DiDPO 均取得了最高分。以 **Qwen2.5-Coder-7B** 为底座时，DiDPO 的综合平均通过率达到了 **48.4%**，相较于此前表现最优的 Agent 状态分组算法 GiGPO（44.2%）高出 4.2 个百分点，相较于经典 GRPO（42.8%）提升了 5.6 个百分点，相对增幅超过 10%。而在 **Qwen3.5-4B** 上，DiDPO 达到了 **58.6%**，相比 GiGPO 取得了 4.9 个百分点的绝对领先。

更具说服力的是在硬核算法竞赛任务上的表现。在极其考验复杂代码设计与边界处理的 **USACO** 基准上，7B 规模的 DiDPO 取得了 **15.6%** 的通过率，而同等设置下的 GRPO 仅为 6.8%，性能实现翻倍；在 LeetCode 上，DiDPO 也从 GRPO 的 26.2% 提升至 32.8%。即便对比未公开权重的庞然大物 GPT-5.5，DiDPO 也将 7B 开源模型与其的性能差距从初始底座的 56.4% 骤降至 43.3%。

<img src="/images/2608.07147v1/dynamics.webp" alt="训练动态与子 Diff 类型的语义演变分析" style="width:90%; max-width:700px; margin:auto; display:block;">

### 机制透视：模型究竟学会了什么？

为什么细粒度的子 Diff 归因能带来如此显著的提升？研究团队通过对训练动态的跟踪给出了微观层面的证据。

从训练曲线（Figure 4 左）来看，在训练前 20 步，所有算法的表现提升幅度大致相当，这主要是由全局轨迹优势 $A^E$ 推动的基础学习。但到了 40 步之后，代码编辑的动作策略日趋多元化，GiGPO 等依赖宏观状态匹配的方法开始陷入平台期；而 DiDPO 凭借深入 Diff 内部的细粒度信用分配，依然能够准确提炼有效子动作，保持强劲的性能攀升。

更具洞察力的是对优势组类别的演进分析（Figure 4 右）。研究人员利用前沿模型对训练过程中涌现的子块类型进行分类标注，结果发现：

* 随着训练的深入，代表实际逻辑核心的 **功能代码块（Block，如主函数体、核心控制循环）** 在所选锚点中的占比持续稳步攀升；

* 与之相对，短小孤立的代码碎屑（Fragment）以及单纯的结构骨架（Scaffold，如 Import 引入、类定义占位符）占比显著下降；

* 无意义的代码行（Other，如纯空行、简单日志）更是迅速被边缘化。

这表明，可成团性评分机制（GS）在实际运行中准确完成了其设计初衷：**自动滤除低价值的语法碎屑，精准捕捉决定程序成败的核心逻辑片段，并为其分配梯度信用。**

### 计算开销：只加 2.3% 耗时的高性价比方案

对于强化学习算法而言，理论精巧往往容易伴随昂贵的算力代价。如果细粒度归因需要部署额外的 Critic 网络，显存和反向传播开销都将成倍增长。

<img src="/images/2608.07147v1/abla.webp" alt="消融实验与每步训练时间开销分解" style="width:90%; max-width:700px; margin:auto; display:block;">

DiDPO 完全避开了额外的参数网络。其所有的锚点提取与优势计算，全部复用多 Rollout 采样原本就会生成的代码输出文本。耗时分解测试（Figure 5 右）显示，相较于最基础的 GRPO，DiDPO 在每一步训练迭代中仅增加了 **约 2.3% 的额外时间开销**。

这微小的耗时绝大部分消耗在基于相似度矩阵的锚点初筛阶段，而后序贪心求优的耗时极短，几乎可以忽略不计。更为关键的是，这一微弱的开销仅存在于训练阶段，在模型推理与部署落地时，模型采用完全标准的代码自回归生成，不会引入任何推理延迟。

以 2.3% 的微量训练代价，换取全任务 5% 以上、硬核任务超 100% 的效果增益，这一极高的投入产出比为大规模 Agentic RL 的工业化部署铺平了道路。

### 总结与展望

在面向代码的智能体构建中，代码不仅仅是一串线性的文本 Token，它本质上包含着清晰的语法树、模块化的逻辑块以及因果分明的执行路径。以往直接将通用多智能体强化学习机械照搬到编程任务的做法，忽略了代码 Diff 本身丰富的高阶拓扑结构。

DiDPO 的启发性在于，它证明了**无需构建复杂的 Value 模型，仅凭对代码动作几何与语义特征的挖掘，就能以近乎免费的方式榨取出极高价值的局部奖励信号**。随着软件开发 Agent 逐渐从单一函数的短平快生成，走向横跨几十个文件、数十轮交互的长程仓库级修复，这种能够精准定位“关键几行改动”的细粒度信用分配范式，正在成为代码智能体训练演进的重要拼图。同时，团队开源的 verl-code 框架，也为后续探索多轮代码交互强化学习的开发者提供了一套开箱即用的技术底座。
