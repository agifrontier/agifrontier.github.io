---
layout: default
title: "美团与南大等提出OCSD：剥离回放模板混杂，Agent强化学习提升14分"
description: "针对这一根本隐患，研究团队提出了 观察校准自蒸馏（Observation-Calibrated Self-Distillation, OCSD） 。该方法通过构建两套结构完全对齐、仅差真实环境观测的回放视图，精准剔除回放脚手架引起的固有偏移，提取出纯粹的“环境观察残差”。"
arxiv_id: "2608.04788"
paper_published: "2026-08-05"
published_at: "2026-09-09T13:15:08.606362+08:00"
topics:
  - "AI Agent"
  - "强化学习"
tags:
  - "GRPO"
  - "OCSD"
  - "OPSD"
  - "agentic reinforcement learning"
  - "observation residual"
  - "observation-ablated replay view"
related_tutorials:
  - "reinforcement-learning"
  - "mera-model-evolution-and-routing-with-skill-adaptation-for-agentic-systems-at-sc"
  - "incorporating-self-rewriting-into-large-language-model-reasoning-reinforcement"
  - "the-landscape-of-agentic-reinforcement-learning-for-llms-a-survey"
---

<p class="paper-original-title" lang="en">Agentic Reinforcement Learning with Observation-Calibrated Self-Distillation</p>

<img src="/images/2608.04788v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在大语言模型（LLM）智能体的训练中，强化学习（RL）已成为推动模型自主探索与复杂决策的核心技术。然而，面对 Web 浏览、具身控制或多跳检索等长流程任务，目前以 GRPO（Group Relative Policy Optimization）为代表的算法普遍面临一个底层痛点：环境奖励往往极其稀疏，且通常只在整条轨迹执行完毕后给出。这就意味着，一整条包含几十步交互、生成了成百上千个 Token 的轨迹，所有 Token 都会被平铺直叙地分配一个相同的轨迹级优势值（Advantage）。至于其中哪一步思考真正切中要害，哪一个动词或实体选错了，全局奖励根本无法提供细粒度的指导。

> ArXiv URL：https://arxiv.org/abs/2608.04788v1

为了给智能体提供更密集的单步反馈，策略内自蒸馏（On-Policy Self-Distillation, OPSD）应运而生。这类方法的核心逻辑是“事后诸葛亮”：在模型采样生成动作后，构造一个包含未来环境反馈的“特权回放视图（Privileged Replay View）”，让模型以教师身份重新评估自己刚才写下的 Token。理论上，未来真实发生的观测能够帮助模型识别当下的失误。但近期由南京大学、美团、北京大学、复旦大学和华东师范大学等机构联合发布的研究指出，现存特权回放打分机制存在严重的“混杂效应（Confounding Issue）”——回放视图对 Token 带来的打分提升，很大程度上并不是未来观测信息的功劳，而是由于回放提示词模板、对话脚手架（Scaffold）自身的格式偏置所导致的虚假增强。

针对这一根本隐患，研究团队提出了**观察校准自蒸馏（Observation-Calibrated Self-Distillation, OCSD）**。该方法通过构建两套结构完全对齐、仅差真实环境观测的回放视图，精准剔除回放脚手架引起的固有偏移，提取出纯粹的“环境观察残差”。在 ALFWorld、WebShop 和 Search-QA 等基准测试中，覆盖 Qwen3 多个尺度的实验表明，OCSD 显著超越了标准 GRPO 及现有的各种蒸馏基线，在 ALFWorld 上相较 GRPO 成功率最高提升达 14.0 个百分点，且仅增加了 1.40% 的微小训练耗时。

<img src="/images/2608.04788v1/fig_motivation_v6.webp" alt="从结构对齐的回放视图中提取观察残差" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 特权回放的隐蔽陷阱：模板偏差与信息混杂

在多轮交互任务中，智能体在第 $k$ 步根据历史上下文 $h_k$ 输出动作 $y_k$，随后环境返回下一个观察 $o_{k+1}$。如果动作执行失败，比如模型输出了 `take the mug from shelf 1`，而环境紧接着提示 `there is no mug on the shelf`，人类很自然地认为：只要把这条提示塞进上下文重新回放，模型就应该精准惩罚刚才妄图去拿杯子的错误举动。

这正是后验特权回放的直觉假设：模型在特权教师视图下的对数概率与原始学生视图下的对数概率之差 $\delta_{i,k,t}$，理应代表环境反馈带来的支撑度。然而，图 1 揭示了这一假设在现实中往往不成立。为了在回放中容纳未来的观测，研究者必须重新组织上下文结构，加入诸如“在看到环境反馈后评估动作”等引导词和回放脚手架。令人惊讶的是，即使研究者将真实的未来环境观测完全抹去（即构建图中的 Observation-Ablated 视图），仅保留回放脚手架格式，模型对许多 Token 给出的打分依然发生了剧烈偏移。

在没有真实反馈的消融视图下，类似 `I`、`should`、`from` 这类虚词，甚至部分动词的概率变化，居然与包含了真实环境反馈的完整回放视图（Full View）高度雷同。这证明了一个关键事实：未经校准的特权回放支持度，本质上是一个混杂信号。它既包含了未来观测的信息，也掺杂了回放脚手架本身诱导的模型先验偏移。如果直接拿这个混杂信号来更新策略，模型学到的可能不是“听从环境反馈”，而是“迎合回放模板的语言偏好”。

<img src="/images/2608.04788v1/fig_token_diagnosis_v16.webp" alt="Qwen3-1.7B在ALFWorld训练过程中的细粒度诊断分析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了量化这一混杂效应，研究人员在 ALFWorld 任务上进行了细致的诊断分析。如图 2 所示，他们分析了 Token 打分变化中“残差贡献比例” $D$ 的分布。实验发现，回放脚手架所带来的虚假变化在功能词（如介词、连词）和通用动词中极为显著；唯有那些直接与环境实体交互的操作词（如拿取、清洗）和物体名称（如杯子、书架），才对真实的未来观测展现出特异性响应。

更关键的证据来自区分动作合法性的受试者工作特征曲线下面积（AUROC）测试。在区分智能体生成的动作是否被环境判定为合法动作时，单纯依赖全量回放视图（Full）的打分表现并不理想；而将全量视图与消融视图相减得到的残差信号（Residual），其区分合法与非法动作的 AUROC 显著高于前者。这直接印证了作者的核心推论：去除脚手架干扰后的纯净观察信号，才真正与局部的环境反馈同频共振。

### 结构对齐与符号保真：OCSD 的方法实现

认清了混杂效应的本质后，OCSD 的整体架构设计显得格外克制且优雅。它并没有试图设计极其复杂的提示工程来强行消除模板影响，而是利用因果推断中常见的控制变量思想，通过结构对齐的“差分”操作来抵消系统误差。

<img src="/images/2608.04788v1/methodology_v6.webp" alt="OCSD算法整体框架与执行流程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整个流程主要由三大核心模块组成：

首先是**双视图差分提取观察残差**。针对智能体采样的轨迹，算法同步构建两个在语法格式和结构上完全一致的回放教师：一个是包含真实后续观测的完整教师（Full Teacher, $\pi_F$），另一个则是删去真实观测、仅保留占位结构的消融教师（Observation-Ablated Teacher, $\pi_A$）。对于生成的任意 Token $y_{i,k,t}$，两者对数似然的差值即为纯粹的环境观察残差：




{% raw %}$$e_{i,k,t} = \log\pi_F(y_{i,k,t} \mid h_{i,k,t}, E^F_{i,k}) - \log\pi_A(y_{i,k,t} \mid h_{i,k,t}, E^A_{i,k})$${% endraw %}



由于两个视图共享了完全相同的脚手架格式，减法操作自然对消掉了脚手架引入的打分偏移。为了便于后续稳定调控梯度，研究团队进一步将残差映射至有界区间 $q_{i,k,t} = \tanh(e_{i,k,t} / 2) \in [-1, 1]$。当真实反馈倾向于鼓励该动作时，$q$ 值为正；反之若反馈证明该动作为无效尝试，$q$ 值则为负。

其次是 **NLL 引导的关键步骤筛选**。智能体在长程环境中每一步的决策并非同等重要，大量常规移动或确定性动作并不需要过度介入。如果对轨迹中的每一个 Token 都强行计算双视图回放，不仅浪费计算算力，还可能在低熵平庸步骤引入微小的估计噪声。因此，OCSD 在交互轨迹中计算旧策略生成各步动作时的平均负对数似然（NLL）：




{% raw %}$$u_{i,k} = -\frac{1}{T_{i,k}} \sum_{t=1}^{T_{i,k}} \log\pi_{\theta_{\text{old}}}(y_{i,k,t} \mid h_{i,k,t})$${% endraw %}



算法仅挑选不确定性最高的前 $\rho$ 比例（实验默认取 20%）的步骤作为干预对象。在这些真正让模型感到“犹豫不决”的分水岭节点上，环境反馈的校准价值最大。

最后是**方向保真的优势调制**。如何将 Token 级别的细粒度残差融合进原本的强化学习目标中？以往部分自蒸馏工作直接将教师的 KL 散度作为辅助 Loss，或者粗暴地颠覆优势值的正负，这往往导致训练极不稳定。OCSD 采取了“大方向服从轨迹，小步幅局部微调”的策略。在更新被选中的交互步时，优势函数被重构为：




{% raw %}$$\widehat{A}^{\text{OCSD}}_{i,k,t} = \widehat{A}_i \left[ 1 + \beta \operatorname{sgn}(\widehat{A}_i) q_{i,k,t} \right]$${% endraw %}



其中 $\widehat{A}_i$ 是 GRPO 依据最终任务成败计算出的全局优势值，$\beta$ 为调制权重。通过引入 $\operatorname{sgn}(\widehat{A}_i)$ 项，更新公式确保了无论内部 Token 的 $q$ 值如何振荡，该 Token 所获优势值的正负符号始终与整条轨迹的成败方向严格保持一致。如果整条轨迹成功（$\widehat{A}_i > 0$），环境观测积极的 Token 会获得额外激励（放大更新步长），而逻辑不连贯的 Token 激励被削弱，但绝不会被逆转为负；反之，若整条轨迹失败，更有问题的 Token 也会受到更沉重的惩罚。

### 跨任务与跨规模的全面超越

为了验证 OCSD 的泛化能力与鲁棒性，研究团队在三种截然不同的智能体基准上展开了严格测试：侧重具身实体操作的 ALFWorld、模拟复杂电商决策的 WebShop，以及极具挑战性的多跳与事实型检索问答 Search-QA。实验模型横跨了 Qwen3 的 1.7B、4B 和 8B 三个关键尺寸。


| 模型尺寸 | 评测基准 | GRPO 基础成功率 | RLSD 对比 | SDAR 对比 | OCSD（本文方案） | 相比 GRPO 绝对增益 |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen3-1.7B** | ALFWorld | 46.6% | 41.7% | 44.5% | **55.5%** | **+8.9%** |
| | WebShop | 48.4% | 49.0% | 52.1% | **54.4%** | **+6.0%** |
| | Search-QA | 40.9% | 41.0% | 41.8% | **43.1%** | **+2.2%** |
| **Qwen3-4B** | ALFWorld | 70.6% | 68.8% | 79.2% | **82.8%** | **+12.2%** |
| | WebShop | 69.5% | 71.3% | 68.5% | **73.7%** | **+4.2%** |
| | Search-QA | 45.5% | 46.3% | 45.9% | **47.5%** | **+2.0%** |
| **Qwen3-8B** | ALFWorld | 73.2% | 78.6% | 85.4% | **87.2%** | **+14.0%** |
| | WebShop | 75.5% | 76.1% | 74.2% | **78.1%** | **+2.6%** |
| | Search-QA | 48.7% | 48.4% | 47.0% | **49.1%** | **+0.4%** |

实验数据清晰展现了现有蒸馏方案的局限与 OCSD 的统治力。未经校准直接引入特权蒸馏往往会带来负面效果：在 Qwen3-1.7B 模型下，直接结合自蒸馏的 RLSD 在 ALFWorld 上的成功率不仅没有提升，反而暴跌至 41.7%，大幅落后于仅靠环境奖励的朴素 GRPO（46.6%）。这直接印证了回放脚手架噪声在小模型上的破坏力。而 SDAR 虽然在 8B 模型上表现不俗，但在多项任务的中小模型适配中依然暴露出不稳定性。

反观 OCSD，在全部三种模型规格、三大类任务测试中，均取得了稳居第一的综合表现。尤其是在交互步数极多、容错率极低的具身环境 ALFWorld 中，随着模型参数从 1.7B 扩大到 8B，OCSD 展现出了更强的放大效应，相较 GRPO 的增益从 8.9 个百分点持续扩大到 14.0 个百分点，最终实现了 87.2% 的高成功率。在具备明显分布外偏移（OOD）特性的 Search-QA 评测中，OCSD 在未见过的检索数据集上同样维持了最高的分数，证明这种校准不是在训练分布上强行拟合，而是让模型真正获得了对环境线索更健康的归纳能力。

<img src="/images/2608.04788v1/fig_training_dynamic_v8.webp" alt="ALFWorld在不同模型规模下的训练动态曲线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从图 4 呈现的训练收敛动态来看，训练早期各基线方案曲线交织缠绕，但在训练步数过半后，OCSD 的成功率曲线持续稳步爬升，并在最终阶段与 GRPO 及其他蒸馏基线拉开了非常显著且稳定的差距。

### 为什么有效？残差因果性与算力开销剖析

为了彻底排除“性能提升仅仅是因为增加了额外文本输入”这一怀疑，研究人员设计了一组极具说服力的“观察置换实验（Observation Swapping Analysis）”。

他们保持回放结构与模型所有输入完全不变，唯一的操作是将 Full View 中的真实未来观测，随机替换为同轨迹中其他无关步骤的观测文本。如果模型只是单纯从冗长回放文本中提取某种通用特征，那么置换后的残差应当仍然具备一定的区分能力。

<img src="/images/2608.04788v1/fig_deep_v7.webp" alt="真实观测与置换观测下的动作合法性AUROC诊断对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

然而如图 5 所示，在真实的未来环境反馈下，残差信号区分动作有效性的 AUROC 分别高达 0.803、0.668 和 0.664；一旦将其置换为不相关的假反馈，AUROC 骤降至 0.542、0.484 和 0.552——这几乎已经退化到了完全随机盲猜的水准。这项消融实验极具说服力地证明：OCSD 所捕捉并利用的残差，其判断力几乎百分之百来自于“动作与随之产生的具体环境反馈”之间的真实因果对应，而非虚假的相关性。

除了学习质量的提升，工程落地的代价同样是强化学习算法必须权衡的核心指标。由于 OCSD 需要在前向计算中调用双视图回放，许多人可能会担忧这是否会成倍拖慢训练速度。

<img src="/images/2608.04788v1/fig_time_v5.webp" alt="Qwen3-4B在ALFWorld单次迭代耗时分布分解" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

图 6 给出了在 8 张 A100 GPU 上针对 Qwen3-4B 模型单次迭代耗时的精细拆解。可以看到，占据绝大部分训练耗时的依然是与环境反复交互的 Rollout 采样、参考模型 KL 计算以及底层的梯度反向传播等 GRPO 固有流程（共计约 544.7 秒）。由于 OCSD 巧妙地采用了 NLL 不确定性筛选机制，只对整条轨迹中 20% 的关键交互步进行回放推理，双视图评分耗时仅为 7.52 秒，残差调制耗时仅 0.12 秒。两个专有模块合计仅增加了 7.64 秒的微弱额外负载，占整体流程的比例仅为 1.40%。用几乎可以忽略不计的计算代价，换取全任务范围内的两位数胜率提升，OCSD 展现出了极其诱人的工程实用性。

### 结语

在追求通用智能体（Generalist Agents）的演进道路上，强化学习正从原本粗放的“黑盒试错”逐渐走向“细粒度 Credit 分配”。此前学术界普遍认识到了后验回放信息的宝贵价值，却长期忽视了将这套信息具象化表达时所伴生而来的格式与模板混杂。

OCSD 的价值在于它精准指出了特权回放机制中的这一认知盲区：回放不等于无偏监督。通过构建结构严密对齐的差分视图，研究团队不仅用极简的数学形式抹平了提示脚手架的系统性偏置，更在保全强化学习全局探索方向的前提下，将未来观测转变为精准打击错误动作的微观手术刀。这项研究不仅为长流程大模型智能体的高效对齐提供了坚实的方法论支撑，同时也为后续探索如何更加严谨、无偏地利用交互环境中的后验反馈，树立了极具启发性的范式。
