---
layout: default
title: "清华提出DynamicRubric：评估器协同进化，8B打破70B上限并全量落地微信"
description: "针对这一瓶颈，清华大学联合腾讯团队提出了一种全新的评估器与策略协同进化框架—— DynamicRubric 。该研究从概率分配的理论视角，严格证明了候选回复之间的“相对分差”正是驱动策略概率质量转移的本质信号；并据此设计了基于当前候选回复集合动态生成打分细则（Rubric）的双层协同演进机制。"
arxiv_id: "2607.20083"
paper_published: "2026-07-22"
published_at: "2026-10-03T13:15:07.852414+08:00"
topics:
  - "基础模型"
tags:
  - "基础模型"
  - "AI论文解读"
related_tutorials:
  - "to-add-is-machine-to-delete-is-human-measuring-and-mitigating-deletion-avoidance"
  - "dont-offer-what-cant-be-done-deterministic-executability-gating-for-llm-skill-se"
  - "bunraku-turning-a-single-illustration-into-an-editable-live2d-character"
  - "sg-wam-self-guided-world-modeling-in-geometry-aware-policy-space"
seo_title: "Co-Evolving LLM Evaluators and Policies via DynamicRubric"
---

<p class="paper-original-title" lang="en">Co-Evolving LLM Evaluators and Policies via DynamicRubric</p>

在大语言模型进入后训练（Post-training）深水区的今天，强化学习与偏好对齐已经成为激发模型高阶能力的核心支柱。然而，业界几乎所有对齐流程都在面临一个共同的隐形天花板：随着策略模型（Policy）能力的快速提升，它针对同一个提示词（Prompt）采样出的多个候选回复，其表面质量变得越来越接近。

> ArXiv URL：https://arxiv.org/abs/2607.20083

当候选回复都同样流畅、结构完整、礼貌得体时，传统的标量奖励模型（Reward Model）或 LLM-as-a-Judge 往往会出现评分差距塌陷（Score Gap Collapse）。评估器给出的分值差异微乎其微，甚至在微小的扰动下发生颠倒。这种模糊且充满噪声的相对反馈，直接导致策略优化失去准确的梯度指引，甚至引发奖励作弊（Reward Hacking）与过优化。

针对这一瓶颈，清华大学联合腾讯团队提出了一种全新的评估器与策略协同进化框架——**DynamicRubric**。该研究从概率分配的理论视角，严格证明了候选回复之间的“相对分差”正是驱动策略概率质量转移的本质信号；并据此设计了基于当前候选回复集合动态生成打分细则（Rubric）的双层协同演进机制。实验显示，采用 8B 参数底座训练的 DynamicRubric 评估器，不仅在偏好评估上超越了 70B 标量奖励模型和 235B 静态量规生成器，更作为核心对齐技术全量部署于微信搜索 AI 问答场景，每天稳定承接数千万次线上请求。

### 为什么大模型变强后，评估器反而不会打分了？

在典型的 RLHF 或强化学习探索过程中，模型会根据输入的 Prompt $x$ 采样出一组候选回复 $\mathcal{C} = \{y_1, \dots, y_K\}$。在训练初期，模型生成的回复良莠不齐，评估器很容易挑出好坏。但经过几轮迭代后，模型生成的各个选项在通用指标（如帮助性、安全性、逻辑清晰度）上均已饱和。此时，真正决定回复优劣的，往往变成了极为特异且隐蔽的细节：某个微小约束是否被遗漏、多步推导中的某一步假设是否合理、语气是否契合特定意图等。

如果沿用传统的静态量规（Static Rubric）——即在看到候选回复之前，仅根据 Prompt 预先列出评分标准——很容易漏掉那些在具体样本中才显现的关键差异。而使用单一标量打分的传统奖励模型，更容易在面对质量极度相似的文本时退化为随机猜测，导致相对分差趋近于零。

<img src="/images/2607.20083/mainfigv4.webp" alt="DynamicRubric 框架总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了解决这个问题，研究团队将评估视作一种“因材施教”的动态对比过程，提出了图 1 所示的 DynamicRubric 架构。它将评估器拆解为可学习的量规生成器（DR-Generator）和冻结的细则检验器（DR-Verifier）。核心思想在于：评估器必须以策略模型当前吐出的候选回复集合为条件，针对性地找出区分这几篇文本的“决定性差异”，生成带有权重的二值细则项，然后再打出具备高辨识度的相对分数。

### 概率分配视角：相对分差为何是优化的本质？

为了给上述设计建立坚实的理论基础，论文从局部概率分配的视角切入，重新审视了强化学习后训练的数学本质。

给定 Prompt $x$ 与策略模型采样的候选集 $\mathcal{C}$，策略模型在候选集内部的局部条件概率可以归一化表示为权重向量 $\boldsymbol{\alpha} = (\alpha_1, \dots, \alpha_K)$。在此局部空间内，局部期望奖励定义为各个候选回复得分的加权和：




{% raw %}$$

\mathcal{J}^{\mathrm{local}}_{x,\mathcal{C}}(\boldsymbol{\alpha};E) = \sum_{i=1}^{K}\alpha_i E(x, y_i)

$${% endraw %}



当优化算法试图调整策略参数，使得概率质量从劣质回复 $y_j$ 转移到优质回复 $y_i$ 时（记微小位移方向为 $\mathbf{d}_{i\leftarrow j}$），局部目标函数沿该方向的方向导数恰好满足：




{% raw %}$$

\left.\frac{\mathrm{d}}{\mathrm{d}\varepsilon}\mathcal{J}^{\mathrm{local}}_{x,\mathcal{C}}(\boldsymbol{\alpha}+\varepsilon\mathbf{d}_{i\leftarrow j};E)\right\vert{}_{\varepsilon=0} = E(x, y_i) - E(x, y_j)
$${% endraw %}



这个简洁的等式揭示了一个被长期忽视的事实：在候选集内部，将概率从一个回复重分配给另一个回复的方向性收益，严格等于评估器对二者判定的相对分差。相对分差并非只是评估结果的副产物，它本身就是驱动策略更新的优化信号强度。

由此，论文进一步证明了两个关键推论：

1. **输入结构的表征能力界限**：相比于仅依赖输入提示词打分的独立评估器 $\mathcal{E}_{\mathrm{prompt}}$，能够同时感知当前候选全集的评估器 $\mathcal{E}_{\mathrm{set}}$，在最小化排序期望损失上具有严格更优的下界。也就是说，只有“看完全部选项再定标准”，才能在信息论层面上保留最具分辨力的得分差距。

2. **分布偏移导致评估器必须协同演进**：随着策略模型参数 $\theta$ 的迭代更新，采样的候选集分布不断发生偏移。在旧分布上表现良好的评估器，面对新策略生成的高质量候选时，分差必然重新塌陷。因此，评估器必须与策略模型保持双层优化（Bilevel Optimization）同步推进。

### DynamicRubric 的核心机制与双目标演进

基于这一理论推导，DynamicRubric 构建了一个交替迭代的协同演进闭环。

在每一轮演进中，DR-Generator 接收提示词 $x$ 与策略模型采样的候选集 $\mathcal{C}$，自适应输出 $M$ 个加权二值量规项。每个量规项包含一段具体的考察标准和对应的权重 $w_m$。随后，保持参数冻结的 DR-Verifier 分别对每个候选回复 $y$ 在各项准则下做出“是/否”（1/0）的明确判断 $v_m(y;\mathcal{C})$。最终，候选回复的综合得分通过加权平均得出：




{% raw %}$$

E_{\phi}(x,y\mid\mathcal{C})=\frac{\sum_{m=1}^{M}w_{m}v_{m}(y;\mathcal{C})}{\sum_{m=1}^{M}w_{m}}

$${% endraw %}



为了让 DR-Generator 学会生成最切中要害、最具辨识度的准则，研究团队为其设计了两个互补的优化目标，并通过 GRPO（Group Relative Policy Optimization）进行强化学习训练：

- **候选区分度目标（Discriminability Objective, $\mathcal{J}_{\mathrm{disc}}$）**：量规不能是毫无区分度的套话。如果一项细则所有回复都满足或都不满足，它的信息量就是零。该目标惩罚方差过小的量规，最大化项内得分方差 $\bar{v}_m(1-\bar{v}_m)$，强制要求量规在当前候选回复间拉开区分度。

- **锚定对齐目标（Anchor Objective, $\mathcal{J}_{\mathrm{anchor}}$）**：仅有区分度可能诱导评估器寻找无意义的表面特征（如字数、特定标点）。因此，算法引入包含排序标注的高质量基准偏好数据集（如 Nectar）作为锚点，使用类似于 ListNet 的排序损失，强制要求动态量规在已知偏好的样本上计算出的得分排名与真实偏好完全一致。

通过联合优化这两个目标，DR-Generator 既能敏锐捕捉当前策略输出的特定分歧，又不会偏离人类真实偏好的主轴，从而为下游策略更新构筑了下界更有保障的相对分差。

### 协同进化的威力：8B 评估器超越 70B 标量奖励模型

为验证协同演进的实际增益，研究团队以 Qwen3-8B 为基础底座分别初始化了生成器与策略模型，并在多个主流基准上进行了系统测试。

在评估器本身的辨识能力评测中，仅有 8B 参数的 DR-Generator-8B 展现出了跨量级的表现。对比基线包括未经动态微调、参数量大 4 倍的 Qwen3-32B 动态生成器，以及参数量达 27B 的 Skywork-Reward 和 70B 的 Nemotron-70B 标量奖励模型。

<img src="/images/2607.20083/continuous_generator.webp" alt="评估器与策略模型连续协同进化曲线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

<img src="/images/2607.20083/continuous_policy.webp" alt="评估器与策略模型连续协同进化曲线2" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

结果表明，DR-Generator-8B 在所有成对偏好与列表偏好基准上均超越了 32B 的零样本动态生成器，并稳定击败了 70B 的标量奖励模型。特别值得注意的是，若仅向标量奖励模型输入完整的候选集，后者的性能提升非常有限甚至偶有下降；这说明仅仅“增加候选集输入”并不能解决问题，核心在于能否通过可微分的“细则生成 + 显式验证”结构，将全局的宏观好坏解构为可追溯的具体事实。

图 2 进一步揭示了持续协同进化（Continuous Co-Evolution）的动态特性。实验对比了“评估器滞后（用旧策略数据训练）”与“评估器随策略同步更新”的差距：如果评估器停止随策略进化（即 $G_1 \rightarrow \tilde{G}_2$），策略优化很快陷入停滞；而当评估器持续在最新策略生成的难分辨样本上迭代更新时（$G_1 \rightarrow G_2$），它能不断为后续策略 $P_2$ 注入具有强梯度指引价值的高方差信号，从而打破传统后训练的性能平台期。

### 能力外溢：为何开放域对齐没有损害代码与数学推理？

在以往的大模型后训练实践中，“对齐税”（Alignment Tax）是一个广为人知的困境：当模型针对开放式对话或主观偏好进行高强度对齐优化后，往往会导致数学推导、科学问答和代码编写等客观推理能力的大幅回退。

然而，由 DynamicRubric 指导优化的策略模型却打破了这一常态。

<img src="/images/2607.20083/general_bench.webp" alt="策略模型在可验证推理基准上的泛化表现" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从图 3 的测试结果可以看出，随着 DynamicRubric 的轮次推进，策略模型在完全没有显式针对推理数据微调的前提下，在数学（Mathematics）、代码（Coding）以及科学问答（Scientific QA）等客观可验证基准上均呈现出平稳且一致的提升态势。

这种逆向泛化现象的深层原因在于量规评分的细粒度属性。以往标量奖励模型之所以会损害推理能力，是因为其粗暴的分值反馈容易让策略模型学到“更长的篇幅、更客气的废话”这类投机取巧的模式，劣化了底层的紧凑推导逻辑。而 DynamicRubric 强迫评估器将好坏拆解为具体的因果判断——例如“是否准确执行了反事实前提”“多步推导是否存在因果断裂”。在这种高度细致的监督下，策略模型在探索阶段获得的强化信号不是模糊的语感倾向，而是严格的逻辑遵循与约束满足能力。这种底层能力的提升自然外溢到了可验证的代码与严谨推理场景中。

### 工业级检验：千万级日调用场景下的全面替代

算法在学术评测集上的优异数字固然可喜，但面对工业级真实流量的检验往往更加苛刻。在真实搜索场景下，用户的意图极度多元且长尾，微小的幻觉或结构性瑕疵都会立刻反映在用户的即时行为中。

论文报告了 DynamicRubric 在**微信搜索 AI 问答场景**中的全量落地成果。在线上环境，研发团队将该技术应用于千万级活跃用户的真实生产管线，基座模型为微信自研的 WeLM-V4-80B-A3B，对比基线为上一代正在生产环境中运行的成熟服务模型。

经过严密的线上双盲 A/B 测试，采用 DynamicRubric 演进优化的模型展现出了显著的业务增益：在搜索总请求量、用户页面停留时长以及绝对正向交互行为（如点赞、采纳、积极互动）等核心业务指标上，均取得了具备统计显著性的全面正向增长。

基于坚挺的线上收益与稳定性表现，该模型已经实现 100% 全量切流，全面替换了原有的生产模型，稳定承载每日数千万次的线上高并发访问。在生产系统的长效运维中，动态量规生成器还与线上锚点机制结合，通过定期回流线上新增的优质人工校准样本，实现策略模型的锚定持续自进化（Continual Self-Improvement），彻底改变了过去静态奖励模型上线后容易“逐渐钝化”的被动局面。

### 从静态判别到动态共生

大语言模型后训练的发展历程，本质上是一部“如何给模型提供更高质量梯度”的演进史。从最初依赖人工昂贵标注的成对偏好，到离线预训练的标量奖励模型，再到近期的过程奖励模型（PRM），技术的重心始终在向“更细致、更可控”的方向倾斜。

DynamicRubric 的实践表明，评估器不应当是一个悬挂在策略模型头顶的、静止不变的“考官”。随着考生的水平日新月异，曾经具有辨识度的考题必然失去效力。让评估器进入候选集合的微观语境中，动态制定具有强区分度的考核细则，并跟随策略模型一同迭代蜕变，正是解决后训练信号塌陷的关键钥匙。

这种评估器与策略模型协同演进的范式，不仅大幅拉低了超高质量对齐所需的算力门槛——让小参数量评估器战胜数倍于己的庞大模型——更为通往自主反思、自我进化的下一代自适应大语言模型体系铺平了工程与理论道路。
