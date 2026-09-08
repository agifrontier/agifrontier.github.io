---
layout: default
title: "SKILLER：自然语言强化学习优化技能，让4B模型性能反超9B！"
description: "为了解决这种典型的“模型不匹配”（Model-Mismatch）难题，来自哈工大、上海人工智能实验室、上海交通大学以及清华大学的研究团队提出了 SKILLER 。"
arxiv_id: "2608.10538"
paper_published: "2026-08-11"
published_at: "2026-09-08T13:15:08.162312+08:00"
topics:
  - "强化学习"
tags:
  - "Executor-specific skills"
  - "LLM actor-critic"
  - "Language-level RL"
  - "Natural-language-driven RL"
  - "Reusable skill extraction"
  - "SKILLER"
related_tutorials:
  - "natural-language-actor-critic-scalable-off-policy-learning-in-language-space"
  - "rangefactory-scalable-construction-of-multi-hop-cyber-ranges"
  - "reinforcement-learning"
  - "scribes-web-scale-script-based-semi-structured-data-extraction-with-reinforcemen"
---

<p class="paper-original-title" lang="en">SKILLER: Language-Level Reinforcement Learning for Reusable Skill Extraction in Small Language Models</p>

<img src="/images/2608.10538v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在大模型智能体（Agent）系统的构建中，“智能体技能”（Agent Skills）正在迅速成为一种标准的基础设施。从 Anthropic 提出的框架到当下流行的各类代码与通用智能体 Harness，技能早已超越了简单的 Prompt 模板，演变为一种封装了严密过程性知识、工具调用规范与领域专有经验的标准化载体。它就像是给大模型配置的一套标准作业程序（SOP），通过持续约束模型的行为空间，保障复杂任务能够在多步交互中稳定、高质地复现。

> ArXiv URL：https://arxiv.org/abs/2608.10538v1

这种高可靠性在过去几乎完全依附于顶尖的闭源前沿大模型。然而，昂贵的 API 推理成本让大规模落地变得异常沉重。随着开源小尺寸模型在消费级硬件上的可用性大幅提升，业界迫切希望将这些结构化技能迁移到端侧或轻量模型上，从而换取极高的成本效益比。然而，直接将为顶尖闭源模型定制的技能灌入小模型，往往会导致严重的灾难：小模型缺乏超大规模参数所隐含的容错、推断与反思能力，面对复杂的自然语言指令和多分支流程，极易产生参数幻觉、忽略关键校验步骤，甚至陷入上下文认知过载。

<img src="/images/2608.10538v1/x1.webp" alt="SkillsBench 单技能任务的成本与表现权衡" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了解决这种典型的“模型不匹配”（Model-Mismatch）难题，来自哈工大、上海人工智能实验室、上海交通大学以及清华大学的研究团队提出了 **SKILLER**。这是一套完全由自然语言驱动的强化学习框架，其最具颠覆性的设计在于：它不需要更新小模型的任何神经网络权重，而是将“文本形态的技能文档”本身视作可优化的策略（Policy）；由前沿大模型担任 Actor 与 Critic，将被优化的小模型 Agent 闭环视作环境，所有状态转换、诊断奖励和策略更新完全通过自然语言完成。实验证明，经过 SKILLER 专门调优后，仅有 4B 参数规模的 Qwen3.5-4B 在基准测试中的表现，甚至直接反超了配备常规技能或人工编写技能的 9B 模型。

### 为什么大模型的优秀技能，小模型“学不会也用不好”？

在智能体 Harness 系统中，技能的核心价值在于给模型画定“行为边界”。一个经过精心调优的技能文档，不仅告诉模型该用什么工具，还规定了何时读取环境状态、如何自我校验中间产物，以及怎样格式化输出结果。但这种设计长期存在一个默认假设：执行这些技能的基座模型具有强大的上下文理解能力和自我纠错韧性。

当开发者尝试把这种成熟的技能平移给 Qwen3.5-9B、Qwen3.5-4B 乃至更小的模型时，严重的排异反应立刻显现。小模型受限于上下文建模深度与指令遵循容量，很容易被冗长的 SOP 说明书所淹没。面对充斥着各种边缘情况说明和发散性分支的长文本，小模型不仅无法准确捕捉关键步骤，反而在提取工具参数时频繁出现幻觉，或者在遇到轻微环境报错时直接停滞。更严重的是，闭源前沿模型往往拥有极强的隐式推理能力，许多在人类开发者或 GPT-5 级模型看来“理所当然”的操作衔接，在小模型眼里却是完全断裂的执行断层。

目前业界已有的自动化技能生成方案，如 AutoSkill、EvoSkill 以及 SkillX 等，大多是围绕强基座模型的动作空间和表征能力设计的，并不理解轻量化模型的“认知短板”。直接将通用技能压缩或搬运，既不能消除小模型特有的失败模式，也无法为其定制最易执行的轻量约束边界。如何自动为特定规格的小模型量身打造一套专属技能，成为了端侧智能体走向实用的核心瓶颈。

### 角色大反转：用自然语言重构强化学习闭环

SKILLER 的破局思路跳出了传统的模型微调思维。既然调整小模型的权重成本高、破坏通用性且难以持续沉淀模块化知识，那么不如固定小模型本身的参数，直接在外部演化文本技能。SKILLER 将整个流程形式化为一套基于自然语言的强化学习框架。

<img src="/images/2608.10538v1/x2.webp" alt="SKILLER 框架总览与技能演化示意图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在传统的强化学习设定中，策略由神经网络参数表征，策略更新依赖于梯度反向传播。而在 SKILLER 的体系中，待优化的策略实体是一个纯文本技能 $\mathcal{K}$。对于给定的任务实例 $\mathbf{x}$，冻结参数的小模型 $\pi$ 在技能 $\mathcal{K}$ 的约束下执行动作，诱导出特定的有效策略：




{% raw %}$$ \pi_{\mathcal{K}}(a_{t}\mid h_{t},\mathbf{x}) \triangleq \pi(a_{t}\mid h_{t},\mathbf{x},\mathcal{K}) $${% endraw %}



系统在这个交互过程中发生了奇妙的角色反转：

1. **执行环境（Environment）**：这里的环境不再是单一的软件沙盒，而是包含了小模型 $\pi$ 本身及其运行工具、工作区和官方验证器（Verifier）的整个 Agent 循环。当输入技能 $\mathcal{K}_i$ 时，小模型在环境中展开推演，产出交互轨迹 $\tau_i$，并由验证器返回标量奖励 $r_i$ 和诊断结果 $\mathbf{v}_i$。

2. **状态四元组（State）**：控制器将推演过程构造成一个结构化状态 $\mathbf{s}_i = (\mathbf{x}, \tau_i, \tau^{\star}, \mathbf{v}_i)$。其中 $\tau^{\star}$ 为专家参考轨迹。将当前失败轨迹与成功范例直接并置，能够精确暴露小模型的动作序列到底在哪个关键节点首次偏离了正确路径。

3. **语言层面的 Critic**：由前沿强模型（如 GPT-5.4 级模型）担任。Critic 不仅看标量奖励 $r_i$ 是通过还是失败，更深入比对轨迹偏差与验证器报错信息 $\mathbf{v}_i$。它能够清晰分辨出小模型的失败究竟是因为缺乏步骤引导、工具参数误用、输出契约违背，还是非操作性的底层基础设施异常。随后，Critic 输出具体的、局部的自然语言修改建议 $\mathbf{g}_i$。

4. **重放记忆（Replay Memory）**：系统在演化过程中维护一个文本形式的记忆库 $\mathcal{M}_i$，记录失败特征模式、Critic 诊断摘要以及先前被证明有效的编辑历史。这避免了策略在迭代过程中反复犯同样的错误，同时提供了遭遇性能回退时的回滚依据。

5. **语言层面的 Actor**：同样由强模型驱动，Actor 严格遵循 Critic 的修改建议 $\mathbf{g}_i$，针对当前技能 $\mathcal{K}_i$ 进行有边界的局部编辑（Bounded Skill Update），生成更新增量 $\Delta_i$，从而得到下一轮迭代的技能 $\mathcal{K}_{i+1}$。

通过这一闭环，所有的策略评估与策略更新都在结构化自然语言层面流动。这种设计不仅天然具备可解释性，更关键的是，它将昂贵的强模型算力完全收拢在离线构建期，最终交付给线上部署的仅仅是一个轻量、确定性强、专为小模型裁剪的文本技能文件。

### 性能倒挂：4B 小模型如何逆袭 9B 智能体？

为了验证 SKILLER 生成技能的真实有效性，研究团队在涵盖软件工程、综合信息检索以及专业地球科学等领域的五大基准上进行了全面评测，包括 SkillsBench、SWE-Skills-Bench、SkillLearnBench、GAIA 和 EarthBench。执行端小模型选用了开源领域极具代表性的 Qwen3.5-9B 与 Qwen3.5-4B，并与无技能基线、三种开源技能演化方案（AutoSkill、EvoSkill、SkillX）、闭源方案 Manus 以及官方人类专家编写的技能进行了系统对标。

实验结果展现了非常显著的性能飞跃。在 Qwen3.5-9B 上，SKILLER 驱动的 Agent 在各个基准上实现了 4.3 至 20.4 个百分点的绝对胜率提升；在参数减半的 Qwen3.5-4B 上，也获得了 1.8 至 13.3 个百分点的增长。在 SkillsBench 单技能任务中，配备该技能的小模型表现已经开始逼近甚至追平拥有定制技能的顶尖闭源模型。

更为关键的一项发现在于跨尺寸的“性能倒挂”。在极具挑战性的软件工程基准 SWE-Skills-Bench 上，搭载了 SKILLER 专属技能的 Qwen3.5-4B 模型，其测试通过率不仅超越了使用无技能策略的基线，甚至大幅超过了搭载人类编写技能、AutoSkill、EvoSkill、SkillX 乃至闭源 Manus 技能的 Qwen3.5-9B 模型。

这一现象极具启发性。它雄辩地证明，在多步结构化任务中，盲目堆砌模型参数并不一定能带来对应的成功率提升；一个精准契合小模型行为分布、将其容易犯错的行为严格约束住的外部语言策略，其产生的实际工程价值，完全可以弥补数倍的参数容量劣势。

### 为什么人类写的长文档，反而不如自动生成的紧凑技能？

除了最终通过率，研究团队在零样本泛化测试和技能结构特征分析中，揭示了小模型在理解和执行技能时的一些深层规律。

<img src="/images/2608.10538v1/x3.webp" alt="SWE-Skills-Bench 与 SkillLearnBench 上的学习动态收敛曲线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

<img src="/images/2608.10538v1/x4.webp" alt="SWE-Skills-Bench 与 SkillLearnBench 上的学习动态收敛曲线（续）" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

通过观察技能在连续 5 轮迭代中的学习动态可以发现，在不同的任务复杂度下，技能的演化形态呈现出清晰的分化。在流程相对直接、以格式校验为主的 SkillLearnBench 上，小模型的策略在两轮迭代内就迅速收敛，Critic 能够极快地通过错误输出定位到边界契约的缺失；而在高度复杂的软件工程任务 SWE-Skills-Bench 中，技能的表现则呈现出持续递进的上升曲线。早期迭代主要用于清除致命的执行流程偏差，而后期的迭代则不断向技能内注入细粒度的输入落脚点校验（Input Grounding）和自验证逻辑（Self-Validation）。

而在 GAIA 这类需要多跳信息检索的泛化测试中，不同技能生成范式的差异被进一步放大。闭源商业系统 Manus 生成的技能由于偏向人类阅读习惯，包含了大量背景上下文与长篇大论的操作指引。这种技能虽然在专业数据处理任务 EarthBench 上表现尚可，但在复杂的 GAIA 上却导致 4B 小模型性能出现断崖式下跌，甚至落后于完全不给技能的原始 Baseline。原因恰恰在于：过于冗长的上下文在推理过程中不断分散小模型的注意力，导致错误累积被逐级放大。

对比各项技能的结构统计指标，SKILLER 生成的技能展现出了极其独特的“极简主义”特征：

- **自然语言极度精简**：SKILLER 生成文本的词数远少于开源自动化方案，其跨任务的 TF-IDF 相似度与人类手写技能一样保持在极低水平。这表明它没有套用虚浮的模板化套话，而是真正针对具体任务生成了紧凑的硬性约束。

- **外部代码脚本大量增加**：SKILLER 生成的技能中，平均调用的独立脚本数量和实际代码总行数（LOC）均显著高于所有基线。

这一对比揭示了 SKILLER 成功的最核心机制：**将复杂的过程性逻辑从不确定的“自然语言推理”中剥离出来，降维沉淀为确定性的“外部辅助脚本”**。小模型不擅长在思维链中进行多层嵌套运算，但如果技能直接为其提供一段可调用的 Python 脚本或自动化检测程序，小模型只需要精准发起执行并接收返回，整个推理容错空间就被大幅拓宽。

### 重新定义端侧智能体的落地路径

SKILLER 的探索为智能体工程落地提供了一个极具吸引力的全新范式。在以往的落地设想中，企业要么承担高昂的 API 调用账单使用顶尖大模型，要么投入巨额算力去微调小模型，但往往会陷入“微调破坏通用底座能力、且无法应对多变业务流程”的泥潭。

这项研究证明，智能体的推理成本完全可以通过“将重计算前置到技能编译期”来化解。通过在前瞻离线阶段使用最强大脑（如 GPT-5.4 级模型）对轻量级小模型进行充分的“对抗演练”与“自然语言强化学习”，我们能够提炼出一套兼具高压缩度、强约束性与执行确定性的外部技能库。当这套系统部署至端侧消费级显卡时，小模型只需依托极少的上下文开销，就能依靠精准的外挂脚本与 SOP 走出过去大模型才能完成的复杂路径。端侧智能体走向产业规模化应用的关键，或许正是在于这种模型能力与执行策略的精细解耦。
