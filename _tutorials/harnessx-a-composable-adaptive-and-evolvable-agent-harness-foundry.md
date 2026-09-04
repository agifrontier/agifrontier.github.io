---
layout: default
title: "告别手搓Agent！HarnessX实现大模型与运行时外挂协同进化，性能最高飙升44%"
description: "目前开发AIAgent，最痛苦的环节莫过于“手搓”外挂代码。换个新模型，提示词模板可能得全部重写。加个新工具，原本顺畅的控制流直接崩溃。大量在运行中产生的报错记录和轨迹，最后只能被当成垃圾数据扔掉。究竟有没有一种方法，能让Agent的运行框架自己学习、自己修改代码？"
arxiv_id: "2606.14249"
topics:
  - "AI Agent"
  - "推理"
tags:
  - "AEGIS"
  - "HarnessX"
  - "harness-model loop"
  - "runtime harness composition"
  - "substitution algebra"
  - "trace-driven multi-agent evolution"
related_tutorials:
  - "training-task-reasoning-llm-agents-for-multi-turn-task-planning-via-single-turn-"
  - "agentgym-rl-training-llm-agents-for-long-horizon-decision-making-through-multi-t"
  - "agentic-harness-engineering-observability-driven-automatic-evolution-of-coding-agent-harnesses"
  - "tthe-test-time-harness-evolution"
---

<p class="paper-original-title" lang="en" style="font-size:1rem; line-height:1.5; color:var(--global-text-color-light, #6c757d); margin:-0.5rem 0 1.5rem;">HarnessX: A Composable, Adaptive, and Evolvable Agent Harness Foundry</p>

目前开发AI Agent，最痛苦的环节莫过于“手搓”外挂代码。
换个新模型，提示词模板可能得全部重写。
加个新工具，原本顺畅的控制流直接崩溃。
大量在运行中产生的报错记录和轨迹，最后只能被当成垃圾数据扔掉。
究竟有没有一种方法，能让Agent的运行框架自己学习、自己修改代码？

> **ArXiv URL**：http://arxiv.org/abs/2606.14249v1

本文将深度解读一项能彻底改变这一现状的重磅研究。
这套名为HarnessX的新系统，让Agent运行时框架具备了“自我进化”的能力。
在五大主流基准测试中，它实现了平均提升14.5%，最高暴涨44.0%的惊人成绩。

如果把基础大模型比作赛车的**“发动机”**。
那么提示词、工具调用、记忆管理等运行时框架，就是**“底盘与传动系统”**。
过去，开发者只能依靠人工经验，为每款发动机死板地定制底盘。
而HarnessX不仅实现了底盘组件的完全模块化，还能根据试车数据自动改装。
更硬核的是，它甚至能让发动机和底盘在赛道上进行“同步调优”。

### 框架组合：打造可灵活拼装的标准化底盘

要让系统能够自动修改自身代码，首先得让代码具备绝对的“可插拔”特性。
该研究提出了一种基于类型替换代数的基础架构。
通过这种设计，整个运行时框架不再是一团相互耦合的“面条代码”。

<img src="/images/2606.14249v1/x1.webp" alt="[Uncaptioned image]" style="width:85%; max-width:600px; margin:auto; display:block;">

研究团队将Agent的行为空间，极其严谨地划分为了九个独立维度。
这包括了模型选择、上下文组装、记忆管理、工具生态等。
所有的行为都被封装成了独立的处理单元（Processor）。
只要输入输出的类型契合，任何组件都能被安全地插入、替换或拔除。
这种类型的安全性保证，是后续实现自动化代码改造的基石。

### 框架自适应：AEGIS进站维修团队

底盘组件实现标准化之后，系统究竟是如何依靠自身自动完成“改装”的？
答案在于其核心引擎：AEGIS。
它巧妙地将**强化学习**（**RL**）的经典概念，映射到了符号化的代码修改空间。

在AEGIS的视野里，当前的框架配置就是“状态”，而代码的修改操作就是“动作”。
但这引出了一个棘手的问题。
经典的RL“病态”现象一旦进入符号空间，破坏力将被急剧放大。
为了应对这些挑战，AEGIS精心设计了四个环环相扣的流水线阶段：

<img src="/images/2606.14249v1/x2.webp" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">

1. **消化器**（**Digester**）：负责处理海量日志。
运行一次测试可能会产生上千万个Token的原始轨迹。
它负责将其压缩提炼，精准提取出发生故障的底层组件和证据。

2. **规划器**（**Planner**）：负责对抗**探索不足**（**Under-exploration**）。
如果系统只知道盯着眼前的错误，很容易陷入“只改提示词”的局部微调陷阱。
规划器会构建全局的调整策略，强制系统尝试引入新工具或重构记忆模块等结构性改变。

3. **进化器**（**Evolver**）：负责生成实际的代码候选方案。
它会产出带有明确“变更清单”的代码修改指令。
这就好比技师在更换零件前，必须清晰写下修改动机和预期效果。

4. **批评家**（**Critic**）与门控机制：负责对抗**灾难性遗忘**（**Catastrophic forgetting**）。
在参数优化中，修复故障A经常会导致正常的流程B崩溃。
AEGIS引入了严格的“跷跷板约束”。
无论批评家的评估如何，系统必须通过确定性的回归测试。
绝对不允许任何新的修改，破坏以前已经能够完美跑通的测试用例。

### 模型与框架协同进化：发动机与底盘的极致共鸣

当底盘被优化到了极致，受限于小尺寸发动机本身的马力，成绩依然会遇到天花板。
无论外部工具给得多么完善，模型本身的推理能力才是最终的瓶颈。
此时，HarnessX启动了最高阶的玩法：模型与框架的协同进化。

<img src="/images/2606.14249v1/x3.webp" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">

研究团队引入了**组相对策略优化**（**GRPO**）算法，并进行了一次极其优美的工程架构设计。
系统的精妙之处在于，为模型和框架构建了一个共享的经验回放缓冲区（Replay Buffer）。
这意味着，同一次测试跑出来的轨迹 $\tau$，既是框架演进的诊断证据，也是模型训练的强化学习信号。

在这个缓冲区中，跨越了不同框架版本的运行轨迹被混合在了一起。
同一个任务在不同底盘版本下的表现，被统一打包成一个组 $\mathcal{G}_{x}$。
模型需要计算每一条轨迹的优势函数 $\hat{A}(\tau_{i})$。




{% raw %}$$ \hat{A}(\tau_{i})=\frac{r_{i}-\mu(\mathcal{G}_{x})}{\sigma(\mathcal{G}_{x})+\epsilon} $${% endraw %}



这种跨越框架版本的任务级对齐，产生了一种奇妙的化学反应。
它不要求动作空间严格一致。
即使底盘版本发生了翻天覆地的变化，模型依然能从“不同策略的对比”中吸收经验。
框架负责搭建粗粒度的宏观战略，而模型则负责学习如何在复杂的上下文中精准执行。

### 实验验证与核心启示

HarnessX在ALFWorld、GAIA、WebShop等五个极具挑战性的基准测试中，证明了其惊人的威力。
通过最高15轮的自主进化，在15个不同的“模型-基准测试”组合中，平均取得了14.5%的绝对提升。

这里面呈现出了一个非常反直觉的“逆缩放”规律。
越是推理能力偏弱的小模型，从框架进化中获得的收益反而越大（例如Qwen3.5-9B在ALFWorld上暴涨了44.0%）。
这强有力地证明了，在当前算力昂贵的背景下，优秀的底盘设计能极大弥补发动机算力的不足。
此外，引入协同进化机制后，系统在仅靠框架进化达到的极限之上，又榨取出了额外4.7%的性能提升。

当然，该方案在实践中也存在一定的门槛。
驱动四个阶段运转的Meta-Agent需要消耗大量的Token，这在早期探索阶段会带来不小的API成本。
同时，要求开发者按照九大维度完全重构现有的业务代码，短期内存在较高的工程阵痛。

但这篇论文带来的核心启示是深远的。
它告诉所有的AI开发者：不要再把Agent的外部框架仅仅当成一次性的“胶水代码”。
将运行时框架提升为一等公民，使其具备可序列化、可替换、可演算的特性。
这是Agent技术从“手工作坊”走向“工业化自动进化”的必经之路。
