---
layout: default
title: "openJiuwen：华为双层自适应架构，长程代码Agent双榜超榜首3.4分"
description: "大语言模型在软件工程领域的应用，正在经历一场从“单轮代码片段生成”向“长程（Long-Horizon）自主智能体”的范式转移。面对真实代码仓库中盘根错节的依赖、长达数十甚至上百步的探索轨迹，智能体脚手架（AgentHarness）的角色已经从最初薄薄的一层API封装，蜕变成为决定整个系统上限的核心底座。"
arxiv_id: "2608.27969"
published_at: "2026-09-04T10:33:26.870465+08:00"
topics:
  - "AI Agent"
tags:
  - "Delegated sub-agents"
  - "Long-horizon coding agents"
  - "Rail-based capability composition"
  - "Runtime Adaptivity"
  - "SWE-bench Verified"
  - "Shared execution substrate"
related_tutorials:
  - "a-subgoal-driven-framework-for-improving-long-horizon-llm-agents"
  - "agentgym-rl-training-llm-agents-for-long-horizon-decision-making-through-multi-t"
  - "harnessing-uncertainty-entropy-modulated-policy-gradients-for-long-horizon-llm-a"
  - "agentfold-long-horizon-web-agents-with-proactive-context-management"
---

<p class="paper-original-title" lang="en">openJiuwen: Beyond Static Harnesses for Long-Horizon Coding Agents</p>

<img src="/images/2608.27969v1/A__title.jpg" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

大语言模型在软件工程领域的应用，正在经历一场从“单轮代码片段生成”向“长程（Long-Horizon）自主智能体”的范式转移。面对真实代码仓库中盘根错节的依赖、长达数十甚至上百步的探索轨迹，**智能体脚手架**（Agent Harness）的角色已经从最初薄薄的一层 API 封装，蜕变成为决定整个系统上限的核心底座。然而，目前多数开源方案仍停留在相对静态的编排逻辑中：要么难以灵活组合异构能力与复杂拓扑，要么无法随着运行期新证据的涌现而动态调整控制策略。

> ArXiv URL：https://arxiv.org/abs/2608.27969v1

华为开源的 **openJiuwen** 框架正是为了破除这一瓶颈而生。它从系统工程视角重新审视长程代码智能体，提出了以**结构可组合性**（Structural Composability）和**运行时自适应**（Runtime Adaptivity）为核心的全新架构。该框架在保持底层大模型策略固定的前提下，通过双层执行循环与能力轨道（Rail）机制统一了从单 Agent 到 Swarm 群体协同的语义，同时利用渐进式上下文管理、目标模式、LSP 语义被动反馈以及跨任务自省，让运行时状态根据代码演化证据自适应转向。在严格的软件工程基准 SWE-bench Verified 与真实终端基准 Terminal-Bench 2.1 上，openJiuwen 分别取得了 82.6% 与 87.19% 的高分，双双超越对应官方榜单的最优成绩达 3.4 与 3.39 个百分点。

### 长程代码智能体的系统性困境

当开发者尝试构建一个能自主定位 Bug、修改仓库并跑通测试套件的系统时，通常会遇到两类相互交织的挑战。

第一类挑战在于**结构维度的组合成本**。当前的脚手架架构呈现出高度分化的特点：Pi 倾向于极简运行时并通过扩展实现高级特性；DeepSeek Harness 走向插件级别的全面可替换性；DeerFlow 专攻基于主从节点的层级式委托；而 Claude Code 与 OpenAI Codex 则在产品级工程中深度集成了权限控制、上下文裁剪与技能钩子。然而，现存方案往往将特定的协作拓扑与专属的执行引擎死锁在一起。当开发者希望将一个单 Agent 系统重构为带有子智能体委派（Delegated Sub-agents）或群体流（Swarm Flow）的复杂系统时，通常被迫重写底层的调度循环与工具调用逻辑。这种架构上的碎片化，使得能力拓展的维护成本极其高昂。

第二类挑战在于**运行维度的执行不确定性**。长程编码任务本质上是一个信息不对称、环境动态演化的探索过程。任务启动之初编写的固定 Prompt 或预设路径，会随着代码修改、测试报错、静态类型诊断和资源压力的变化而迅速失效。大量实证研究表明，长程代码 Agent 最终失败的轨迹往往伴随着冗长的无效重试、死循环调用以及上下文溢出。如果框架不能动态捕获运行期涌现出的新证据，并在不重新微调模型参数的情况下主动重构可见上下文、纠偏语义错误或干预终止时机，Agent 就很容易在长轨迹中迷失。

openJiuwen 的定位并非再造一个局部的辅助工具，而是将 Harness 本身确立为一级系统抽象层，通过解耦“结构如何组装”与“轨迹如何自适应演进”，彻底破解上述双重难题。

### 结构可组合性：双层循环与 Rail 机制

为了让任意复杂度的智能体拓扑都能运行在同一套底层语义之上，openJiuwen 抽象出了两个核心基础原语：分层的双循环执行引擎（Inner Loop / Outer Loop）与基于 Rail 的能力组合机制。

传统的 ReAct 循环通常把单步工具调用与整个任务的生命周期混为一谈，导致异常捕获、重试机制与长程决策逻辑严重耦合。openJiuwen 将其严格解耦为两层控制：

- **内循环（Inner Loop）**：负责严格受限的局部模型–工具交互。在每一个时间步，框架负责构建当前模型可见的上下文，调用固定的策略模型 $\pi$，执行其请求的工具调用并记录观测结果。内循环提供围绕模型调用和工具调用的生命周期回调，使得外部插件无需侵入循环内部即可实施观察或修改。

- **外循环（Outer Loop）**：将一次完整的内循环执行定义为一个回合（Round）。外循环不关心局部的单步推理细节，而是站在任务宏观维度，评估当前是应该彻底终止、切换分支、注入恢复逻辑，还是开启下一个回合的内循环。在模型交互过程中异步到达的新输入，也会在外循环的轮次边界处有序合入，避免直接打断进行中的推理。

在双层循环之上，跨切面能力（如安全校验、记忆持久化、规划管理、上下文裁剪）的接入通过 **Rail**（能力轨道）机制实现。一个 Rail 实例在数学上被定义为三元组 $\rho = (\mathcal{H}_\rho, f_\rho, p_\rho)$，其中 $\mathcal{H}_\rho$ 指定其挂载的生命周期钩子集合（涵盖模型调用阶段 $\mathcal{H}_{\mathrm{invoke}}$、工具调用阶段 $\mathcal{H}_{\mathrm{call}}$ 和宏观任务阶段 $\mathcal{H}_{\mathrm{task}}$），$f_\rho$ 为具体处理逻辑，而 $p_\rho$ 则定义了执行优先级。

```

H = H_invoke ∪ H_call ∪ H_task

ρ = (H_ρ, f_ρ, p_ρ)

```

更关键的设计在于**可见性门控**（Visibility Gating）。引入指示函数 $g(\rho, u) \in \{0, 1\}$ 表示特定执行主体 $u$ 是否能够访问某个 Rail $\rho$。因此，特定主体可用的能力集可以精确表示为：




{% raw %}$$ \mathcal{R}(u) = \{\rho \in \mathcal{R} \mid g(\rho, u) = 1\} $${% endraw %}



这一机制同样对称地适用于工具集。借此，主智能体、受限子智能体、审查者或群体节点可以完全复用同一个底层执行引擎，仅通过配置不同的可见性掩码，即可天然实现能力隔离、渐进式权限暴露和有界的递归委派，消除了为不同角色编写定制引擎的繁琐负担。

在此基础之上，openJiuwen 进一步推出了 **Swarm Flow**，将常见的多智能体协作模式抽象为可自由插拔、串联或并联的流式算子。如图所示，从需求拆解、方案规划、编码执行到独立验证，开发者能够通过模块化组合快速落地生产级的复杂流水线。

### 运行时自适应：固定模型之下的动态导航

如果说结构可组合性定义了 Agent 系统的骨架，那么运行时自适应则赋予了它在复杂多变的执行环境中实时“避障与纠偏”的动态调节能力。openJiuwen 将单任务运行期的动态控制形式化为一个在固定模型策略 $\pi$ 下的受约束在线适应过程。

在运行控制节点 $t$，框架所掌握的全部信息集合记为 $\mathcal{M}_t$（包含交互历史、中间诊断、目标达成状态及资源消耗）。框架通过自适应配置组 $\Theta_t = (\kappa_t, \Phi_t, \iota_t)$ 来驱动轨迹走向，其中包含上下文构建组件 $\kappa_t$、接受与终止判定组件 $\Phi_t$ 以及诊断反馈注入组件 $\iota_t$。模型在每个时刻仅针对经过动态加工后的上下文 $c_t = \kappa_t(\mathcal{M}_t)$ 进行动作采样 $a_t \sim \pi(\cdot \mid c_t)$。整个轨迹追求长期综合收益最大化，同时严格受限于上下文窗口预算约束 $|c_t| \le B_{\mathrm{ctx}}$：




{% raw %}$$ J(\Theta_{0:T}) = \mathbb{E}\left[\sum_{t=1}^{T} (r_t - \lambda \operatorname{cost}_t)\right], \quad \lambda \ge 0 $${% endraw %}



这一抽象在工程上被具体实例化为四项相辅相成的自适应支柱：

首先是**自适应上下文管理**（Context Management）。面对动辄消耗数十万 Token 的长程编码轨迹，静态的历史拼接必然引发灾难性的截断或上下文迷失。openJiuwen 引入了**渐进式压缩**（Progressive Compression）机制，使任意上下文单元 $m$ 经历由粗到细的多级表征演变 $m^{(0)} \rightarrow m^{(1)} \rightarrow \cdots \rightarrow m^{(L)}$，最新产生的关键交互保留完整精度，早期历史则被提炼为语义摘要。针对代码差异（Diff）、环境日志等强结构化数据，框架先使用确定性的**结构感知处理器**压缩，并在检测到重复低效的调用时触发**死循环折叠**（Dead-loop Collapse）。更进一步，体量巨大的非活跃实体被完全卸载出活动窗口，仅保留轻量句柄（Handle）以供按需召回，并与推理服务层的 KV 缓存亲和性调度相配合，大幅削减吞吐开销。

其次是**目标模式**（Goal Mode）。在传统长程交互中，智能体容易因一次微小的报错而误判任务失败退出，或是陷入盲目重试。Goal Mode 在外循环中显式维护高阶目标状态机 $\Phi_t(\mathcal{M}_t) \in \{\mathsf{continue}, \mathsf{complete}, \mathsf{blocked}\}$。只有当任务经由验证判定完成、明确受阻，或者触碰硬性资源上限时，宏观轨迹才会终止，有效保障了长程任务的执行耐力：




{% raw %}$$ T = \inf\bigl\{t \colon \Phi_t(\mathcal{M}_t) \neq \mathsf{continue} \lor g_{\mathrm{cap}}(\mathcal{M}_t) = 1\bigr\} $${% endraw %}



第三项设计是 **LSP 驱动的主动语义反馈**（LSP-Driven Passive Feedback）。这是该框架在代码工程领域极具实操价值的创新点。智能体修改代码后，往往难以立刻感知由此引发的类型破坏或跨文件符号引用错误。openJiuwen 在底层挂载了语言服务协议（Language Server Protocol），在代码发生变动后立即捕获机械推导出的精确诊断信息 $\Delta(s_t)$，经过排序、去重与数量截断后生成增量反馈 $\delta_t$，主动塞入下一轮状态 $\mathcal{M}_{t+1}^+$ 中。这种闭环机制让模型在尚未触发整体编译或单元测试之前，就能在单步中迅速修正低级符号与语法疏漏。

最后，系统通过**自省机制**（Self-Reflection）实现跨任务非参数化自适应。当一条轨迹 $\tau_n$ 结束后，自省模块从中提炼出结构化的成败经验 $X_n = \Psi(h^{\tau_n})$ 并更新至经验库 $\mathcal{E}_{n+1}$。在开启新任务 $q_{n+1}$ 时，相关经验被动态检出并注入初始上下文 $\mathcal{M}_0$。这意味着系统能在不更新任何神经网络权重的条件下，实现“越用越聪明”的持续演进。

### 基准实测：复杂长程环境下的鲁棒性突破

为了验证系统在真实高难度任务下的有效性，openJiuwen 在两大业界公认的高壁垒基准上进行了严谨的对照实验。

在针对真实 GitHub 仓库缺陷修复的 **SWE-bench Verified**（500 道经过人工校验的真实软件工程问题）评估中，搭载 Claude Opus 4.5（高思考预算配置）的 openJiuwen 取得了 **82.6%** 的 Pass@1 解决率。与目前公开可见的权威榜单记录对比，该表现高出此前后端领先点估计达 3.4 个百分点。更深入的时间步分析揭示了一个关键趋势：随着任务解决所需轨迹步数的显著拉长，传统静态脚手架的成功率通常呈断崖式下滑，而 openJiuwen 凭借自适应上下文压缩与 Goal Mode 的韧性，在中长轨迹区间依然保持了稳健的命中率。

而在更加贴近全功能开发环境、不仅限于代码修复的 **Terminal-Bench 2.1**（89 个涉及复杂容器环境交互、长程系统运维与工具调用的终端任务）上，openJiuwen 展现出了更为显著的优势。如测试数据所示，在主配置下使用 GPT-5.6 Sol 作为基座时，openJiuwen 创下了 **87.19%** 的准确率纪录，超越此前官方榜单上由 Claude Code（配备 Fable 5 基座）保持的 83.8% 成绩，幅度达到 3.39 个百分点。

为了排除基座大模型能力差异带来的干扰，研究团队特别进行了一组**严格对齐模型基座**的消融评测：当三者均统一切换至 Fable 5 模型时，openJiuwen 依然交出了 84.04% 的答卷，在同等参数基座下稳稳超越 Claude Code（83.8%）与 Terminus 2（80.4%）。

深入分析 Terminal-Bench 2.1 的 16 个细分任务类别可以发现，openJiuwen 的胜势主要建立在强工具依赖和深度环境交互的领域。在同样搭载 Fable 5 的严苛对比下：

- 在**文件操作**（File Operations）类目中，openJiuwen 斩获了 0.76 的得分，远超 Claude Code 的 0.56 和 Terminus 2 的 0.52；

- 在**系统运维**（System Administration）类目中，openJiuwen 达到了 0.889，高于另外两者的 0.778 与 0.844；而在 GPT-5.6 Sol 的全功能配置下，这一指标进一步攀升至 0.956，并在软件工程（0.908）与数据科学（0.950）任务中保持极高胜率。

这些细分领域的显著提升，直接印证了 LSP 闭环反馈、确定性文件工具集以及状态感知机制在应对真实终端不确定性时的巨大威力。

### 总结与展望

openJiuwen 的提出为火热但略显混乱的 Coding Agent 领域注入了一剂系统工程的理性思考。它有力地证明了：在模型基础推理能力相对给定的阶段，**Agent 脚手架绝非简单的提示词包裹器或胶水代码，而是一个具备严格生命周期抽象、能力组合范式与闭环自适应调控能力的完整系统层**。

通过将单智能体到多智能体拓扑统合于“双层循环 + Rail 门控”的基础设施之上，openJiuwen 赋予了开发者极高的架构自由度；而将运行时控制形式化为带有约束的在线状态自适应，则有效攻克了长程轨迹中信息迷失与错误累积的顽疾。伴随代码的开源，这种兼顾开发者可组合性与运行时自适应性的系统范式，有望为下一代能够真正接管复杂工程任务的自主软件工程智能体确立坚实的基础设施标准。
