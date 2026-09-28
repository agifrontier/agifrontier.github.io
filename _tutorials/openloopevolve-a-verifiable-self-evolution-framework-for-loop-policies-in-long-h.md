---
layout: default
title: "清华提出 OpenLoopEvolve：让 Agent 循环策略成为可自主演化的独立资产"
description: "清华大学团队提出的 OpenLoopEvolve （简称 OLE ）试图从底层逻辑上改变这一现状。该研究不再把交互控制视为不可分割的代码黑盒或易受干扰的纯文本上下文，而是首次提出了可治理、可验证的“循环策略”（ Loop Policy ）资产概念。"
arxiv_id: "2608.09380"
paper_published: "2026-08-10"
published_at: "2026-09-28T13:15:07.830421+08:00"
topics:
  - "AI Agent"
tags:
  - "Champion-Challenger evaluation"
  - "LLM"
  - "Loop Policy"
  - "Offline evolution"
  - "Online evolution"
  - "OpenLoopEvolve"
related_tutorials:
  - "learning-on-the-job-an-experience-driven-self-evolving-agent-for-long-horizon-ta"
  - "longhorizon-harness-advancing-long-horizon-agents-for-real-world-tasks"
  - "ahabench-do-agents-learn-from-prior-experience-a-benchmark-for-long-horizon-cont"
  - "a-subgoal-driven-framework-for-improving-long-horizon-llm-agents"
seo_title: "OpenLoopEvolve: A Verifiable Self-Evolution Framework for Loop Policies in Long-Horizon Complex Tasks"
---

<p class="paper-original-title" lang="en">OpenLoopEvolve: A Verifiable Self-Evolution Framework for Loop Policies in Long-Horizon Complex Tasks</p>

在长时间跨度的复杂交互任务中，自主智能体（Agent）所面临的真正考验，往往不是单步推理有多聪明，而是如何应对持续多轮决策带来的不确定性与系统性偏差。无论是在多步骤软件开发、深度研究，还是在复杂的业务经营模拟中，环境状态时刻都在动态演化。Agent 不仅需要反复读取观察、调整规划、调用工具，还必须频繁执行结果验证、异常恢复并严格约束推理预算。

> ArXiv URL：https://arxiv.org/abs/2608.09380v1

然而，在现有的 Agent 架构中，这一套至关重要的“控制经验”始终处于一种尴尬的生存状态：它们要么被硬编码在固定的 Python 框架逻辑中，要么被塞进单一的静态 Prompt 模板里。一旦运行出错，这些交互轨迹往往随着上下文的刷新而烟消云散，无法跨越不同的历史执行过程进行沉淀和迁移。换言之，大模型本身拥有强大的通用能力，但指挥它在长周期任务中如何“循环运转”的控制策略，却依然缺乏系统化的演化与治理手段。

清华大学团队提出的 **OpenLoopEvolve**（简称 **OLE**）试图从底层逻辑上改变这一现状。该研究不再把交互控制视为不可分割的代码黑盒或易受干扰的纯文本上下文，而是首次提出了可治理、可验证的“循环策略”（**Loop Policy**）资产概念。通过引入生产系统常用的“擂主-挑战者”（**Champion–Challenger**）成对评测与鲁棒准入机制，OLE 让 Agent 控制系统能够在在线运行或离线归档中自主迭代，真正实现了控制经验的跨任务复用与自我演化。

<img src="/images/2608.09380v1/openloopevolve_framework.webp" alt="OpenLoopEvolve 整体自演化架构图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 从固定脚手架到可独立治理的 Loop Policy

要想理解 OpenLoopEvolve 的核心贡献，首先需要厘清长周期复杂任务（Long-Horizon Complex Tasks）对控制逻辑提出的严苛要求。在这类任务中，单次目标往往需要几十甚至上百个连续动作步骤才能完成，每一个决策都会改变环境状态并产生新的反馈，后续步骤高度依赖先前的历史轨迹。

过去几年，学术界与工业界主要从两端尝试解决长周期执行问题。一端是增强模型外部记忆与反思，例如将历史会话记录在向量库或结构化文本中供后续检索，但这类方案更新的主要是“陈述性知识”或“语义记忆”，并未改变执行逻辑本身的结构；另一端则是“循环工程”（Loop Engineering），即通过代码显式编排任务状态机、门禁检查与重试逻辑。这类框架虽然提升了稳定性，却将控制规则深嵌在框架代码内，导致优化结构依然高度依赖人工手写规则，无法像权重或提示词那样实现自动化搜索与演进。

OpenLoopEvolve 提出的突破点在于：**将控制 Agent 完整生命周期演进的一整套外部规则抽离出来，抽象为一个独立的、可移植的策略实体——Loop Policy**。

在形式化定义中，一个完整的 Loop Policy 被解耦为八个高度协同的核心控制构件：




{% raw %}$$ \pi = \langle \pi_{\mathrm{obs}}, \pi_{\mathrm{plan}}, \pi_{\mathrm{mem}}, \pi_{\mathrm{act}}, \pi_{\mathrm{ver}}, \pi_{\mathrm{rec}}, \pi_{\mathrm{stop}}, \pi_{\mathrm{bud}}\rangle \in \Pi $${% endraw %}



这八个构件分别对应 Agent 在环境交互中的关键控制切面：

- $\pi_{\mathrm{obs}}$（观察策略）：决定如何从环境中过滤、压缩并提取有效状态信息；

- $\pi_{\mathrm{plan}}$（规划策略）：决定何时触发重新规划、如何拆解子目标；

- $\pi_{\mathrm{mem}}$（记忆策略）：规范交互历史的沉淀机制与上下文窗口之外的存储逻辑；

- $\pi_{\mathrm{act}}$（行动策略）：约束工具调用的组织方式与执行边界；

- $\pi_{\mathrm{ver}}$（验证策略）：定义子任务完成度与环境反馈结果的校验协议；

- $\pi_{\mathrm{rec}}$（恢复策略）：当遭遇环境异常或动作失败时的错误捕获与状态回退机制；

- $\pi_{\mathrm{stop}}$（停止策略）：依据收敛条件或任务目标判断何时安全终止循环；

- $\pi_{\mathrm{bud}}$（预算控制策略）：根据任务进度对调用轮次、Token 消耗及 API 开销进行动态分配。

为了让 Loop Policy 能够独立于具体框架流通，研究团队进一步提出了**资产包（Bundle）**的概念，将其形式化表示为：




{% raw %}$$ \mathcal{B}_v = \langle\pi_v, \chi_v, \ell_v\rangle $${% endraw %}



其中，$\pi_v$ 是当前版本的策略规则实体；$\chi_v$ 记录了该策略的适用约束、依赖环境与兼容上下文；$\ell_v$ 则记录了该策略的版本谱系（Lineage），包括它的父版本哈希、演化依据与变更历史。通过这一层封装，Agent 的交互控制机制彻底摆脱了零散的 Prompt 碎片或静态代码脚本，升级为拥有完整版本生命周期的规范化软件工程资产。

### 证据契约与 Champion-Challenger 演化引擎

拥有了明确的策略对象后，自演化框架的核心挑战便转向了演化证据从何而来，以及如何确保新生成的策略不会发生性能退化。大模型自主优化代码或策略时，最容易出现的风险是“灾难性退化”或在局部测试用例上的“伪优化”。OpenLoopEvolve 为此构建了一套闭环的演化协议。

首先是**轨迹与证据契约（Evidence Contract）**。在任务执行过程中，每一次交互都会生成包含任务目标、执行策略版本、多轮交互历史及最终产出的运行轨迹 $\tau_i = (g_i, \pi_{v(i)}, \mathcal{H}_i, y_i)$。当历史轨迹累积到一定规模时，系统的证据提取算子 $\Phi$ 会将轨迹转化为结构化的演化证据：




{% raw %}$$ \mathcal{E} = \Phi(\mathcal{D}) = \{\varepsilon_j = (\rho_j, \delta_j, u_j, \mathcal{I}_j)\}_{j=1}^J $${% endraw %}



这些证据不是松散的文本日志，而是严格指向特定策略构件的归因证据链：$\rho_j$ 标明诱发问题的上下文状态；$\delta_j$ 指向具体的决策偏差或失败原因；$u_j$ 则是可量化的效用差异；$\mathcal{I}_j$ 则对应到具体的控制切面（例如验证不严或恢复机制失效）。

在这套证据体系的驱动下，演化引擎启动了基于**擂主-挑战者（Champion–Challenger）**的双轨评测流：

1. **自主候选提案**：大语言模型作为元优化器，读取当前的 Champion 策略 Bundle、新提炼的演化证据 $\mathcal{E}$ 以及历史评估记录，自主生成候选策略（Challenger）。

2. **同条件成对测试（PairEval）**：系统不会盲目相信大模型给出的策略说明，而是让挑战者策略与当前的擂主策略在严格一致的任务环境、随机种子和初态设置下进行成对运行，计算相对增益 $\Delta_{i,j}$ 与相对收益率 $r_{i,j}$。

3. **鲁棒准入机制（Robust Gate）**：候选策略要取代擂主，必须穿透由复合准则构成的刚性门禁：




{% raw %}$$ \operatorname{Gate}(\mathcal{P}_{k,j},\Theta) = \mathrm{accept} \Longleftrightarrow \prod_{\alpha\in\mathcal{J}_{\mathrm{gate}}}\Gamma_{\alpha}(\mathcal{P}_{k,j},\Theta_{\alpha}) = 1 $${% endraw %}



门禁条件涵盖了任务综合表现、动作成功率下限以及验证风险指标。只有在所有约束项上均达到或超过阈值，挑战者才能被加盖版本印记，正式发布为下一任 Champion。

值得一提的是，OpenLoopEvolve 在实际落地时提供了两种互补的执行模式。**在线演化（Online Mode）**持续监听 Agent 的实时交互，在触发特定条件时在后台生成新策略，并在后续的任务执行边界（Task Boundary）无缝切入。若线上监测到性能回撤，系统可立即利用血缘记录将其回滚至父版本。**离线演化（Offline Mode）**则脱离执行主链路，利用归档的历史运行痕迹和失败案例库，执行多候选、多代际的“生成-评估-选择”搜索演进，最终将淬炼后的稳定版本推送至生产环境。

### YC-Bench 极限压力测试：长达一年的动态商业模拟

为了验证控制策略自演化的有效性，研究团队选择了具有极高难度的长周期基准 **YC-Bench** 进行实验。YC-Bench 要求 Agent 扮演初创公司创始人的角色，在高度动态的仿真市场环境中持续运营一年时间。

这个基准的复杂之处在于，Agent 必须处理任务选择、团队招聘、产品研发、资金预算和市场推广等长链条决策。单步的轻微误判可能会在几个月后引发不可逆的现金流断裂，最终导致破产。评估长周期任务能力的最终指标，则是经营一年后的留存资金、存活周期以及在全流程中面临的违约风险指标。

实验在统一的基座模型 `deepseek-v4-flash`、相同种子与 20 轮上下文窗口的标准中展开，对四种典型配置进行了严苛对比：

- **Baseline (Native)**：使用 YC-Bench 原生执行循环，未引入外部策略资产与演化机制；

- **Fixed-$\pi_0$**：加载与 OLE 相同的初始循环策略，但在整整一年的模拟经营中保持完全静态、不做任何调整；

- **OLE-offline**：预先基于历史轨迹完成离线多轮搜索迭代，并在全年运行中执行固化后的优质策略；

- **OLE-online**：从相同的初始策略出发，在全年的执行过程中动态触发了 12 次候选更新与评估尝试，在任务边界持续吸纳最新反馈。

<img src="/images/2608.09380v1/fig_ycbench_annual_progress.webp" alt="YC-Bench 全年运营演化进展对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从上图所示的全年动态表现可以清晰地观察到，缺乏自适应机制的 Native 基准和固定的 $\pi_0$ 策略在运行到中后期时，其经营资本积累与任务成功率均呈现出显著的滞后甚至下滑趋势。这是因为固定控制逻辑难以应对不同阶段不断演变的环境复杂度。

相比之下，引入了自演化能力的两种 OLE 模式均展现出了显著的抗风险能力与长期收益优势：

首先，静态加载统一控制策略的 **Fixed-$\pi_0$** 虽然规范了单步行为，但在长期经营中无法灵活应对风险变化。而通过离线或在线自演化后，**OLE-offline 与 OLE-online 相较于初始策略 Fixed-$\pi_0$ 分别获得了巨大的综合性能改善**。在长周期任务的终态表现上，自演化机制直接阻断了因策略僵化而引发的雪崩式破产。

其次，两种演化模式展现出了不同的互补特性：**OLE-offline** 经过多代全局候选搜索，策略结构更为稳健，在降低验证风险和提升平均存活周期方面表现出色；而 **OLE-online** 凭借对最新失败案例的敏锐捕获，能够在当前环境中快速产生针对性的恢复策略，在处理突发业务扰动时展现出极高的纠偏灵活性。

此外，研究对推理资源与演化成本进行了全口径统计。实验结果表明，尽管 OLE 在成对评测阶段产生了一定量的候选验证 Token 消耗（在线验证消耗约 29.82M Tokens，离线验证消耗约 24.02M Tokens），但就单次交互调用的平均 Token 消耗以及任务存活带来的整体收益而言，系统的推理效率得到了有效约束，并未出现失控的上下文膨胀。

### 为什么说“策略资产化”是 Agent 进阶的必由之路

回顾近两年大模型 Agent 的发展轨迹，学术界与产业界的注意力大多集中在基础模型的能力跃迁（如长文本上下文扩展、强化学习后训练），或是提示词工程层面的精雕细琢。然而，当开发者试图将 Agent 部署至跨度数天甚至数月的复杂工业级场景时，往往会遭遇一道无形的阻碍：Prompt 调优触及天花板，而硬编码的 Workflow 代码又过于脆弱。

OpenLoopEvolve 带来的关键启示在于，**智能体外部的交互循环（Loop）不应该是静态的脚手架，而应当被视为一种可观测、可追踪、可演进的独立系统资产**。

通过将复杂的长周期控制切分成明确的八元策略结构，OLE 使得大模型“自我优化”的对象不再是难以捉摸的全局上下文，而是具备明确边界的功能算子。而引入生产级别的成对评测与鲁棒准入机制，则在根本上为 Agent 的自主进化扣上了安全绳，避免了在复杂长链任务中由于一次幻觉或盲目自信导致的系统崩溃。

可以预见，随着 Agent 走向复杂流程自动化与企业级业务托管，这种将控制逻辑沉淀为可治理资产的范式，将成为下一代自主系统构建的重要基石。无论是在代码自动重构、自主科学研究还是高频商业仿真中，学会自我沉淀并演进“交互策略”的智能体，才能真正跑赢长周期任务的时间马拉松。
