---
layout: default
title: "LivePlan：解耦“裁决”与“建议”，仅增0.08美元将Agent解题率提升15.2%"
description: "由 IBM 与伊利诺伊大学厄巴纳-香槟分校（UIUC）联合团队提出的 LivePlan 框架，换了一种更务实的解题思路： 将运行时的“漂移裁决”与“纠偏建议”彻底解耦 。"
arxiv_id: "2608.06701"
paper_published: "2026-08-07"
published_at: "2026-09-06T13:15:07.886648+08:00"
topics:
  - "AI Agent"
tags:
  - "LivePlan"
  - "SWE-agent"
  - "SWE-bench Pro"
  - "SWE-bench Verified"
  - "advisor LLM"
  - "corrective steering"
related_tutorials:
  - "swe-bench-can-language-models-resolve-real-world-github-issues"
  - "prune4web-dom-tree-pruning-programming-for-web-agent"
  - "socratic-swe-self-evolving-coding-agents-via-trace-derived-agent-skills"
  - "harnessbridge-learnable-bidirectional-controller-for-llm-agent-harness"
---

<p class="paper-original-title" lang="en">Online Monitoring and Corrective Steering of Programming Agents</p>

在真实的软件工程场景中，让大语言模型（LLM）去修复大型开源项目的 GitHub Issue 是一项典型的长程任务（Long-horizon Task）。一个有效的补丁往往需要跨越多个文件、涉及定位、复现、修改与验证等多轮复杂动作。当下的软件工程 Agent（如 SWE-agent）通常遵循一套固定的规划流程，但在长上下文交互中，模型极易出现“行为漂移”（Behavioral Drift）：陷入死循环、在同一个错误工具调用上反复挣扎、或者在验证环节盲目乱转直至耗尽 Token 额度。

> ArXiv URL：https://arxiv.org/abs/2608.06701v1

针对 Agent 运行时的失控问题，学术界以往的干预手段大多代价高昂且脆弱。有的方案在执行失败后让 LLM 完整复盘并重新规划整个长程轨迹（如 SAGE），不仅成本翻倍，而且模型生成的全局新方案经常包含幻觉；有的方案则每隔几步就调用一次大模型进行状态审查（如 SWE-PRM），带来巨大的 Token 消耗与推理延迟。由 IBM 与伊利诺伊大学厄巴纳-香槟分校（UIUC）联合团队提出的 LivePlan 框架，换了一种更务实的解题思路：**将运行时的“漂移裁决”与“纠偏建议”彻底解耦**。

LivePlan 引入了确定性的轻量级规则监视器，在无需调用任何 LLM 的情况下，以毫秒级的极低开销捕捉 Agent 轨迹中的循环震荡、过度停滞与违规跳步；只有在检测到异常时，才会激活 Advisor LLM，基于受限的局部上下文给出一步轻量的下一步纠偏建议。在 SWE-bench Verified 与代表真实工业复杂度的 SWE-bench Pro 测试中，LivePlan 为不同基座模型带来了最高 15.2%（平均 9.9%）的解决率提升，而每个任务实例的额外成本仅约 0.08 美元，新增的成功用例高度集中在最具挑战性的中高难度任务上。

<img src="/images/2608.06701v1/overview.webp" alt="LivePlan 架构与工作流全览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么长程编程 Agent 总是容易“跑偏”？

要理解 LivePlan 的必要性，首先要看清编程 Agent 在真实长轨迹中究竟是如何崩溃的。

一个标准的软件修复流程通常包含定位（Localize）、复现（Reproduce）、打补丁（Patch）和验证（Validate）四个关键阶段。然而，受限于上下文窗口膨胀以及模型自身在长链条推理中的注意力衰减，执行体（Executor）常常在未完成当前阶段时发生漂移。团队对数千条运行轨迹进行抽象建模后发现，Agent 的失控并非完全不可预测，而是具有高度共性的病态模式：

一种典型情况是**死循环与动作震荡（Action Oscillation）**。当 Agent 遇到环境报错或文件未找到时，容易陷入“尝试相同命令 - 报错 - 再次尝试微调参数 - 再次报错”的无限循环；另一种情况是**长程停滞（Long Stagnation）**，例如模型在没有写出最小复现脚本的情况下，花费数十步调用文件搜索命令，在庞大的仓库目录里盲目漫游；还有一种常见模式是**跳步与计划违背（Plan Violation）**，特别是在较弱的模型上，Agent 往往在草率修改完代码后直接跳过验证阶段提交，误以为问题已经解决。

以往的研究试图利用另一个更强的模型担任“监工”，但这往往带来两难处境。如果频繁进行定期评估（Periodic Monitoring），会急剧推高 API 调用成本和排队延迟；更严重的问题在于，让 LLM 进行全局性重新规划极易引发“二次幻觉”——评审模型往往在缺乏完整代码上下文的情况下强行给出具体的行号或代码片段，执行模型由于能力较弱，往往盲目服从这些错误的幻觉指令，直接导致原本有望修复的任务彻底报废。

### 核心机制：解耦“规则裁决”与“局部建议”

LivePlan 的核心破局点，在于把“发现问题”和“给出解法”明确拆分为两个阶段，由不同的机制负责。系统由三个紧密联动的模块构成：**Monitor（监视器）**、**Advisor（建议器）** 以及 **Intervention Processor（干预处理器）**。

#### 1. 零 LLM 开销的确定性监视（Monitor）

在每一步执行前，Monitor 会维护两种进程视角的轨迹抽象表示：基于图结构的 Graphectory 和基于阶段语言符号的 Langutory。

- **Graphectory（动作图表示）**：将动作抽象为节点，动作转移抽象为边。如果在图中检测到回退边，表明 Agent 正在执行重复动作；如果出现连续的自环（Self-loop）或局部多节点回路，则直接触发动作震荡（Action Oscillation）的警报。

- **Langutory（阶段序列表示）**：将工具调用映射到宏观的软件工程阶段（如定位、打补丁等）。当同一阶段的连续步数超过阈值 $\theta_p$（论文根据未干预成功运行的统计分布，将其设定为 7 步），Monitor 便将其判定为长程停滞；若序列中跳过了关键的验证步骤，则判定为计划违背。

这套检测逻辑完全是确定性、基于规则的图遍历与序列匹配，执行时间仅需数毫秒，不产生任何 LLM API 成本，也不占用执行上下文。

#### 2. 抑制幻觉的局部上下文建议（Advisor）

只有当 Monitor 亮起红灯时，系统才会调用 Advisor。Advisor 的输入被严格约束在四个关键要素内：原始 Issue 描述、自上次干预以来的局部轨迹切片（避免庞大全局上下文带来的注意力分散）、上一次的建议内容、以及 Monitor 检测出的预定义漂移类型。

最为关键的是，Advisor **只给出针对下一步的微观高层行动建议（Next-step Recommendation），坚决不制定跨阶段的全局长远计划**。例如，当检测到 Agent 在编写 Ad-hoc 测试脚本上陷入长时间震荡时，Advisor 不会指导它如何重写整个测试套件，而是明确发出指令：“暂停当前测试脚本的调试，重新检查并直接修改核心控制流”。为了防止 Advisor 本身过于激进，系统还设置了冷却机制 $\theta_c$（设置为 5 步），确保模型在两次大模型干预之间有足够的自主探索空间。

#### 3. 分级阻断的干预处理器（Intervention Processor）

针对不同程度的漂移，干预处理器采取分级响应：

- **阻断式干预（Blocking Drift）**：针对完全重复、显式死循环等高危行为，处理器会直接拦截并丢弃 Agent 刚刚生成的有害动作，将其回滚并注入矫正建议，强制 Agent 换个思路重新推理。为了防止连续拒绝导致系统卡死，阻断次数设有上限 $\theta_i = 5$。

- **非阻断式干预（Non-blocking Drift）**：针对阶段漫游、过度探索等较软性的漂移，系统允许当前动作继续执行，但在环境反馈中附带纠偏建议，柔性引导下一步决策。

### 实验评测：在 SWE-bench Pro 上啃下“硬骨头”

为了验证 LivePlan 的有效性，研究团队不仅在学界常用的 SWE-bench Verified 上进行了测试，更重点聚焦于代码仓库规模更大、多文件修改要求更苛刻的 SWE-bench Pro 基准。实验选取了 DeepSeek-V3、Gemini-2.5-Flash 和 MiniMax-M2.5 三款主流执行模型，并搭配了不同层级的 Advisor 进行了总计 7,752 条轨迹的超大规模实测。

从整体解决率来看，LivePlan 相对原版 SWE-agent 取得了极其显著且稳健的增长。在难度更高的 SWE-bench Pro 上，LivePlan 将 DeepSeek-V3 的解决率提升了 12.33 个百分点，将 Gemini-2.5-Flash 提升了 15.24 个百分点；即便是代码能力极强、基线已经较高的 MiniMax-M2.5，也实现了 5.45 个百分点的增益。

相比之下，依赖“全量复盘并重写计划”的 SAGE 方法在多个模型上出现了性能负增长，甚至低于原版 Vanilla 基线。其根源就在于 SAGE 全局重规划产生的高置信度幻觉，给执行模型灌输了错误的修改代码与行号，诱使 Agent 在错误的方向上一错到底。

<img src="/images/2608.06701v1/upset_ru_difficulty.webp" alt="LivePlan 与 Vanilla 解决问题的难度分布重叠（UpSet 图）" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

仔细审视上图展示的解决用例分布，可以发现一个更具实践价值的现象：**LivePlan 带来的性能红利绝大多数落在中等与高难度任务上**。在 SWE-bench Verified 中，LivePlan 独占解决的实例（即原版 Vanilla 失败、LivePlan 成功的用例）超过一半属于 Hard 级别；而在 SWE-bench Pro 中，新增解决的实例几乎全部分布在 Medium（修改 2-3 个文件）和 Hard（修改 4-10 个文件）类别中。

这反映出一个核心事实：对于单文件、直觉式的简单 Bug，模型凭借自身微调权重便可直接解决，外部干预的价值有限；但随着问题复杂度上升，多文件之间的跳跃和反复调试不可避免地带来行为震荡，此时实时的确定性监控与轻量纠偏才真正成为了分水岭。

而在资源消耗方面，LivePlan 展现出了极高的工程实用性。由于规则监视器完全免费，全流程仅触发了少量的 Advisor 调用，平均每个任务实例由 Advisor 产生的开销仅在 0.01 美元至 0.06 美元之间，端到端额外成本稳定在 0.08 美元左右。对于 Gemini 和 MiniMax 等推理速度较快的模型，由于 LivePlan 有效压制了原本无意义的无序试错步骤，轨迹总步数被显著压缩，最终甚至降低了端到端的整体花费。

### 深入轨迹：干预是如何在微观层面生效的？

为了打开黑盒、厘清 Agent 究竟从哪些维度获得了提升，论文对轨迹的微观行为进行了系统的过程分析（Process-Centric Analysis）。

<img src="/images/2608.06701v1/inefficiency_patterns_heatmap.webp" alt="最终完成轨迹中的行为漂移热力图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从上图的漂移热力图可以看出，在未受干预的原版运行中，阻断性漂移（如动作震荡、计划跳步）极为普遍，特别是在 DeepSeek-V3 和 Gemini-2.5-Flash 上表现得尤为明显。引入 LivePlan 后，最终完成轨迹中的高危循环被大幅抹除——在受影响的轨迹中，高达 81.3% 的动作震荡和 51.4% 的计划违规被彻底消除。

但值得注意的是，诸如长时间导航（Prolonged Navigation）等非阻断性漂移依然有一定比例保留在轨迹中。研究团队指出，这并非系统缺陷，而是长程任务的必然属性：解决极难的工程 Bug 本身就需要一定程度的“试错式长程探索”，强行消除一切冗长步骤反而会抹杀模型发现复杂解法的可能。LivePlan 的精妙之处在于“掐死恶性循环，保留良性探索”。

<img src="/images/2608.06701v1/metrics_plan_compliance_heatmap.webp" alt="Vanilla 与 LivePlan 解决与未解决用例的计划合规性评分（PC）" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

上图对规划合规性（Plan Compliance, PC）的量化评估进一步佐证了机制的合理性。无论是在 Vanilla 还是 LivePlan 中，能够成功解决问题（Resolved, 标记为 R）的轨迹，其合规性得分普遍高于未解决轨迹（Unresolved, 标记为 U）。LivePlan 主要通过提高阶段完整性（PPC）和阶段顺序性（POC）来改善运行质量，即确保 Agent 按照“复现 - 打补丁 - 验证”的闭环完成任务。

在对具体案例的溯源中，团队观察到在 244 个从失败转化为成功（$U \rightarrow R$）的典型用例中，LivePlan 的首次干预全部发生在前三分之一的步骤内，平均每例经历 3.7 次局部微调。这种“及早发现、多次轻推”的连续纠偏模式，使模型在陷入上下文中毒（Context Poisoning）之前就被导回正轨，避免了沉没成本的积累。

### 为什么有时干预反而导致原本正确的用例失败？

在全方位的实证研究中，作者并未回避技术局限性，而是深入剖析了极少数原本成功但干预后失败（$R \rightarrow U$）的“性能回退”现象。这种客观归因对后续 Agent 系统的设计极具参考价值。

第一类根源在于**执行模型的非完全服从**。即使 Advisor 给出了完全正确的定位建议，较弱的执行模型有时无法理解或无法精确执行这一意图，甚至在超出连续干预阈值 $\theta_i$ 后，继续执拗地重复之前的错误动作。

第二类根源在于**基线本身的“脆弱成功”（Brittle Success）**。部分原版 Vanilla 成功用例其实带有极大的偶然性：模型在完全没有执行代码检索和本地测试验证的情况下，仅凭运气或预训练数据中的记忆直接“蒙”对了补丁。当 LivePlan 的监视器判定其存在“跳过验证”的违规行为并强制其进行本地复现时，由于环境配置或测试编写的琐碎报错，原本凑巧能过的代码反而被模型在慌乱中改坏了。

第三类则是**次要任务分心**。有时 Advisor 敏锐地察觉到了当前环境依赖配置不当，并建议 Agent 先行修复依赖。但执行模型一旦将注意力转移到环境编译、依赖冲突等琐碎的技术债务中，就彻底遗忘了最初的业务代码修复目标，最终因超时而告负。

### 总结与未来启示：从推理期纠偏到干预感知训练

这项研究清晰地给出了两点重要的范式启示：

其一，**大模型长程规划中的“判”与“导”必须在架构上解耦**。让大模型既当运动员又当裁判员，甚至让评审模型每一步都重新画图纸，已被证明是昂贵且容易被幻觉反噬的做法。利用确定性的程序逻辑、图算法去捕捉那些模式极其固定的死循环与停滞，把最宝贵的 Token 额度和注意力留给关键节点的局部策略纠偏，才是高性价比落地 Agent 系统的工业解。

其二，**推理期的干预框架必须与后训练（Post-training）走向协同**。单纯依靠在 Prompt 中塞入外部建议，往往受制于执行模型对复杂指令的依从能力。未来的编程 Agent 需要在训练阶段就引入“带有介入信号的轨迹数据”，让模型习惯在执行过程中倾听外部微调指令，并在过程级奖励模型（PRM）的约束下学会从震荡中自我复原。这篇工作所留下的数千条真实的干预与回退轨迹，也为后续端到端协同训练提供了极佳的经验切片。
