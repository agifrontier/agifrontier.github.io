---
layout: default
title: "MTAC-IFBench：代码能跑却难合规，多轮编程严格遵循率不足10%"
description: "清华大学 CoAI 团队联合智谱 AI、电子科技大学等研究人员提出了 MTAC-IFBench ，针对“多轮自主代码智能体（Multi-Turn Agentic Coding）”的过程指令遵循能力建立了系统化评测基准。"
arxiv_id: "2609.14992"
paper_published: "2026-09-14"
published_at: "2026-09-16T13:15:08.728194+08:00"
topics:
  - "AI Agent"
tags:
  - "MTAC-IFBench"
  - "agentic coding"
  - "autonomous code agents"
  - "checklist verification"
  - "functional correctness"
  - "instruction verification"
related_tutorials:
  - "music-multi-step-instruction-contrast-for-multi-turn-reward-models"
  - "a-survey-on-large-language-model-based-autonomous-agents"
  - "harness-if-evaluating-instruction-following-across-instruction-surfaces-in-codin"
  - "handbookmd-a-benchmark-for-long-context-agentic-instruction-following"
---

<p class="paper-original-title" lang="en">MTAC-IFBench: Benchmarking Instruction-Following in Multi-Turn Agentic Coding</p>

<img src="/images/2609.14992/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在以大语言模型驱动的软件工程革命中，代码智能体（Code Agent）的演进正经历从“单次函数生成”到“多轮自主构建”的深刻范式迁移。类似 Claude Code、Kilo 这样具备自主规划、环境交互、工具调用及循环排错能力的开发框架，已经能够接管真实的端到端工程任务。然而，业界对这类智能体的评测，长期以来一直深陷于“结果导向”的偏狭视角：只要代码最终通过了单元测试，或者像 SWE-bench 那样跑通了预设 Patch，模型就被判定为表现优秀。

> ArXiv URL：https://arxiv.org/abs/2609.14992

这种评价方式忽视了真实企业软件交付中极为严酷的一面——**过程合规与约束遵循**。真实的工业级开发绝非单纯输出一段逻辑闭环的代码，开发人员和项目策略文件（如 `Claude.md`）会贯穿整个生命周期提出极其细致的过程约束。这些约束覆盖架构风格、特定工具的入参规范、严格的测试覆盖率阈值、文件命名规约、甚至是“不要重构某模块”或“将之前定义的接口参数重命名”等动态调整。

清华大学 CoAI 团队联合智谱 AI、电子科技大学等研究人员提出了 **MTAC-IFBench**，针对“多轮自主代码智能体（Multi-Turn Agentic Coding）”的过程指令遵循能力建立了系统化评测基准。这项研究揭开了一个此前被高分测试遮蔽的工程真相：即使是当前性能领先的 GLM-5.2、Claude-Opus-4.6 等顶级模型，在面对长程多轮交互中的密集约束时，也暴露出严重的疲劳与失控。即便最终生成的项目功能正常，绝大多数模型在单轮内严格做到“一条约束都不违背”的概率甚至不足 10%；当交互轮次递增到第 6 轮之后，严格遵循率更是断崖式跌至接近归零。

<img src="/images/2609.14992/intro.webp" alt="MTAC-IFBench 实例与约束动态演进示例" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

### 为什么说“代码能跑”掩盖了 Agent 的真实缺陷？

现有的代码评测生态存在显著的评价断层。一端是以 SWE-bench、SWE-Together、Terminal-Bench 为代表的 Agent 评测，它们聚焦于复杂的终端与仓库环境，考核的是功能正确性（Functional Correctness），约束数量几乎为零。另一端是以 IFEval、ComplexBench、CodeIF 为代表的指令遵循基准，它们虽然设计了多重指令限制，但主要局限在单轮通用对话或单文件算法脚本生成，完全脱离了长期、带状态、依赖环境工具调用的工程场景。

在长程开发中，智能体面临两大全新维度的复杂性：

第一是**高度异构的过程约束**。在 Agent 执行过程中，约束不仅针对最终输出的文本，更作用于中间动作。比如“每次修改前必须备份文件”、“禁止使用 `cat` 读取全量日志而必须用 `tail -n 50`”、“测试未达 80% 覆盖率前不可提交”等。这些指令分布在工作流编排、工具调用、代码架构和测试规约等多个层面。

第二是**多轮交互下的动态演进与上下文退化**。真实需求很少一步到位。人类开发者往往先要求搭骨架，随后提出新功能，中途还会根据前序执行结果随时推翻或微调先前的规定。智能体不仅要在超长上下文中记住全局规则，还必须准确区分“何时追加约束”与“何时覆盖旧约束”，并在长达数千甚至上万 Token 的工具输出噪声中保持对规则的持续敬畏。

MTAC-IFBench 填补了这一空白。如上图所示，一个典型的任务不仅包含仓库顶层的长效策略文件，还包含横跨多轮的需求递进。用户在每一轮交互中都会动态插入或重构规范要求。当功能逻辑与过程约束相互交织，模型面临的认知负荷被急剧放大。

### 体系化拆解：6大类与18个维度的约束解构

为了让评测具备可解释性与完备性，研究团队从服务超百万用户的真实商用代码智能体平台以及现有文献中，提炼出了一套覆盖软件开发全生命周期的分层约束分类体系，共包含 6 个主类别与 18 个子类别：

1. **工作流约束（Workflow）**：规范智能体的动态行为与生命周期。包含**工具使用（Tool Usage）**，如强制指定参数格式或加壳调用；**编排规划（Orchestration）**，如动作执行的前后顺序、强制分解步骤；**测试验证（Testing）**，如指定的断言库、覆盖率门槛与验证记录归档。

2. **代码风格与设计（Style & Design）**：约束架构与规范。涵盖**编程范式（Paradigm）**，如限定使用面向对象或函数式；**命名约定（Naming）**；**代码格式与排版（Format）**。

3. **环境与依赖（Environment & Dependency）**：限定工程上下文。包括**依赖版本（Dependency）**、**运行环境与系统变量（Environment）**、**文件操作路径与落位（File Manipulation）**。

4. **语法与语言特征（Syntax & Language）**：包括**特定语言规范（Language Specification）**、**受限语言特性（Restricted Features）**（如禁用全局变量或禁止使用反射机制）。

5. **内容与文本约束（Content & Text）**：针对非代码资产或文本呈现。包含**文件编码（Encoding）**、**禁止包含的内容（Disallowed Content）**、**精确内容匹配（Exact Content）**。

6. **数量与限制（Quantity & Limit）**：涉及精确的数值把控。包括**数量控制（Quantity）**，如生成指定数量的类或模块；**规模阈值（Threshold）**，如单个函数代码行数限制。

<img src="/images/2609.14992/framework.webp" alt="MTAC-IFBench 的构建流程与混合验证框架" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

基准构建流程采取了半自动化合成与严苛人工校验相结合的双轨机制。首先汇聚涵盖前端开发、数据分析、通用工具、算法实现等领域的单轮高质量种子任务，利用先进模型（如 Gemini-3.1-Pro）通过渐进式提示词扩写为 5 到 10 轮的多轮指令流。随后，借助分类法指导模型（如 Seed-2.0-Pro）将各类约束均匀织入全局策略文件与各轮交互中。在此过程中，系统动态模拟了约束的新增与替换操作，并实施自检与回溯重试机制。

在数据清洗阶段，17 名专业背景人员对合成任务的清晰度、一致性、可行性进行了逐一审查，剔除一切存在逻辑冲突或歧义的样本。最终沉淀出 100 个具有高难度的代表性实例。这 100 个任务平均交互轮次达 7.04 轮，单实例平均包含 91.33 条约束（折合每轮 12.97 条约束），其约束密度与复杂性远超目前所有同类工作。

<img src="/images/2609.14992/constraints.webp" alt="MTAC-IFBench 数据集中的约束类型分布" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

针对这样密集的规则体系，单纯使用字符串匹配或简单的正则表达式根本无法完成判定。MTAC-IFBench 创新性地引入了“逐项清单（Checklist）+ 混合验证器”的评估协议。对于每一轮交互，基准都会生成对应的约束清单；在全流程结束后，再生成最终功能清单。

评估器由双轨驱动：对于诸如字符匹配、文件编码、静态路径等确定性约束，采用由 Claude-Opus-4.6 编写并人工审查的**代码验证脚本（Code Evaluation）**执行硬性检查；对于编程范式、动作编排、测试流程等高阶逻辑与主观要求，则部署基于 Claude Code 框架的**裁判智能体（Judge Agent）**，驱动底层大模型深入生成的代码仓库环境、执行构建产物、甚至调用 Playwright 驱动浏览器执行 UI 交互与截图分析，完成深度动态核验。

### 实验指标：从单项完成率到严苛的全局达标

为了全景呈现模型的真实水平，研究设计了涵盖“过程约束”与“最终功能”的两套核心指标：

- **约束成功率（CSR, Constraint Success Rate）**：在第 $t$ 轮中，被满足的约束数量占该轮所有约束的比例：

  


  {% raw %}$$\mathrm{CSR}_{t} = \frac{1}{\lvert \mathcal{C}_{t} \rvert} \sum_{j=1}^{\lvert \mathcal{C}_{t} \rvert} z_{t,j}$${% endraw %}


- **严格约束遵循率（C-ISR, Constraint-Instruction Success Rate）**：在第 $t$ 轮中，当且仅当该轮的所有约束全部无误被满足时才计为 1，哪怕漏掉一条也记为 0：

  


  {% raw %}$$\mathrm{C\text{-}ISR}_{t} = \prod_{j=1}^{\lvert \mathcal{C}_{t} \rvert} z_{t,j}$${% endraw %}


- **功能成功率（FSR）与严格功能满足率（F-ISR）**：针对任务最终交付的功能清单，分别统计功能的平均满足比例以及完全满足所有功能特性的严格达成率。

- **构建成功率（BSR）**：衡量最终交付的工程项目是否能够顺利通过编译、打包或启动执行。

评测覆盖了 11 款主流大语言模型，包括商业闭源前沿模型 Claude-Opus-4.6、Gemini-3.1-Pro、Qwen3.6-Plus、Seed-2.0-Pro、Claude-Haiku-4.5，以及知名开源/权重公开模型 GLM-5.2、GLM-5.1、DeepSeek-V4-Pro、DeepSeek-V4-Flash、Kimi-K2.6 和 Qwen3.5-27B。基准框架默认挂载于被广泛使用的 Claude Code (v2.1.14) 之上，以确保 Agent 环境的一致性。

### 核心发现：断崖式衰减与“伪成功”陷阱

实验结果展示出了极为残酷的一面。在整体表现上，综合排名第一的开源模型 GLM-5.2 在平均约束成功率（CSR）上达到了 80.4%，但这意味着即便顶尖模型，依然会漏掉约 20% 的过程指令。更值得警醒的是代表严谨度的 C-ISR 指标：绝大部分顶级模型的全流程平均 C-ISR 徘徊在 5% 至 10% 之间，GLM-5.2 也仅为 12.7%。这意味着在绝大多数交互轮次中，模型几乎必然会在某些细节要求上出现遗漏或违规。

不仅如此，随着对话轮次的不断加深，智能体的约束遵循能力呈现出急剧的“长程衰退（Long-Horizon Degradation）”。在第 1 至 2 轮交互中，模型的 CSR 尚能维持在 70% 至 85% 左右；但到了第 7 轮以上，所有模型的 CSR 均持续下滑，降幅普遍达到 10 到 20 个百分点。而在单轮全达标率（C-ISR）上，衰退表现得更为彻底：在 6 轮交互之后，多数模型的 C-ISR 直接跌至 0%。长程上下文中海量的工具调用返回信息严重冲淡了模型对初始策略和历史约束的注意力权重。

另一个极具启示性的现象是：**最终功能的成功，掩盖了过程约束的严重缺陷**。从功能层面看，主流模型在 FSR 上的得分并不低，GLM-5.2、Gemini-3.1-Pro 等模型均能达到 65% 以上的功能覆盖率，严格功能满足率（F-ISR）也能达到 20% 到 28%。然而，各模型的 C-ISR 却普遍显著低于 F-ISR。很多时候，智能体完全是以“破坏既定架构规则、跳过测试要求、忽略环境依赖策略”为代价，用黑盒打补丁的方式强行跑通了最终功能。传统只看输出成果的 Benchmark，实际上一直在为这类“饮鸩止渴”式的高风险 Agent 代码拍手叫好。

<img src="/images/2609.14992/harness_comparison.webp" alt="不同智能体运行框架（Claude Code 与 OpenCode）下的表现对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 框架依赖：运行底座（Harness）带来的隐形差距

代码 Agent 的表现并不完全等同于模型本体的能力，外部脚手架（Harness）起到了决定性的中枢作用。研究团队在 Claude Code 与 OpenCode 两种不同架构底座上进行了控制变量实验。

如上图所示，当同一个模型从 Claude Code 切换到 OpenCode 时，几乎所有模型的 CSR 与 C-ISR 都出现了肉眼可见的滑坡。以 GLM-5.2 为例，其 CSR 从 80.4% 下滑到 76.8%，而严格指标 C-ISR 更是直接从 12.7% 跌落至 6.7%。

分析表明，Claude Code 在上下文剪枝、工具调用错误回传提示、多轮历史压缩策略上做了极高密度的工程优化。更合理的上下文调度框架能够有效延缓模型注意力分散的速度。但从排名一致性（Kendall 相关系数）来看，头部梯队（如 GLM-5.2 与 Claude-Opus-4.6）在两个框架下的相对领先地位依然稳固。这表明底层模型本身的指令理解深度仍然是决定上限的基石，而评测代码 Agent 时，明确规范并公布运行 Harness 具有极其重要的科学严谨性。

<img src="/images/2609.14992/constraint_from_categories.webp" alt="大模型在不同约束类别上的表现差异" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

### 软肋何在：静态规则轻松应付，动态协同与精确控制全线崩溃

将 18 个细分约束类别单独拆解后（如上图），模型的强弱项呈现出两极分化的态势：

在**单点、静态的确定性规则**上，大模型表现出接近完美的掌控力。例如文件编码（Encoding）与文件落位（File Manipulation），各类模型的平均 CSR 普遍在 95% 以上。模型十分擅长处理这些“局部且无需推演”的明确模式。

然而，在涉及**全局系统协同与连续状态追踪**的领域，模型暴露出极其严重的无能：

1. **工作流编排（Orchestration）与编程范式（Paradigm）**：这些类别要求智能体在多步执行中时刻保持行为模式的连贯性，例如“每编写一个功能模块前必须先输出架构拆解计划，并采用工厂模式进行解耦”。模型在几轮工具调用后，往往会迅速退化为直觉式的单行修改，完全将全局范式抛在脑后。

2. **精确数量控制（Quantity）**：这是全场得分最低的重灾区之一。当指令要求“重构出恰好 4 个派生类”、“一次性编写 5 个单元测试用例”或“单函数控制在 30 行以内”时，模型的遵循率出现暴跌。这暴露出当前自回归模型缺乏精确的状态计数与全局校验回路，无法像编译器一样对代码规模实施硬性自省。

<img src="/images/2609.14992/constraint_from_sources.webp" alt="不同约束来源下的平均 CSR 演进趋势" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 策略文件 vs 动态交互：更新旧约束比听从新要求难得多

约束从何而来，也极大影响了 Agent 的执行遵从度。如上图所示，研究人员将约束来源解构为三种：仓库全局策略文件（Policy，如 `Claude.md`）、用户各轮新增约束（Added）、以及用户针对先前要求的动态修改与覆盖（Replaced）。

数据揭示出两个关键规律：

首先，**策略文件的遵从表现明显优于多轮交互中插入的约束，且具备更强的抗衰减能力**。由于策略文件通常作为系统级上下文或长效前缀始终驻留，模型在长程推进中对其具有持续的感知力，CSR 始终维持在 85% 左右的高位。

其次，**在用户指令内部，“修改旧约束”的难度远超“添加新约束”**。当用户说“增加一个支持 JSON 导出的接口”时，模型能较好地追加代码；但当用户在第 4 轮提出“把第 2 轮中使用的蛇形命名（snake_case）全部推翻，改为驼峰命名（camelCase），但其余逻辑保持不变”时，模型的 CSR 遭遇了最为严重的下挫。面对冲突指令时，模型极容易陷入记忆混淆，既无法彻底擦除旧提示词在隐空间造成的惯性偏置，又难以在不破坏现有功能的前提下实施系统级重命名。理解并追踪人类意图的“动态演化”，是当下代码智能体最为脆弱的技术短板。

### 迈向高可靠 Agent 的必经之路

MTAC-IFBench 带来的工程警示十分明确：**在代码智能体领域，仅仅依靠单元测试或黑盒结果驱动的评测时代正在过去。**

一个在 GitHub 仓库中能够自主修复 Bug、甚至通过 SWE-bench 评测的 Agent，在真实工业产线中可能会轻易违反代码安全审计规则、肆意篡改既有的系统架构模式，或者对运维团队指定的日志规约充耳不闻。这种“代码能跑但违规累累”的技术债务，往往比单纯的编译报错更具隐蔽性和破坏性。

要构建出真正可投产的可靠代码智能体，研究界与工业界不能仅依靠堆叠参数量或延长上下文窗口来被动抵抗注意力的退化。未来的破局之道，更在于体系化的系统设计：

- 构建独立的**状态追踪与元认知审计模块**，使 Agent 在向开发环境派发工具调用前，自动执行前置与后置的 Checklist 过滤；

- 探索针对“指令动态撤销与重写”的专项后训练（Post-training）策略，强化模型在长上下文中的冲突消解能力；

- 推动代码 Harness 从“被动环境交互器”向“主动规则执行沙箱”演进。

MTAC-IFBench 的发布，将软件工程中隐形却致命的“规矩与秩序”搬上了评测擂台，为下一代高严谨度、真正具备工业化交付能力的代码智能体开发指明了清晰的技术航道。
