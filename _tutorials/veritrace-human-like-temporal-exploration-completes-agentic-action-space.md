---
layout: default
title: "VeriTrace：时序自主探索，Verilog基准首破100%"
description: "为了打破这一僵局，研究者提出了 VeriTrace 框架，核心是赋予专职排错智能体（Inspector）完全自主的“时序探索能力”（Agentic Temporal Exploration）——它可以像资深验证工程师一样，自由指定观察哪些内部信号、随意向前或向后拉动时间窗口、自主迭代验证假设。"
arxiv_id: "2608.02878"
paper_published: "2026-08-03"
published_at: "2026-10-05T13:15:07.965576+08:00"
topics:
  - "AI Agent"
tags:
  - "AI Agent"
  - "AI论文解读"
related_tutorials:
  - "actfovea-runtime-safeguarding-for-vla-policies-via-spatiotemporal-visual-action-"
  - "guardianagentbench-where-agents-fail-and-how-to-guard-them"
  - "cogguide-human-like-guidance-for-zero-shot-omni-modal-reasoning"
  - "one-hand-watches-the-other-dynamic-multi-agent-cooperation-for-sample-efficient-"
seo_title: "VeriTrace：时序自主探索，Verilog基准首破100%"
---

<p class="paper-original-title" lang="en">VeriTrace: Human-Like Temporal Exploration Completes Agentic Action Space</p>

利用大语言模型自动生成硬件描述语言（如 Verilog RTL），正在成为电子设计自动化（EDA）领域最受关注的方向之一。然而，无论底层模型从 GPT-4 换到 Claude 3.5 甚至更强的推理模型，现有的多智能体系统在标准基准测试 VerilogEval-V2 上的通过率，始终被锁死在 95% 左右的玻璃天花板之下。

> ArXiv URL：https://arxiv.org/abs/2608.02878

硬件设计的容错率为零，5% 的失败率在芯片工程中足以带来致命的流片失败。为什么无论怎么堆叠提示词或增加反思轮数，这最后的 5% 始终攻克不下？

论文《VeriTrace: Human-Like Temporal Exploration Completes Agentic Action Space》给出了一个直击本质的解释：过去自动化系统的“调试动作空间”（Debugging Action Space）是不完整的。现存系统要么只给大模型截取错误发生那一瞬间的输入输出切片，要么通过固定的静态规则反向追踪信号，本质上把复杂的硬件排错降级成了狭隘的“模式匹配”。

为了打破这一僵局，研究者提出了 VeriTrace 框架，核心是赋予专职排错智能体（Inspector）完全自主的“时序探索能力”（Agentic Temporal Exploration）——它可以像资深验证工程师一样，自由指定观察哪些内部信号、随意向前或向后拉动时间窗口、自主迭代验证假设。这一改动让 VeriTrace 在 VerilogEval-V2 基准上拿下了 100% 的 Pass@1 成绩，成为该基准上首个实现全功能正确的开源系统；在同等 Claude Sonnet 4.0 底座下，也比同类最强方案高出 5.1 个百分点。

<img src="/images/2608.02878/Overview.webp" alt="VeriTrace 架构总览图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 硬件调试的本质：时空交织的因果链

要理解 VeriTrace 的突破，首先需要认清软件代码调试与硬件 RTL 验证之间的鸿沟。

在常规软件开发中，程序大多是顺序执行的，报错往往伴随着明确的堆栈跟踪（Stack Trace）与行号定位，输入异常与输出崩溃之间通常具有较短的因果链条。但数字硬件电路完全不同，硬件本质上是海量门电路和触发器在时钟驱动下的并行运作。

在时序电路中，一个功能错误具有强烈的“时空交织性”（Spatiotemporal Bug）：仿真器在时钟周期 $t$ 报出的一个输出信号不匹配，其真正的根因往往潜伏在数十个周期之前的某个计数器溢出、有限状态机（FSM）的异常跳转，抑或是另一个看似无关的子逻辑块内部。

此前最具代表性的 RTL 生成系统（如 VerilogCoder、MAGE 和 ACE-RTL）虽然引入了“仿真在环”（Simulation-in-the-Loop）反馈，但在波形调试的交互设计上存在严重短板：

1. **截断的时间视野**：以 MAGE 为例，它采用状态检查点机制，仅捕获仿真检测到第一个不匹配时刻（$t_{\text{err}}$）的输入输出信号切片。然而，在 $t_{\text{err}}$ 观察到的仅仅是“车祸现场”，制造事故的“违章行为”早就在多个周期之前发生了。

2. **僵化的空间范围**：部分方案尝试通过抽象语法树（AST）沿数据流向后追溯信号，但这种静态回溯不仅难以跨越寄存器反馈环路，更无法根据中间观察结果动态切换追踪目标。

当智能体只能通过被动、机械喂入的局部数据来排错时，它根本无法建立对电路时序演进的因果理解，只能沦为在代码表面修修补补的碰运气过程。

人类工程师面对波形时绝不是这样工作的。在使用 Synopsys Verdi 等工业级波形查看工具时，验证工程师会先选定可疑信号，向前拉动时间窗口观察其历史翻转，提出“状态机是不是卡死在特定状态”的假设；为了验证假设，再把内部状态寄存器、使能信号和计数器拖入波形窗口，逐步向更早的时间点追溯，直到锁定诱发故障的根本逻辑。这种“假设—查询—求证—修正”的闭环，正是现有智能体所缺失的核心能力。

### 补全动作空间：将“诊断”与“修复”解耦

VeriTrace 的核心设计哲学，在于严格模拟人类硬件工程师的工序，将复杂的生成与排错任务分解为确定性控制流，并在架构上彻底拆分了“故障诊断”与“代码修复”。

整个工作流由四个专门的智能体协作完成：

- **Testbench Agent**：负责检查或修改原始测试平台（Golden Testbench），加入波形转储指令（如 `$dumpvars`），确保仿真过程记录下电路内部所有深层信号的完整波形数据 $\mathcal{W}$，为后续排错提供物理基础。

- **RTL Agent**：根据自然语言功能规范 $\mathcal{S}$ 生成初始 RTL 代码 $C$。这里沿用了高温度采样与快速语法检查策略，对于结构简单的组合逻辑和基础电路，直接一轮通过，避免让昂贵的调试循环介入简单任务。

- **Inspector（检查员智能体）**：一旦仿真出现功能错误，系统立刻切入时序探索闭环。Inspector 拥有对波形的绝对查询自由，专门负责顺藤摸瓜定位根因，最终输出一份结构化的诊断报告 $\mathcal{R}$。

- **RTL Debugger（调试器智能体）**：拿到 Inspector 提交的诊断报告后，只针对报告中指出的具体故障模块和逻辑缺陷进行外科手术式精准修改，生成修正后的代码 $C^{\prime}$。

这种“诊断者不写代码，修代码者不看原始波形”的解耦至关重要。过往尝试让单一模型既读庞大波形又修代码的做法，往往会导致大模型的上下文窗口被大量杂乱的时序数据淹没，注意力机制发散，最终导致“越改越错，越错越改”的雪崩效应。

值得注意的是，VeriTrace 在设计中刻意向 Inspector 隐藏了 Golden Testbench 的具体实现代码。智能体自我纠错的目标，是弥合 RTL 代码行为 $C$ 与自然语言设计规范 $\mathcal{S}$ 之间的语义偏差，而非去投机取巧地迎合测试代码中的断言。这种约束倒逼 Inspector 必须回归设计本身的逻辑规范展开推理。

### Agentic Temporal Exploration 是如何运转的？

Inspector 的核心能力被称为“自主时序探索”（Agentic Temporal Exploration）。论文将其形式化定义为一个基于 ReAct 范式的交互式动作循环。

设自然语言设计规格为 $\mathcal{S}$，当前缺陷代码为 $C$，仿真报告的首次故障时间戳为 $t_{\text{err}}$，完整波形数据为 $\mathcal{W}$。在每一个探索步数 $i$ 中，Inspector 自主决定一组目标信号列表 $\boldsymbol{\sigma}_i$ 以及观察时间窗口的起止范围 $[t^{\text{start}}_i, t^{\text{end}}_i]$。

系统环境通过底层工具提取对应的切片波形：




{% raw %}$$ \omega_{i}=\textsc{Trace}(\mathcal{W},\,\boldsymbol{\sigma}_{i},\,t^{\text{start}}_{i},\,t^{\text{end}}_{i}) $${% endraw %}



随后，Inspector 结合设计规范、源代码、初始报错时刻以及迄今为止累积的所有波形切片进行推理思考，并决定下一步动作：




{% raw %}$$ \tau_{i},\,a_{i}=\textsc{Think}(\mathcal{S},\,C,\,t_{\text{err}},\,\omega_{1},\dots,\omega_{i}) $${% endraw %}



这里的动作 $a_i$ 既可以发起新一轮针对其他信号或更早时序窗口的查询，也可以在确认根因后宣告结束并生成报告 $\mathcal{R}$。

<img src="/images/2608.02878/Inspector_gram.webp" alt="Inspector 在 Problem 155 上的时序自主探索过程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

上图展示了基准测试中著名的复杂有限状态机难题 Problem 155 的实际排错轨迹：

在仿真运行中，系统在 $t = 135\text{ns}$ 处捕获到输出信号不匹配。如果不具备时序探索能力，模型只知道此时输出错了，但无从得知为什么错。

进入调试后，Inspector 首先建立初始假设，将时间窗口回拨至故障点之前的 $[115\text{ns}, 135\text{ns}]$ 区间，并调取当前状态寄存器 `state` 与输入信号。观察发现，状态机在特定时钟沿发生了一次非预期的状态转移。

但这仍不是源头。Inspector 并没有盲目猜测，而是进一步发起了第二次查询，继续向更早的时间段 $[85\text{ns}, 115\text{ns}]$ 追溯，并同时调阅了中间使能信号与分支判断变量。通过两轮层层递进的时序探索，Inspector 彻底锁定了根因：问题不在输出逻辑，而是在数十个周期之前，次态跳转逻辑（Next-State Logic）中的一个分支边界条件判断符号写反了，导致整个状态机在后续运转中逐步偏离正确轨道。

最终，这份详尽的分析转化为了精准的诊断报告。RTL Debugger 接手后，只修改了那一行出错的状态跳转条件，代码一次性通过全部测试用例。

而在消融实验中，如果移除了 Inspector，失去时序追溯能力的 RTL Debugger 在面对 $t = 135\text{ns}$ 的错误输出时陷入了绝望的盲试。在经过多轮失败的局部修补后，模型甚至直接把整个模块全盘重写，导致该题目单题消耗了惊人的 619k Token，测试通过率依然为 0%。相比之下，VeriTrace 仅消耗 110k Token 便彻底解决了问题，Token 消耗缩减至前者的六分之一以下。

### 登顶 VerilogEval-V2：数据背后的质量重构

VeriTrace 在由 156 道数字电路问题构成的权威评测集 VerilogEval-V2 上进行了系统评测。为了抹平模型采样带来的随机扰动，评测均基于三次独立重复实验（$n=3$）计算 Pass@1。

在同构对比实验中，VeriTrace 展现出了决定性的架构优势：

在相同的 Claude Sonnet 4.0 底座下，此前的开源 SOTA 方案 MAGE 的复现成绩为 92.3% Pass@1，而 VeriTrace 达到了 97.4%，实现了 +5.1% 的显著绝对增益。这一差距完全来源于排错动作空间的完备性，证明了决定复杂硬件生成上限的关键变量在于智能体的调试自由度，而非单纯依靠基座模型的指令遵循能力。

当底座升级至推理与代码能力更强的 Claude Sonnet 4.5 时，VeriTrace 的 Pass@1 最终达到了 100%。这是 VerilogEval-V2 问世以来，首次有自动化系统实现全用例通关。


| 系统架构 | 是否开源 | 底座模型 | VerilogEval-V2 Pass@1 |
| :--- | :---: | :--- | :---: |
| Single Agent (Zero-shot) | — | Claude Sonnet 4.5 | 81.4% |
| VerilogCoder | 是 | GPT-4-Turbo | 94.2% |
| ACE-RTL | 否 | Custom & Claude Sonnet 4.0 | 95.5% |
| MAGE (原论文最高) | 是 | Claude 3.5 Sonnet | 95.7% |
| MAGE (基线复现) | 是 | Claude Sonnet 4.0 | 92.3% |
| **VeriTrace (Ours)** | **是** | **Claude Sonnet 4.0** | **97.4%** |
| **VeriTrace (Ours)** | **是** | **Claude Sonnet 4.5** | **100%** |

更具工程价值的发现体现在效率与消耗的取舍上。直觉上，引入专门的 Inspector 进行多轮波形查询似乎会大幅拉高 API 调用量和 Token 消耗。但统计数据打破了这一刻板印象：

在整个测试集的平均数据中，由于排除了盲目试错和大面积重写，VeriTrace 在调试阶段消耗的总 Token 数反而比没有 Inspector 的精简版下降了约 18%。尽管因为波形探索增加了交互轮次，导致端到端的绝对调用次数有所上升，但每一次交互所传递的信息密度极高、提示词目标明确，从根本上消除了“大模型因迷茫而疯狂输出冗余代码”的算力浪费。

对常驻调试循环的 7 道极限难题展开分类消融时，结果更为悬殊：

对于没有内部反馈的纯组合逻辑和无反馈时序电路，是否配备 Inspector 最终都能收敛到 100% Pass@1，Inspector 的价值主要体现在提速和降低消耗。

但对于包含复杂反馈回路的有限状态机（FSM）问题，没有 Inspector 的配置在这些题目上的 Pass@1 暴跌至 0%～33%，完全丧失了收敛能力；而 VeriTrace 凭借跨时间窗口的状态转移追踪，将这批硬骨头题目的通过率全部拉升到了 100%。这一对比清晰地表明：时序探索不是锦上添花的可选特性，而是解决深度时序逻辑错误的唯一有效路径。

### 工业化落地的距离与启示

尽管在 VerilogEval-V2 上刷出了满分，论文作者对当前成果的局限性依然保持着清醒的技术研判。

目前的 VerilogEval-V2 评测集本质上是由 156 个单模块电路构成的学术基准，并提供了现成且绝对正确的黄金测试平台。而在真实的芯片设计场景中，工程复杂性呈现指数级上升：芯片设计由多级模块嵌套而成，接口协议繁复，设计文档往往伴随着不完整或存在歧义的自然语言描述，更重要的是，信号空间极其庞大。在动辄上千个信号的工业网表中，模型能否依然准确选出关键节点，依然是未解的问题。

针对这一挑战，研究团队在多任务复杂基准 CVDP 上进行了前瞻性试水。在初步实验中，他们将自主时序探索机制嵌入到分层多智能体架构中，成功攻克了两道极具挑战性的从规范到多模块 RTL 的端到端设计难题。这表明通过时序波形回溯定位根因的认知模式，具有泛化至工业级复杂电路的潜力。

从更广阔的 AI 智能体技术演进来看，VeriTrace 带来了一个超越 EDA 领域的深刻启示：当我们在特定专业垂直领域遭遇大模型的性能瓶颈时，单纯扩大模型参数、微调专用底座或无脑增加反思轮数，往往收效甚微。真正的破局点，在于审视智能体操作环境的“工具接口”与“动作空间”是否完整。

在软件工程中，给模型一个完整的 Bash 终端和 LSP 服务能带来质的飞跃；在芯片设计中，赋予智能体自由拉动时间轴、自由圈定波形信号的完整动作空间，才真正补齐了机器通往专家级调试能力的最后一块拼图。
