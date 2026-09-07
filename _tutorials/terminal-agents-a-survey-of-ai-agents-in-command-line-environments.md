---
layout: default
title: "Terminal Agent最新综述！七维能力框架解构大模型终端执行生态"
description: "Terminal：该工作首次从工作负载级边界出发，明确将终端 Agent 界定为“任务核心推进闭环依赖于命令行执行、文本反馈与有状态环境交互”的智能系统，并提出了覆盖执行全生命周期的七维终端能力画像（Competence Profile）。"
arxiv_id: "2608.20485"
paper_published: "2026-08-20"
published_at: "2026-09-07T13:15:08.344824+08:00"
topics:
  - "AI Agent"
tags:
  - "Benchmark families"
  - "Executable trajectories"
  - "Fixed-condition diagnostics"
  - "Process-level evaluation"
  - "Replayable traces"
  - "Stateful environment interaction"
related_tutorials:
  - "scaling-environments-for-llm-agents-in-the-era-of-learning-from-interaction-a-su"
  - "contextweave-a-real-world-workflow-benchmark"
  - "gpqa-a-graduate-level-google-proof-qa-benchmark"
  - "appdeltaworld-transition-grounded-delta-code-world-model-for-mobile-gui-agents"
---

<p class="paper-original-title" lang="en">Terminal Agents: A Survey of AI Agents in Command-Line Environments</p>

<img src="/images/2608.20485v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

当前大语言模型正在经历从“被动生成文本的聊天界面”向“能够在外部环境中自主行动的数字实体”的根本转变。在这个演进过程中，终端命令行环境（Terminal）凭借紧凑的文本交互、强大的系统级组合能力和有状态的运行时支撑，迅速成为代码智能、运维排障、自动化工作流乃至未来通用计算机操作的核心执行底座。然而，学术界与工业界过去长久将终端执行能力碎片化地分散在软件工程代码修复、工具调用（Tool Use）或图形界面（GUI/OS）计算机操作等各个领域，缺乏一套统一的分析透镜。

> ArXiv URL：https://arxiv.org/abs/2608.20485v1

来自上海创新研究院、上海交通大学、香港理工大学以及同济大学的研究团队发布了首篇全面聚焦终端智能体（Terminal Agents）的系统性综述《Terminal Agents: A Survey of AI Agents in Command-Line Environments》。该工作首次从工作负载级边界出发，明确将终端 Agent 界定为“任务核心推进闭环依赖于命令行执行、文本反馈与有状态环境交互”的智能系统，并提出了覆盖执行全生命周期的七维终端能力画像（Competence Profile）。更关键的是，论文剖析了困扰整个社区的“归因困局”：在诸多高分评测中，Agent 性能的跃升究竟来自大模型本身的推理突破，还是源于 Harness（工程脚手架）与运行时的隐形赋能？

<img src="/images/2608.20485v1/terminal-agent-definition.webp" alt="终端 Agent 的定义与边界" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么终端是 Agent 的核心执行底座，而非普通工具接口？

理解终端 Agent 的首要前提，是将“终端作为底层基质（Substrate）”与“表层命令行接口（CLI）”区分开来。传统工具调用往往将环境交互抽象为无状态、即插即用的函数调用，模型只需输出规范的 JSON 即可；而在图形界面（GUI）中，模型需要消耗大量视觉 Token 来感知界面布局，并依赖模糊的鼠标点击动作。

终端环境提供了一种完全不同的交互模式：它是高度结构化且脚本化的文本世界。模型直接与解释器（Shell）对话，面对的是一个具备持久状态的文件系统、依赖项、进程树、网络配置与系统权限的真实运行时。每一条命令的执行都会产生诸如标准输出（stdout）、错误输出（stderr）、退出状态码（Exit Code）、代码差异（Diff）以及堆栈回溯（Traceback）等显式反馈。这种文本进、文本出的强因果闭环，让大模型的 Token 生成接口与物理计算系统的状态转移之间实现了近乎零阻抗的对接。

正因为这种紧耦合，本文给出了明确的工作负载边界判定。只要一个系统的核心推进逻辑高度依赖命令执行、文本反馈指导下一步动作以及与持久状态深度互动，它就属于终端 Agent。相反，仅仅通过 CLI 包装发送一次性提示词、或者调用一个孤立脚本却不依赖终端状态反馈的系统，则只是表层 CLI 工具，被严格排除在此范畴之外。

<img src="/images/2608.20485v1/terminal_grounded_scope.webp" alt="终端 Agent 的操作范围与边界测试" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 解构终端能力：七维画像超越单一的代码修复

在以 SWE-bench 为代表的流行评准占据统治地位的今天，许多从业者容易将“软件工程能力”等同于“终端 Agent 能力”。但这种等同掩盖了大量的执行盲区。软件仓库修复往往预设了现成的测试套件和相对干净的环境，而真实的终端操作不仅要“会修代码”，更要面对混沌的底层系统。

为此，综述构建了包含七个维度的终端能力画像（Terminal Competence Profile），将抽象的系统交互行为分解为可观测、可衡量的能力单元：

1. **动作生成与命令选择（Action Formulation）**：不仅涉及命令语法本身，还包含命令组合逻辑、管道连接以及多步 Shell 脚本的构造能力。

2. **文本反馈与状态感知（Feedback Interpretation）**：从巨量、杂乱的标准输出、报错回溯与进程日志中提炼关键因果信号，过滤环境噪声。

3. **环境管理与依赖修复（Runtime Management）**：面对缺失依赖、版本冲突、端口占用和环境漂移时，主动搭建、诊断和修复运行时的能力。

4. **状态跟踪与长程上下文一致性（State Tracking）**：在数十个交互轮次中，持续维护对当前工作目录、文件修改历史、环境变量与后台进程状态的准确记忆。

5. **内生验证与自我检验（Verification）**：在没有现成测试框架时，主动构造断言、编写临时检查脚本或分析系统指标，以确认阶段性目标是否达成。

6. **故障诊断与容错恢复（Recovery）**：当遇到非零退出码或死循环时，放弃无效假设、回滚脏数据并调整策略的能力，这是决定 Agent 能否自主闭环的分水岭。

7. **执行治理与安全防护（Governance）**：在操作高危指令（如带有破坏性的删除、无界网络调用、权限提升）时，展现出的安全边界感知、权限控制配合与风险阻断能力。

这七个维度在运行过程中并非孤立存在，而是高度咬合。例如，一次成功的故障恢复，往往需要精准的文本反馈解析、严谨的当前状态追踪以及即时的内生验证；而长程任务的推进，则始终受到安全治理策略的约束。

### 系统架构的分层解耦与“归因困局”

终端 Agent 的实际表现并非单单取决于基础大模型（Base LLM），而是由模型、交互接口、控制逻辑、Harness（脚手架）与运行时环境五者共同决定的复杂系统涌现。论文将终端 Agent 的技术演进划分为四个重叠的重心转移：从早期的工具增强提示词工程（Shift 1），演进到结构化可执行动作（Shift 2），再到将终端交互视作一等公民设计目标（Shift 3），直至当下兴起的以运行时与 Harness 为中心的原生架构（Shift 4）。

<img src="/images/2608.20485v1/technical_evolution.webp" alt="终端 Agent 技术的演进路径" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在典型的分层架构中，系统责任被清晰地拆解到四个核心层级：

- **接口与感知层（Interface & Observation）**：负责命令格式化与观察值整形。例如 SWE-agent 提出的 ACI（Agent-Computer Interface），专门设计了轻量级、对模型更友好的文件浏览与编辑指令，将长文件切片呈现，防止模型被海量文本撑爆上下文窗口。

- **运行时与工作区层（Runtime & Workspace）**：提供执行容器（如 Docker、轻量级 VM）和状态持久化机制，决定了系统是否支持断点恢复与多环境切换。

- **控制、验证与恢复层（Control, Verification, Recovery & Governance）**：调度多角色协作（如 Planning、Execution、Review 角色分离），管理回滚机制，并注入权限沙箱与拦截策略。

- **Harness 与上下文管理层（Harness & Context）**：负责跨轮次会话压缩、动态信息检索、Prompt 组装以及自动化重试逻辑。

<img src="/images/2608.20485v1/layered_loop_architecture.webp" alt="终端 Agent 的分层循环架构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

这种高度复杂的系统设计带来了一个极其棘手的科学问题：**组件归因困难（Attribution Difficulty）**。当前许多在榜单上刷新 SOTA 的终端 Agent，其提升究竟源自底层模型推理能力的跃迁，还是来自于脚手架工程（如 Meta-Harness、AutoHarness）注入的特定先验流程？

研究表明，单纯依靠外部精心设计的上下文压缩规则、定制化的编辑工具集以及重试容错机制，往往能够掩盖模型弱项，使表现看似优异；然而一旦脱离特定 Harness 预设的保护罩，系统的泛化性能就会断崖式下跌。Agentless 框架便曾有力证明：一个没有复杂交互循环的静态流水线，在某些代码修复任务上完全可以媲美耗费巨大 Token 的复杂交互式 Agent。这意味着，当前评测若不将模型与 Harness 的变量进行控制分离，就无法得出客观的模型能力演进结论。

### 训练飞轮：为什么失败轨迹是更珍贵的学习信号？

为了让 Agent 学会玩转终端，如何从可执行交互中提取训练信号成为当前数据工程的核心战场。传统的指令微调（SFT）习惯于收集静态的“提示-响应”对，但在终端环境中，单步动作的意义极其有限，必须以完整的状态转移轨迹（Stateful Trajectory）作为最小学习单元。

目前社区主要依赖三大数据源：原生终端仿真环境（如 CLI-Gym、TerminalTraj）、真实可执行仓库环境（如 SWE-bench 衍生集）以及合成与排障环境。然而，数据流水线中的一个普遍误区正在制约 Agent 能力的上限——对“成功轨迹”的过度偏爱。

<img src="/images/2608.20485v1/competence_acquisition_ecosystem.webp" alt="终端能力获取与适应生态体系" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在常规的强化学习（RL）或 SFT 过滤策略中，数据清洗管道（如 CLEANER）往往会无情剔除所有未完成任务的轨迹，只留下一步到位的成功样例。但综述敏锐地指出：**终端操作的核心壁垒恰恰在于排错与恢复**。在真实开发中，即便是资深工程师也会频繁遭遇包安装失败、语法错误或参数不匹配。如果训练集全部由“上帝视角”般一击必中的平滑轨迹构成，模型就永远无法学会在遭遇非零退出码时如何读懂报错、如何执行 `git checkout` 或删除脏状态、如何在死胡同中回滚并调整技术路线。

正因如此，诸如 AgentHER 这种利用后验重放（Hindsight Replay）对失败轨迹进行二次标签重写、保留错误诊断与多轮修复过程的探索，正在成为更具价值的技术路径。终端能力的内化，本质上是让模型理解“动作后果（Action Consequences）”——知道某条命令不仅能达成既定目标，还会在特定边界条件下产生破坏性副作用，并掌握将系统从危险或损坏状态拯救出来的技能。

### 评测的真实困境：终态基准的盲区与重放危机

在评测侧，以 SWE-bench、SWE-bench Lite 以及多语言的 Claw-SWE-Bench 为代表的基准，成功确立了“可执行验证”的黄金标准：一个 Patch 到底修得对不对，通过预先埋设的隐藏单元测试说了算。然而，以最终测试通过率为唯一指挥棒，正在掩盖评测的多重维度失效。

首要问题是**过程不可见性（Process Inobservability）**。一个 Agent 可能耗费了数万个 Token，在终端中无规律地盲目尝试了上百次脆弱的暴力替换，碰巧碰对了测试用例；另一个 Agent 则逻辑严密、步步为营，只在最后一步因细微路径拼写错误被扣分。终态指标无法区分二者过程质量的云泥之别，不仅无法暴露中间交互过程中的高危越权行为，也无法测量 Agent 到底是在推理还是在投机。

其次是**运行时的动态污染与重放危机（Replayability Crisis）**。终端 Agent 与静态问答模型不同，其执行结果深度耦合于当时的网络状况、第三方软件仓库（如 PyPI、npm）的版本依赖、容器内核权限甚至系统随机种子。如果在评测时未能精确固定环境切片，相同的代码和模型在数月后重试可能完全无法复现原先的高分。

更为隐蔽的则是**数据泄漏与基准过拟合**。开源仓库历史的公开透明性，使得部分基准题目极易在无意间进入基础模型的预训练语料库中。如果评测没有覆盖动态演进的、端到端的环境搭建、异常恢复和跨会话系统操作，榜单分数很可能演变成一种对记忆能力的伪测量。

### 走向下一代终端智能体：研究图景与核心挑战

基于对架构、学习范式与评测体系的全面解构，论文为终端智能体领域描绘了未来亟待攻坚的核心图景：

在**能力泛化**维度，终端 Agent 必须跨出软件仓库修复这一狭窄温床，全面走向异构的基础设施管理、复杂的科学计算流编排以及企业级网络运维。这要求智能体不仅懂 Python 和 Git，更要具备对跨机器通信、后台长服务生命周期管理以及动态系统资源的深度调度能力。

在**评测标准**维度，社区迫切需要从单纯的终态二元判定（Pass/Fail），升级为覆盖七维画像的过程级、可复现诊断框架。构建包含确定性沙箱快照、完整录屏日志与标准化细粒度开销（Token 消耗、执行时长、恢复步数比率）的可追踪基准，是撕开“模型-脚手架归因黑盒”的必由之路。

在**安全与治理**维度，随着终端 Agent 逐步被赋予真实系统的写入与执行权限，基于自然语言的指令劫持（Prompt Injection）、针对底层运行时的逃逸攻击、以及在长程自主操作中不可逆的数据删除与环境污染风险剧增。设计内生于运行时、支持最小权限动态授权和细粒度意图核准的防御机制，将是终端智能体从受限实验走向生产环境不可逾越的前提。

从被动执行预设脚本的机械工具，到自适应探索计算环境的自主实体，终端 Agent 正在重构人类与计算机底层软硬件交互的方式。这项研究通过严谨的边界界定、立体的能力框架与冷静的系统归因诊断，不仅清理了过去碎片化认知带来的迷雾，更为构建下一代真正稳健、可信、高度自主的系统级智能体奠定了坚实的理论与工程基石。
