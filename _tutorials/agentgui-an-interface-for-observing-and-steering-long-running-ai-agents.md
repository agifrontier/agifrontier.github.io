---
layout: default
title: "AgentGUI：苏黎世联邦理工开源长程Agent控制台，排查提速38%"
description: "为了让多智能体协作具备直观的实体感，AgentGUI 引入了“虚拟工位（Desk）”的设计隐喻。在系统主面板中，各个并发运行的任务被抽象为一个像素风格的虚拟办公室，每一个独立的 Agent 都拥有专属的“办公桌”。用户发起新任务只需点击空工位、输入指令并拖入上下文文件。"
arxiv_id: "2607.26300"
paper_published: "2026-07-28"
published_at: "2026-09-22T13:15:08.426544+08:00"
topics:
  - "AI Agent"
tags:
  - "AgentGUI"
  - "agent trajectory visualizations"
  - "automated drift prevention"
  - "locally hosted GUI"
  - "long-running AI agents"
  - "manual steering"
related_tutorials:
  - "os-agents-a-survey-on-mllm-based-agents-for-general-computing-devices-use"
  - "gui-360-a-comprehensive-dataset-and-benchmark-for-computer-using-agents"
  - "evo-harness-context-to-harness-skill-compilation-for-self-evolving-agents"
  - "the-compaction-cliff-in-long-running-ai-agent-memory"
---

<p class="paper-original-title" lang="en">AgentGUI: An Interface for Observing and Steering Long-Running AI Agents</p>

<img src="/images/2607.26300v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在大模型技术从单轮对话向自主决策演进的过程中，基于工具调用的自主智能体（LLM Agent）正在接管越来越复杂的长程任务（Long-running tasks）。从端到端构建软件系统、跨文件重构代码，到自动化推演科学假设乃至直接控制操作系统，这些智能体往往会脱离人类视线，在后台自主运行数小时甚至数天。

> ArXiv URL：https://arxiv.org/abs/2607.26300v1

然而，伴随智能体自主能力膨胀而来的，是极其严峻的“可观测性滞后”（Oversight lag）。一个长期运行的 Agent 会生成一段交织着内部思维链（Thinking traces）、代码执行输出、文件系统修改与网络请求的庞杂文本日志。要对这类黑盒系统进行有效的监督与复盘，人类必须花费大量时间去翻阅杂乱无章的原始记录，这在很大程度上抵消了智能体原本承诺带来的效率收益。更严重的是，当智能体中途出现目标漂移（Goal drift）、卡入死循环或产生幻觉时，由于缺乏交互抓手，人类往往无法在不中断任务的前提下进行动态校正。

<img src="/images/2607.26300v1/townview.webp" alt="The AgentGUI dashboard" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了化解这种失控风险，来自苏黎世联邦理工学院（ETH Zürich）的研究团队开发并开源了面向长程 AI Agent 的交互控制台——**AgentGUI**。它是一个采用本地优先架构的图形化平台，首次将多任务并发调度、多层次轨迹可视化、动态运行时干预（Steering）以及基于大模型的自动化防漂移审计整合在同一界面下。受控用户实验显示，借助 AgentGUI 的可视化重构，研究者在庞杂日志中排查关键信息的耗时缩短了 $38\%$（$p = 0.023$），理解准确率由 $80\%$ 提升至 $93\%$。而在小参数本地模型的长程实验中，其内建的自动化审计机制直接将任务完成率最高拉升了 $34$ 个百分点。

### 为什么长程日志会让人类监督彻底失效？

阅读长程 Agent 的原始轨迹不仅对人类来说痛苦不堪，即使让前沿大模型去排查这些文本，效果也差强人意。前沿评测基准 TRAIL 的测试数据显示，即便是最顶尖的闭源模型，在面对杂乱的原始调用日志时，也只能定位出一小部分系统错误。其核心症结在于传统 Agent 运行环境（Harness）缺乏以人为中心的交互界面设计。

传统的 Agent 运行时（如 SWE-agent、OpenHands 等）大多将所有事件按时间顺序打成线性文本流。在这种表示方式下，模型输出的自然语言反思、JSON 格式的工具调用入参、终端 Standard Output、文件读写内容以及底层 API 消耗数据全部挤在一起。这种信息密度的不均使得人类在排查异常时难以快速切中要害。

与此同时，现有工具在设计取向上存在严重的割裂。一类工具（如 Agent-flow、AgentDiagnose、AgentLens）专注于离线分析与轨迹拓扑渲染，能把子任务分支画得清晰漂亮，但它们仅仅是只读的“仪表盘”，无法在智能体偏离航道时施加干预；另一类工具（如 AutoGen Studio、AGDebugger）虽然尝试引入了中间编辑与回滚机制，但往往深度绑定特定多智能体通信框架，无法适配当下主流的端到端编程与科研 Agent 运行范式。

AgentGUI 的切入点正是填补这块拼图：不仅要让长程任务的执行细节在视觉上“看得清”，还要让人类和外部评估器在运行时“管得住”。

### 分层观测：从终端杂乱信息到“像素办公室”

为了让多智能体协作具备直观的实体感，AgentGUI 引入了“虚拟工位（Desk）”的设计隐喻。在系统主面板中，各个并发运行的任务被抽象为一个像素风格的虚拟办公室，每一个独立的 Agent 都拥有专属的“办公桌”。用户发起新任务只需点击空工位、输入指令并拖入上下文文件。

更关键的技术革新落在工位内部的“四级观测解耦”。在每一个 Agent 的操作详情页中，原始日志被分流到了四个互不干扰但数据联动的视图中。

<img src="/images/2607.26300v1/features_strip.webp" alt="Per-desk views of a single run" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

首先是活动流视图（Activity feed）。该视图将原本混杂的执行轨迹解构为三类视觉元素：智能体的高层推理与总结、工具调用请求，以及环境反馈的工具执行结果。各部分采用明确的视觉区块与色彩进行区分，便于监督者在不陷入代码细节的前提下快速扫读智能体的宏观思考路径。

其次是全局时间线（Overview feed）。长程 Agent 的痛点往往在于“时间黑洞”，用户很难迅速定位 Agent 究竟是在等待耗时较长的终端编译，还是卡死在反复的模型推理重试中。该视图依据现实墙上时钟（Wall-clock time）渲染执行时间轴，清晰标记出每个子步骤的绝对与相对耗时。

第三与第四个视图则针对代码与底层开销进行了剥离。调试终端（Debug terminal）提供逐轮调用的 Token 吞吐遥测、延迟统计与 API 原始 Payload；而代码控制台（Agent console）则吸取重度编程智能体用户的反馈，专门过滤掉自然语言思考，仅展示纯净的 Shell 命令流与代码执行结果。这种剥离极大地迎合了开发者的排错直觉——当定位脚本报错时，直接查看控制台命令比在几万字的思考日志中筛选代码块要高效得多。

此外，当主 Agent 通过派生工具（Delegate tool）在后台拉起子智能体（Sub-agent）处理细分任务时，系统会在主工位旁动态生成微型可折叠头像，点击即可下钻查看子 Agent 的完全独立轨迹，解决了多层级委派过程中轨迹互相污染的问题。

### 动态接管与防漂移：给脱缰的 Agent 踩下刹车

拥有高质量的可视化仅完成了第一步，AgentGUI 的核心技术价值更在于其提供的两套引导干预（Steering）机制：即时人工干预，与基于 LLM 的自动化审计经理（Automated Manager）。

在人工干预维度，当监督者通过控制台发现 Agent 陷入逻辑循环或选错技术路线时，无需强行杀死进程，而是可以通过直接输入框向当前轮次注入修正指令。不仅如此，系统还支持“任务热更新”：用户可以在任务元数据页随时改写目标需求，Agent 会在完成当前执行步后无缝拉取新的目标约束，重新规划行动策略。

更精巧的一项设计是“运行时换脑”（Profile switching）。长程任务在不同阶段对算力与智力密度（Intelligence density）的需求并不均衡。在数据准备或大批量文件扫描阶段，用户可以配置低成本的本地开源模型执行；当任务进入极其复杂的架构设计或排错攻坚阶段，用户可以在保持原有文件系统与记忆上下文不变的前提下，一键将底座模型切换至云端顶尖闭源模型，实现性能与推理成本的最佳平衡。

<img src="/images/2607.26300v1/steering_pair.webp" alt="Manual steering and automated audit" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

然而，让人类全程守在屏幕前盯梢显然违背了委托自主 Agent 的初衷。针对无人值守场景，AgentGUI 设计了一套 LLM-powered Automated Manager 来主动遏制“任务漂移”。

长程任务最常见的失效模式并非程序报错退出，而是“隐式失焦”——随着上下文窗口不断滚动，最初的系统约束逐渐被遗忘，智能体在处理了部分子任务后便误以为整体目标已达成，提前宣布完工。自动化审计机制正是为此设计。它可以由用户手动点击触发，也可以配置为守护进程（Daemon）：当某个工位处于空闲状态且未被判定完成时自动激活。

该审计经理采用“三步验证”范式展开排查：

1. **目标分解**：首先读取原始任务需求，将其程序化拆解为一组离散、可验证的硬性判定准则；

2. **证据抓取**：自动化遍历 Agent 的历史执行记录与 Docker 工作区中的实际文件产物，提取交付证据；

3. **闭环裁决**：将每一项准则与实际证据进行比对打分，生成结构化的审计报告。

如果审计通过，该任务才会被正式打上“已解决”的标签；如果发现遗漏或产物残缺，审计经理会直接将整改建议作为外部 Prompt 注入 Agent 的上下文，强制唤醒该工位使其根据反馈继续推进，从而形成了一个自我修正的封闭回路。

### 安全沙箱与本地优先的工程设计

在系统实现层面，AgentGUI 规避了中心化 SaaS 平台常见的数据外泄顾虑与跨进程干扰隐患。

架构底层采用了强隔离的本地容器化策略。继承自 Hermes 的底层实现，AgentGUI 为每一个工位分配了一个独立的持久化 Docker 容器。这意味着 Agent 运行的所有代码、安装的依赖库以及生成的文件，不仅与用户的宿主操作系统完全物理隔离，各个 Agent 工位之间也互不串扰。

在后端通信上，系统采用本地运行的 FastAPI 框架构建中间层，每一个 Agent 的执行轮次都在独立的 Worker 进程中受控运行，并通过 WebSocket 全双工长连接将事件实时流式推送至 React 前端。

在模型支持谱系方面，AgentGUI 兼顾了绝对隐私与前沿性能。对于高度敏感的代码或医疗数据，它提供了基于 Ollama 的全本地部署通道，允许完全在用户自身的消费级硬件或私有 GPU 服务器上运转；而对于需要极高推理能力的任务，系统通过实验性集成 Claude Agent SDK，不仅支持标准的 API 密钥调用，甚至允许开发者直接接入个人现有的 Claude 订阅，绕过了高昂的商业 API 独立计费门槛。此外，整个工作区的状态快照支持一键导出与完全恢复，极大方便了团队协作中的场景复现。

### 实验实证：信息检索加速 38%，完成率大幅跃升

为了验证 AgentGUI 的实际效能，研究团队在两个维度上开展了严格的实验评估：人类用户的轨迹理解效率实验，以及针对自动化审计机制的消融性基准测试。

#### 用户研究：轨迹可理解性评估

在用户实验中，研究团队招募了 8 名来自量化分析领域的硕博研究生。实验设计采用受试者组内设计（Within-participant design），对比基准为 Hermes 官方的原生可视化工具 Hermes Dashboard（v0.16.0）。

为了避免单一样本轨迹的偶发性干扰，团队设计了两个高难度的实际科研任务作为测试载荷：一是在 OrganSMNIST 数据集上训练卷积神经网络并对抗冻结的评估器；二是为医学推理基准 MedXpertQA 迭代优化系统提示词。由 Qwen3.5-27B 驱动的 Hermes Agent 为每个任务分别跑出两批真实的执行轨迹，并围绕这些轨迹设计了包括全局状态、时间分配、产出物核验、终端动作排查以及运行调试在内的 14 至 15 道结构化问题。实验严格执行了平衡轮换策略（Counterbalancing），消除了测试者由于顺序熟悉度带来的系统性偏差。

<img src="/images/2607.26300v1/results_time.webp" alt="User-study results" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

统计检验结果表明，使用 AgentGUI 的测试者在排查相同问题时的平均耗时从基线的 145 秒大幅下降到了 90 秒，定位速度整体提升了 $38\%$（配对符号翻转置换检验，$p = 0.023$）。细分题目类型的分析揭示出，提速最显著的领域正是信息解耦发挥最大效能的地方：在“时间开销剖析”问题上平均提速 59 秒（$p = 0.008$），在“产出物核查”问题上平均提速 74 秒（$p = 0.023$）。

<img src="/images/2607.26300v1/results_accuracy.webp" alt="User-study results - Accuracy" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

更可贵的是，信息排查的提速并未以牺牲准确度为代价。相反，参与者的答题平均准确率从基线界面的 $80\%$ 上升到了 $93\%$（$p = 0.031$）。在全部五项细分问题类别中，AgentGUI 均未出现准确度下降的情况。这一对照有力地证明了，通过结构化分离推理流与执行流，能够实质性减轻监督者的认知负荷（Cognitive load）。

#### 防漂移实验：小模型能否在审计下完成复杂任务？

为了量化评估自动化审计经理在无人值守环境下的纠偏价值，团队设计了一项高度考验长期依从性的数据聚合基准。

实验模拟了一个包含 98 份合成电子病历档案的环境，要求运行在本地小参数模型上的 Agent 依次遍历档案，并针对 15 项指定的数据汇总任务输出 15 份具体的成果文件。任务的核心挑战在于：由于受托模型的参数量极小，智能体极易在处理了几个文件后发生“注意涣散”或逻辑中断，导致任务大量落空。

在评测中，基座模型选用了 Qwen3.5 家族在 0.8B 至 9B 参数阶梯上的不同规格，每个规格独立运行 $N = 50$ 次。当 Agent 自身宣布停止时，程序化评分器会首先检查 15 项产物的完整性；随后，由 Qwen3.5-27B 充当的自动化审计经理进场扫描整个工作区，向 Agent 发送审计意见并指示其继续修正，最后再次对产物进行打分。

<img src="/images/2607.26300v1/exp2.webp" alt="Effect of one manager audit on task completion" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

实验数据（Figure 6）清晰展示了引入审计机制后的断层式提升：

- 在 0.8B 极小模型上，原始的独立任务完成率极低，但经过审计经理的一次反馈唤醒后，其完成率实现了净增长；

- 随着参数量攀升至 1.5B、2B、4B 及 9B，审计带来的纠偏效果愈发惊人。在多个模型尺寸上，一次自动化审计干预使任务交付率的绝对提升值（Percentage points）最高达到了 **34 个百分点**；

- 这一结果证明，较小规模的开源端侧模型本身并非完全丧失了单步操作的能力，其主要缺陷在于维持长跨度规划时极易“提前放弃”。通过外部独立审阅者的介入，小参数模型也能在无需人工死守的情况下，稳妥达成高吞吐量的繁杂任务。

### 走向可信、可控的 Agent 基础设施

过去一年，大模型智能体的演进重心主要集中在探索工具调用边界与提升自主规划深度上。然而，正如软件工程从纯命令行逐渐演化出集成开发环境（IDE）与分布式链路追踪（Distributed tracing）一样，自主智能体在真正走向工业界关键路径时，必然要经历可观测性与控制权架构的补课。

AgentGUI 的价值正在于此。它并不是一个简单封装 API 的“外壳玩具”，而是将焦点放在了人机协作中最脆弱的环节——即人类如何在不被庞大日志淹没的前提下，保持对自主系统的信任与精准控制。其设计理念折射出未来人机协作范式的一个核心转变：人类的角色正从“亲力亲为的代码编写者”转变为“坐在总控台前的调度主管与质检专家”。

客观而言，论文所展示的原型系统仍存在一定的局限。一方面，受控用户研究的样本体量（$N = 8$）偏小且人群背景偏向技术开发者，在面对更大规模、非技术用户的泛化能力上仍需进一步观察；另一方面，目前的自动化审计逻辑主要针对应交付文件的“存在性”与量化指标，如何将这种审计拓展到开放式、高主观性任务的“定性质量把控”，仍有广阔的探索空间。

但无论如何，苏黎世联邦理工学院这项工作为社区树立了一个清晰的技术标杆：**只有当智能体的行动轨迹变得结构可读，且人类或高阶审计系统能够在运行时对其施加有约束的即时干预时，长程自主 AI 才能真正从学术玩具演变为生产力利器。** 目前，AgentGUI 已基于 MIT 协议全面开源，其本地化部署与强隔离设计，无疑为后续多智能体协同框架的工程落地提供了一块极为扎实的底座。
