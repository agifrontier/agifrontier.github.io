---
layout: default
title: "Agent Harness最新综述！ETCLOVG七层框架+170个项目总结"
description: "随着大语言模型（LLM）Agent在生产环境中的快速部署，一个反复出现的规律逐渐浮出水面：任务执行的可靠性，其实已经不再主要取决于底层的LLM模型能力，而是越来越依赖于包裹在模型外围的基础设施层——也就是所谓的“Agent执行脚手架（AgentExecutionHarness）”。"
arxiv_id: "openreview:3hXEPbG0dh"
published_at: "2026-08-19T17:41:59.674532+08:00"
topics:
  - "AI Agent"
tags:
  - "AI Agent"
  - "AI论文解读"
related_tutorials:
  - "a-comprehensive-survey-on-benchmarks-and-solutions-in-software-engineering-of-ll"
  - "from-prompts-to-contracts-harness-engineering-for-auditable-enterprise-llm-agents"
  - "llmtimesmapreduce-v3-enabling-interactive-in-depth-survey-generation-through-a-m"
  - "what-makes-a-harness-a-harness-necessary-and-sufficient-conditions-for-an-agent-harness"
---

<p class="paper-original-title" lang="en">Agent Harness Engineering: A Survey</p>

随着大语言模型（LLM）Agent 在生产环境中的快速部署，一个反复出现的规律逐渐浮出水面：任务执行的可靠性，其实已经不再主要取决于底层的 LLM 模型能力，而是越来越依赖于包裹在模型外围的基础设施层——也就是所谓的“Agent 执行脚手架（Agent Execution Harness）”。

> ArXiv URL：https://openreview.net/forum?id=3hXEPbG0dh

长期以来，学术界对 LLM Agent 的研究绝大多数都集中在模型本身：研究模型是否能进行多步规划、能否可靠地调用工具、能否检索并压缩相关记忆，或者能否与其他 Agent 协同。这种隐性假设认为，只要模型足够强大，加上足够优秀的提示词（Prompt），就能产生足够可靠的行为。然而，在一线生产环境中，从业者面临着完全不同的现实。OpenAI 曾在内部报告中指出，为了让 Codex Agent 可靠运行，一个小型团队在五个月内编写了大约一百万行内部产品代码，主要用于设计环境、约束、文档和反馈循环。Anthropic 的工程实践也得出了类似结论：有效的 Agent 应当使用简单且可审查的架构，渐进式地暴露上下文，并依赖具备可恢复能力的执行基础设施。

这就暴露出了一个明显的“从业者与学术界之间的鸿沟”。工业界深知基础设施的重要性，但缺乏一套正式的研究词汇来系统性地描述和改进它；学术界则在微观组件（如记忆、规划）上不断钻研，却鲜少系统性地研究如何将这些组件整合成一个可靠运行的系统。

针对这一鸿沟，最新发布的一篇系统综述首次提出并确立了“Agent Harness Engineering（Agent 脚手架工程）”的独立系统层地位。该研究不仅仅停留在理念呼吁，而是提出了一个全新的 **ETCLOVG 七层分类框架**，并将 170 多个开源项目映射到该框架中，揭示了当前 Agent 基础设施的生态格局、设计模式以及覆盖盲区。

### Agent 工程的三阶段演进：从提示词到脚手架

从早期的思维链（Chain-of-Thought）提示到如今的全自主 Agent，整个技术轨迹可以被理解为“从业者必须管理的工程表面积”在不断扩张。根据该综述的梳理，2022 年至 2026 年期间，Agent 领域的工程重心经历了三个连贯的演进阶段：

第一阶段是**提示工程（Prompt Engineering）**，核心是通过精心设计输入来激发模型的内在规划和推理能力；第二阶段是**上下文工程（Context Engineering）**，重点转向 RAG（检索增强生成）和记忆管理，试图通过外部知识注入来突破模型的静态认知窗口；而当前正在进入的第三阶段，则是**脚手架工程（Harness Engineering）**。

在脚手架工程阶段，开发者不再仅仅关注模型单次调用的输入输出，而是将精力投入到状态流转、隔离沙盒、工具协议、生命周期控制以及系统级可观测性上。一个极具说服力的数据是，在 Bench 2.0 评测中，仅仅通过脚手架层的优化（如增加执行约束和反馈循环），无需更换底层模型，就能实现 13.7 个百分点的绝对性能提升，相对提升幅度达到约 26%。这充分证明了外围基础设施对于提升 Agent 真实可靠性的决定性作用。

<img src="/images/openreview_3hXEPbG0dh/page_4_Figure_1.jpg" alt="Agent Harness工程重心的演进路径" style="width:85%; max-width:600px; margin:auto; display:block;">

### ETCLOVG：重新定义 Agent 基础设施的七层框架

为了系统化地描述这层包裹着 LLM 的基础设施，研究提出了 **ETCLOVG 七层分类框架**。该框架打破了以往将 Agent 仅仅视为“大脑+工具+记忆”的简单抽象，而是从现代软件工程和分布式系统的视角，重新定义了 Agent 运行所需的系统边界。

这七个核心层级分别是：

1. **执行环境（Execution Environment）**：提供物理或虚拟的计算底座，隔离 Agent 的操作边界。

2. **工具接口（Tool Interface）**：定义 Agent 与外部 API、协议交互的语义与契约。

3. **上下文管理（Context Management）**：负责跨轮次交互中的状态驻留、记忆裁剪与渐进式暴露。

4. **生命周期与编排（Lifecycle/Orchestration）**：控制 Agent 任务的启动、暂停、恢复、状态机流转以及子任务调度。

5. **可观测性（Observability）**：捕获执行轨迹，提供链路追踪和内部状态的透明度。

6. **验证（Verification）**：在运行前或运行中对 Agent 生成的操作进行评估和断言。

7. **治理（Governance）**：强制执行安全策略、权限边界和运行时的安全护栏。

<img src="/images/openreview_3hXEPbG0dh/page_5_Figure_1.jpg" alt="ETCLOVG分类框架概览" style="width:80%; max-width:300px; margin:auto; display:block;">

与之前六组件框架最大的不同在于，ETCLOVG 将**可观测性（Observability）**和**治理（Governance）**提升为了独立的架构关注点。在生产环境中，状态管理自然地从属于生命周期编排（L），而生命周期钩子（Hooks）和策略强制执行则从属于治理层（G）。这意味着，现代 Agent 基础设施的设计必须在一开始就将“如何监控它”和“如何约束它”作为一级系统需求。

基于这一框架，研究团队对公共可见的 Agent 基础设施进行了一次系统性的文献和代码库审查。通过搜索 GitHub、开源注册表、基准测试论文以及如 OpenAI、Anthropic 等头部机构的工程博客，最终收集并编码了 170 多个项目。

数据分析揭示了一个广泛但不均衡的生态系统：

在执行（E）、工具接口（T）、生命周期（L）和验证（V）层面，开源覆盖非常密集。因为无论是编码、网页浏览还是操作电脑的 Agent，在发挥基础作用前都必须要有可运行的环境和测试基准。然而，可观测性（O）和治理（G）在开源领域的覆盖则显得相当单薄，它们更多地出现在商业平台的闭源 SDK 或企业工程实践的分享中，这表明行业在“如何让 Agent 运行起来”方面已经成熟，但在“如何安全、可控地长期运维 Agent”上仍处于早期探索阶段。

<img src="/images/openreview_3hXEPbG0dh/page_6_Figure_6.jpg" alt="项目收集与筛选流程" style="width:90%; max-width:700px; margin:auto; display:block;">

### 深度剖析第一支柱：为什么沙盒（Sandbox）在 Agent 时代如此关键？

在 ETCLOVG 的七层框架中，执行环境与沙盒（Execution Environment and Sandbox）处于最基础的底层位置。在传统的云计算和多租户架构中，沙盒主要是一种安全防御手段；但在 Agent 时代，沙盒的意义被彻底重塑，它同时服务于三个不可或缺的核心目的：安全隔离、环境可复现性，以及**活跃度（Liveness）**保障。

前两者容易理解，但“活跃度”是 Agent 独有的一项新需求。一个长期运行的 Agent 需要不断试错、修改代码、执行脚本。如果缺乏独立的沙盒环境，Agent 产生的副作用（如改乱依赖树、耗尽系统句柄）很快就会导致其宿主环境崩溃，从而让整个长视野任务提前终止。因此，现代生产级 Agent 系统几乎无一例外地要求在沙盒内物理执行操作。

随着 2024 到 2026 年的技术演进，Agent 沙盒基础设施已经从少量通用运行时，分化为了七个高度针对性的垂直品类。理解这些分类，对于任何试图构建稳定 Agent 系统架构的工程师来说都至关重要：

**1. 通用托管沙盒（General-Purpose Managed Sandboxes）**

这一类平台（如 Daytona、E2B、Modal、Northflank 以及 Docker 官方新推出的 Docker Sandboxes）提供开箱即用的 API 接口，允许执行任意的 OCI 容器镜像。它们的设计共识是默认提供短期短暂（Ephemeral）的执行语义，并辅以可选的持久化会话。

值得注意的是，这一品类正在经历一次底层的技术迁移：从传统的内核容器隔离技术，大规模转向基于微虚拟机（MicroVMs，如 Firecracker）或用户态内核（如 gVisor）的强隔离方案。背后的驱动力在于，LLM 生成的代码往往包含极其不可预测的系统调用（Syscall）模式，传统的容器极易被不可控的异常调用拖垮，必须依赖更重但更安全的微虚拟机来隔离这种不确定性。

**2. 电脑使用 Agent 基础设施（Computer-Use Agent Infrastructure）**

这类沙盒代表了一种截然不同的执行范式：Agent 不再通过 API 或命令行交互，而是直接通过模拟键鼠操作和屏幕像素观察来控制图形界面。典型代表包括 Anthropic 的 Computer Use、开源的 CUA 以及 OSWorld。

这种环境通常需要在沙盒内部打包一个完整的桌面系统（如通过 Xvfb 结合窗口管理器，或干脆运行完整的虚拟机）。它的动作空间极其庞大，且完全依赖视觉对齐（Visual Grounding）。虽然启动延迟较高且难以高密度并发，但它是唯一能让 Agent 操作系统中那些不提供 API 的传统软件（如老旧 ERP 系统）的执行底座。

<img src="/images/openreview_3hXEPbG0dh/page_8_Figure_3.jpg" alt="沙盒与执行环境分类" style="width:90%; max-width:700px; margin:auto; display:block;">

**3. 代码专用沙盒（Code-Specialized Sandboxes）**

这类环境专为代码生成、测试评估和数据分析优化，而非提供通用的 Shell 访问权限。经典项目包括 Judge0、OpenAI 的 Code Interpreter 底层机制，以及 LangChain 的 WebAssembly 方案。

为了追求极致的并发吞吐量和极低的启动延迟，这类沙盒通常预装编译器和解释器，并默认采用无状态的请求级执行模式。一个显著的子趋势是向 WebAssembly（Wasm）架构迁移。Wasm 能够提供基于能力的细粒度安全控制、确定性执行，并将启动时间压缩到微秒级，代价仅仅是牺牲部分原生扩展库的支持。

**4. 框架内置运行时（Framework-Integrated Runtimes）**

这类沙盒并不作为独立产品出售，而是直接与特定的 Agent 框架深度捆绑，与框架的编排循环、工具注册表和提示词约定共同发布。例如 OpenHands 运行时，它将 Bash、IPython、Chromium 浏览器和 API 服务器打包进一个单一的 Docker 镜像中。

这里存在着典型的“大礼包（Bundle）”与“可组合（Compose）”的设计权衡。内置运行时追求开箱即用的极高覆盖率，但这不可避免地导致镜像臃肿、启动缓慢，并与单一框架产生强耦合。

**5. 浏览器评估环境（Browser Evaluation Environments）**

这类系统（如 WebArena、VisualWebArena、BrowserGym）兼具沙盒与评估台的双重属性，提供标准化的 Web 应用集群和基于 Playwright 的交互接口。与桌面级操作不同，浏览器沙盒直面最复杂的网络威胁：由于浏览器会摄入大量不可信的外部网页内容，它是研究“间接提示词注入攻击（Indirect Prompt Injection）”和 Agent 多模态红蓝对抗的天然阵地。

**6. 操作系统级权限沙盒（OS-Level Permission Sandboxes）**

它们并不追求提供一个全新的隔离镜像，而是利用操作系统底层的原语（如 Linux 上的 bubblewrap、macOS 的 Seatbelt 或 seccomp-bpf），对现有的宿主环境实施极度细粒度的文件系统和网络访问控制。

Anthropic 披露的数据极具说服力：在 Claude Code 工具中引入底层沙盒限制后，成功将权限确认弹窗减少了 84%，同时依然维持了极高的安全性。这种“只限制权限，不硬性隔离”的哲学，核心诉求是为了减少开发者的疲劳干预。因为在持续运行的长任务中，频繁弹出的权限确认本身就是一种系统活跃度的失败。

**7. 沙盒抽象层（Sandbox Abstraction Layers）**

以 SWE-ReX 和 Kubernetes SIG Apps 的 Sandbox CRD 为代表，这类组件并不实现具体的隔离技术，而是提供统一的 API 接口，允许底层沙盒后端（如 Docker、AWS Fargate、gVisor）在不修改上层 Agent 代码的情况下实现热插拔替换。

### 运行时安全挑战：大模型带来的三种特有威胁

在详细探讨了七类物理隔离机制后，综述指出，尽管底层容器隔离技术日益成熟，但 Agent 系统依然面临三种被严重放大的新型攻击面，这使得传统的安全模型显得捉襟见肘：

首先是**提示词注入（Prompt Injection）**。外部恶意输入（如被篡改的网页内容、伪造的工具响应或恶意文件）可以轻易绕过模型的认知防御，劫持 Agent 的行为，进而利用 Agent 的合法权限在沙盒内发起恶意操作。

其次是**目标不对齐（Goal Misalignment）**。这属于内生威胁，当模型为了完成某个复杂任务，可能会在推理过程中得出“突破当前沙盒限制有助于更好地完成任务”的错误子目标，从而主动尝试逃逸。

最后是**复合能力放大（Compositional Amplification）**。当 Agent 获得了跨系统的多个工具访问权限后，原本单个沙盒中的微小漏洞，可能会因为 Agent 极其强大的推理整合能力，被组合放大成跨工具链的级联灾难。目前，攻击评估工具的进展已经相对成熟，但系统性的防御架构设计仍处于严重的碎片化阶段。

### 工具接口标准化与 MCP 协议的崛起

除了底层的执行底座，**工具接口层（T）**是决定 Agent 能否顺利融入现有软件生态的另一大支柱。当前最具代表性的趋势是标准化协议的快速收敛。

模型上下文协议（Model Context Protocol, MCP）已经成为编码和企业级 Agent 领域最受瞩目的工具集成基座。它采用清晰的宿主-客户端-服务端（Host-Client-Server）架构，基于 JSON-RPC 标准，使得外部数据源、计算引擎和本地文件系统能够以标准化的方式向 LLM 暴露其能力。标准化接口的出现，使得 Agent 不再需要为每一个独立的 API 编写定制化的调用解析逻辑，极大地降低了能力接入的门槛。

<img src="/images/openreview_3hXEPbG0dh/page_13_Figure_7.jpg" alt="工具接口标准化与协议机制" style="width:90%; max-width:700px; margin:auto; display:block;">

### 结论与未来展望：基础设施层的核心博弈

将 170 多个项目映射到 ETCLOVG 框架后，不仅勾勒出了当前 Agent 基础设施的演进脉络，也揭示了系统设计中的几个永恒博弈。

最突出的博弈在于**能力与控制（Capability vs. Control）**的权衡。为了让 Agent 解决复杂问题，我们希望赋予它高保真的执行环境（如全功能的虚拟机）、无缝集成的多种工具以及跨多轮的上下文保留能力；然而，每一次能力的赋予，都伴随着不可预见的边界扩张和安全隐患。基础设施工程师必须在“微虚拟机级别的强隔离”与“系统级轻量权限控制”之间不断寻找平衡，以适应不同风险级别的任务。

另一个不可回避的问题是**成本、质量与速度的“不可能三角”（Cost–Quality–Speed Trilemma）**。在评估或训练场景下，我们需要能够在一秒钟内启动成千上万个瞬时沙盒；但在实际生产环境中，我们往往需要维持长期稳定、带有完备状态保存的庞大环境。针对这一矛盾，自建沙盒（Self-hosted）、云端托管沙盒（SaaS）以及混合部署模式（BYOC）将在未来很长一段时间内并存，各自服务于不同象限的工程需求。

总体而言，这篇综述明确释放了一个信号：LLM 模型的军备竞赛只是 Agent 发展的前半场；决定 Agent 能否真正实现大规模可靠落地的下半场，关键在于谁能构建出更坚固、更灵活、更具可观测性的脚手架（Harness）。未来的研究重点，不应再仅仅局限于让模型学会某项新技能，而是要在 ETCLOVG 框架的指导下，系统性地消除运行时崩溃、状态丢失、权限滥用等工程痛点。这不仅是一场属于研究员的范式转移，更是一场属于一线工程师的基础设施保卫战。

## 关键图表

<img src="/images/openreview_3hXEPbG0dh/page_0_Figure_5.jpg" alt="关键图表" style="width:85%; max-width:600px; margin:auto; display:block;">

<img src="/images/openreview_3hXEPbG0dh/page_3_Figure_8.jpg" alt="关键图表" style="width:85%; max-width:600px; margin:auto; display:block;">
