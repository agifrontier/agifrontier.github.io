---
layout: default
title: "FlashRT：引导Coding Agent重写多模态部署，延迟最高降低70倍"
description: "卡耐基梅隆大学（CMU）、超威半导体（AMD）与纽约州立大学布法罗分校的研究团队提出了一套全新系统 FlashRT。该方案打破了“为人手写调度器”或“用规则编译器解决一切”的传统思路，转而将系统级优化任务交给大模型编码智能体（Coding Agent）。"
arxiv_id: "2607.18171"
paper_published: "2026-07-20"
published_at: "2026-09-06T13:15:07.886648+08:00"
topics:
  - "AI Agent"
  - "多模态&视觉"
tags:
  - "Agent harness"
  - "Chain-of-program paradigm"
  - "FlashRT"
  - "IR"
  - "Intra-model parallelism"
  - "Measurement-gated optimization loop"
related_tutorials:
  - "a-survey-on-agentic-multimodal-large-language-models"
  - "a-survey-on-multimodal-large-language-models"
  - "turbovla-real-time-vision-language-action-model-at-32-hz-on-an-rtx-4090-with-1-g"
  - "multimodal-deep-learning"
---

<p class="paper-original-title" lang="en">FlashRT: Agent Harness for Guiding Agents to Deploy Real-Time Multimodal Applications</p>

<img src="/images/2607.18171v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

大语言模型的实时推理系统在过去两年里已经高度成熟，从 vLLM、TGI 到 SGLang，业界针对自回归解码的批处理、KV 缓存管理与投机采样建立了整套标准原语。然而，当技术演进来到多模态与物理世界交互，如超低延迟语音交互智能体、实时数字人驱动以及交互式视频世界模型时，这套高度均质化的 Serving 范式迅速失效。

> ArXiv URL：https://arxiv.org/abs/2607.18171v1

多模态实时应用并非单一的 Transformer 模型，而是由自动语音识别（ASR）、大语言模型（LLM）、文本转语音（TTS）、扩散模型（Diffusion Transformer/DiT）以及变分自编码器（VAE）等多个异构模块串联而成的复杂管线。这类管线天然包含串行阻断、流式产出、跨批次依赖与显存常驻状态，其最优部署形态随着延迟和吞吐的权衡而剧烈摇摆。如果依赖当前规则固化的 Serving 框架或手工编写的多 GPU 通信代码，不仅耗费大量专家工时，面对日新月异的模型组合也极其脆弱。

卡耐基梅隆大学（CMU）、超威半导体（AMD）与纽约州立大学布法罗分校的研究团队提出了一套全新系统 FlashRT。该方案打破了“为人手写调度器”或“用规则编译器解决一切”的传统思路，转而将系统级优化任务交给大模型编码智能体（Coding Agent）。FlashRT 构建了一套严密的引导测试脚手架（Agent Harness），让通用编程 Agent 能够将开发者编写的简单单卡串行参考代码，自主升维重构成具备流式传输、模型间解耦与模型内并行的高性能多卡部署方案。在 NVIDIA B200 与 AMD MI355X 硬件上，FlashRT 实现了最高达 70 倍的端到端延迟降低与 3.6 倍的吞吐提升，甚至在 Qwen3-Omni 语音生成场景中，击败了由人类系统专家精心调优的 vLLM-Omni 实现。

<img src="/images/2607.18171v1/overview.webp" alt="多模态部署从手工调优转向 Agent 驱动的设计全景" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么现有 Serving 框架搞不定多模态管线？

多模态系统部署的本质是一个组合优化难题。假定开发者写出了一套语义正确的单卡 Python 函数，将输入音频依次传给 ASR、LLM、TTS 与数字人渲染模块。要将其搬到由多块 GPU 构成的集群中，系统工程师需要做出一连串相互耦合的决策：哪些模块可以共用显卡以规避跨设备张量传输？哪些模块应当单独拆分以形成跨批次的流水线重叠？哪些计算密集型的大模块必须在卡间做序列并行或张量并行？

现有的自动化与部署工具在这类场景中全线碰壁，核心瓶颈在于三个维度。

其一是部署策略过于僵化。类似于 vLLM-Omni 或 Cornserve 等多模态服务系统，通常将任务划分为少数几个高层阶段，并在阶段边界处施加固定的部署规则（如阶段间强制解耦、阶段内部完全共置）。然而，针对以帧率为导向的应用，模块全面拆分并行可以推高吞吐；而对于极致敏感的交互式对讲，将高频通信的模块放在同一张显卡共享显存反而能大幅压缩首字/首音频时间。固定策略根本无法兼顾截然相反的优化目标。

其二是工作负载的覆盖面有限。FlexFlow、GSPMD、Alpa 等经典自动并行编译框架在深度学习训练阶段战功赫赫，但它们的代价模型严格基于同质、稠密的静态张量计算图。面对多模态推理中不同运行时、不同框架（例如一个用 PyTorch 原生脚本，另一个跑在 vLLM 引擎中）的混合代码，编译器难以抽取统一的控制流与显存上下文。

其三是优化粒度的失配。诸如 TVM 或 TASO 等算子级编译器主要处理算子融合与代码生成，完全无法触及跨模型放置与动态流式重叠等宏观系统架构。在多模态 Serving 中，计算任务调度在数学上可规约为非抢占式多处理器调度问题（Non-preemptive Multiprocessor Scheduling），本身属于 NP-hard 问题。既不能在算子级细粒度下暴力穷举（搜索空间爆炸），又缺乏定义高层优化单元的通用规则，导致每个新应用最终都不得不依靠经验丰富的人类专家进行手写适配。

### 直接让 Agent 写多卡部署，为何必然崩塌？

大语言模型与编码 Agent 在生成单函数或修复小型 Bug 时表现优异，但如果直接将一段数百行的单卡多模态串行代码抛给通用 Agent，并要求其“输出高效的多 GPU 并行版本”，Agent 的表现往往令人失望。论文作者深入剖析了原生 Agent（Naive Agent）在未经约束下的几种典型失败模式。

<img src="/images/2607.18171v1/naive_agent_3.webp" alt="原生 Agent 面临的典型失败模式与盲目探索" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

最常见的问题是盲目聚焦于单一优化维度。以“语音输入到数字人视频输出”（Face-to-Face Conversational Agent）为例，整个链路包含 ASR、LLM、TTS 以及将音频转为头像视频的 S2V 模块。未受约束的 Agent 往往只能识别出表层的流式机会，例如将 LLM 吐出的文本流式喂给 TTS，却完全忽略了后半段 S2V 模块内部去噪扩散与 VAE 解码之间巨大的流水线重叠机会；又或者，Agent 会把全部算力资源押注在给 LLM 增加张量并行上，而对整个系统最严重的通信阻塞视而不见。

更严重的问题是正确性崩溃。在多卡重构中，各个模块经常维护着自己的持久化状态（Persistent-state Scopes），如自回归生成中的 KV Cache、流式音频的环形缓冲区、视频生成的隐变量历史。原生 Agent 在没有明确依赖图约束的情况下重写代码，经常打破变量的生命周期，导致显存泄露、异步竞态条件，甚至输出张量在数值上完全错乱。单步的 Prompting 根本不足以支撑这种跨模型、跨运行时、软硬件交织的复杂推理。

### FlashRT 的核心设计：程序链与带门控的应用级验证

针对原生 Agent 的缺陷，FlashRT 借鉴了编译器的多阶段降级思路（Progressive Lowering）与大模型的思维链机制，提出了“程序链”（Chain-of-Program）范式。这一范式不要求 Agent 一步登天，而是通过一套标准 Harness 将重构过程严格拆解为三个阶段：IR 结构化表示、静态依赖分析与测量门控的优化闭环。

<img src="/images/2607.18171v1/flashrt_agent_2.webp" alt="FlashRT 引导 Agent 进行分层规划与验证的完整流水线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### 第一阶段：构建具备流式语义的中间表示（IR）

FlashRT 强制 Agent 先通读开发者的串行参考代码，但禁止其立即写多卡代码，而是要求其将应用抽象为一个有向无环图形式的中间表示（IR）。在这一 IR 中，计算被封装为自包含的节点，节点间由数据流边缘相连。

尤为关键的是边缘级别的流式标注（Streaming Annotation）。Agent 必须对每一条数据边显式标记为“阻塞型”（Blocking）还是“流式型”（Streaming）。例如，在从 ASR 到 LLM 的路径上，由于 ASR 必须获取完整短语后才能给出高精度转录，该边属于 Blocking；而在 TTS 输出音频碎片到 S2V 模块之间，下游可以在上游生成数个 chunk 后立刻启动去噪，属于典型的 Streaming。这种显式的标注机制在认知上迫使 Agent 识别出所有潜在的并发边界：Streaming 边指示了跨批次流水线化的可能，而 Blocking 边则提示 Agent 两个节点更适合共置（Co-location）以缩减关键路径上的通信开销。

为了防止 Agent 在这一阶段胡乱构建 IR，FlashRT 配备了一个轻量级顺序解释器（IR Interpreter）。Agent 构建完 IR 后，解释器会按拓扑顺序直接在样例数据上执行该 IR，并与开发者的原始单卡代码进行输出对齐校验。只有通过了数值等价性校验，IR 才能进入下一步。

#### 第二阶段：基于静态工具的候选策略挖掘

拿到结构化 IR 后，Agent 会调用 FlashRT 提供的静态分析工具集。这些工具能够自动遍历图拓扑并结合各节点的显存常驻属性，系统性地筛选出两类优化机会：其一是计算无关的独立节点，适宜做模型间并发；其二是数据处于流式状态的阶段，适宜拆分到独立设备构建微流水线。

通过将复杂的拓扑解析沉淀为确定性的静态工具，FlashRT 极大减轻了 Agent 纯文本推理的幻觉风险，为后续的代码生成圈定了一个清晰、可行的变体候选空间。

#### 第三阶段：测量门控的自演化验证闭环

生成多卡候选代码只是起点，如何确保代码能跑、结果正确且真正变快？FlashRT 构建了一套完全由真实测量驱动（Measurement-gated）的迭代回路，由三个互为支撑的机制组成：

1. **真实用户交互视角的测试 Harness**：传统的编译校验多停留在单元测试，但实时多模态系统高度依赖前后端的数据流体验。Agent 会自行编写一套测试脚手架，模拟真实前端向后端的输入缓冲区灌入交互数据，并从输出缓冲区拉取渲染结果。这不仅用于逐元素（Element-wise）校验生成的文本、音频或图像是否与 Baseline 一致，更能够精确统计端到端时延分布与吞吐量。

2. **自演化变体队列（Self-evolving Variant Queue）**：由静态分析生成的候选优化策略被放入一个优先级队列中。在每一次迭代中，Agent 提取一个策略进行单卡向多卡的隔离重写，经过数值检验与 Benchmark 后，根据真实的性能测量数据动态更新队列。如果某种流式策略表现出超预期的时延降低，Agent 会在此基础上进一步衍生出“流式 + 序列并行”的复合变体，并重新排列待探索方案的优先级。

3. **严格的终止准则**：与无序尝试不同，该闭环只有在队列中所有列入的变体均被实测过，或者某些由于显存超限、框架冲突被记录下明确理由并剔除后，才会安全退出，最终在给定的硬件配额下输出一条 Pareto 最优的前沿部署集合。

### 实验评测：不仅战胜单卡基线，更超越人类专家

研究团队在配备 8 张 NVIDIA B200（每卡 180GB 显存）与 8 张 AMD MI355X（每卡 288GB 显存）的服务器上全面评估了 FlashRT 的能力。底层 Agent 统一调用 Anthropic Claude Code（模型为 Claude Opus 4.8），在全自动沙盒环境中运行。Agent 初始只获得开发者的串行单卡 Python 代码和可用的 GPU 数量，没有任何人为的并行提示。

#### 案例一：对讲虚拟人对话系统（Face-to-Face Conversational Agent）

该系统涵盖 ASR、LLM、TTS 与 S2V 四大阶段。开发者的单卡 Baseline 采用完全同步阻塞的方式运行，整个视频生成必须等语音全量跑完后才以批处理形式启动，端到端首帧输出时间极长，毫无可玩性。

FlashRT 在接收该任务后，通过变体队列展开了多轮演化：

- 第一步，Agent 识别出 TTS 与 S2V 之间的音频 Chunk 流式机会，将二者解耦到不同 GPU 上构建流式流水线；

- 第二步，针对计算最为密集的 S2V 阶段（包含 DiT 步进与 VAE 解码），Agent 发现每一步去噪循环与最终的高分辨率解码可以进一步施加流水线重叠；

- 第三步，在多卡充足配额下，Agent 进一步为 DiT 引入跨卡序列并行。

在 4 卡与 8 卡 NVIDIA B200 环境下，FlashRT 最终生成的部署架构将端到端输出延迟从单卡 Baseline 的数十秒直接压缩了约 70 倍，并维持了极为平稳的高帧率输出，使原本只能离线运行的 Python 脚本在完全无人工介入的情况下变成了可实时交互的数字人系统。

#### 案例二：挑战手写标杆 Qwen3-Omni

为了检验 FlashRT 生成的代码是否只对粗糙的 Baseline 有效，作者选择了一个高难度对照组：vLLM-Omni。这是目前社区针对 Qwen3-Omni 全双工语音模型由系统专家精心编写的手写多卡服务实现。

在 2 卡配额下，FlashRT 同样接管了 Qwen3-Omni 的部署任务。令人意外的是，FlashRT 探索出了一套比 vLLM-Omni 更加精炼的跨组件数据通道。vLLM-Omni 为了通用性，在多组件集成时引入了相对厚重的框架层 IPC 抽象；而 Agent 针对 Qwen3-Omni 具体的生成拓扑，编写了极其轻量级且专属于该任务的设备间张量传递逻辑。

在 NVIDIA B200 平台上，FlashRT 生成的部署方案相比专家调优的 vLLM-Omni，将响应延迟进一步降低了 25%，同时实时率因子（Real-Time Factor, RTF）稳定保持在 1 以下（说明音频生成速度快于播放速度，无卡顿）；而在 AMD MI355X 硬件上，这一优势被扩大到极其惊人的 65%。

#### 跨硬件泛化：AMD MI355X 上的惊喜

系统软件领域长期面临的一大痛点是跨硬件平台的优化迁移。NVIDIA 平台由于拥有极其庞大且成熟的人工优化软件栈（CUDA、TensorRT-LLM 等），性能往往被压榨得较为充分；而在新兴硬件如 AMD Instinct 系列上，缺乏足够的人类专家去为每个前沿多模态应用手写高度适配的代码。

FlashRT 在 AMD MI355X 上的表现证明了 Agent 驱动调优的独特价值。在相同的应用管线测试中，FlashRT 不仅在 MI355X 上完整复现了 B200 上的所有延迟缩减幅度，其吞吐量的提升幅度更从 B200 上的 2.8 倍跃升至 3.6 倍。

这一现象传递出一个强烈的信号：在生态尚在追赶、缺乏现成专家级优化库的硬件平台上，依赖 Coding Agent 结合底层运行时快速合成量身定制的调度代码，其工程杠杆效应甚至远超成熟平台。Agent 抹平了新兴硬件由于软件生态积累不足而产生的性能折损。

### 范式转移：多模态系统的工程分工重构

FlashRT 的意义远不止于提供了一套跑得更快的脚本，它实质上对 AI 系统工程的传统研发范式提出了新的思考。

长期以来，AI 基础设施团队的工作流被割裂在两个极端：算法研究员用 Python 写出原型的串行代码，随后由系统工程师介入，耗费数周时间将其重构为基于特定框架（如 Megatron、vLLM 或自研 C++ 引擎）的分布式多卡系统。每当算法端引入一个新的视觉编码器或换用一种新的流式机制，这套重构流程就要重来一遍，极其痛苦。

FlashRT 证明了一种全新的可能：未来的 Serving 系统或许不再是一个个由人类编写的固定二进制服务进程，而是一个以编译器思想构建的 Harness。人类算法人员只需以最自然、直观的单卡单进程方式定义清楚模块的数据流与状态生命周期，底层的 Harness 则调动编码能力足够强的 LLM，自动感知底层实际拥有的 GPU 拓扑与算力预算，通过带门控的生成、验证、压测循环，动态合成为当前应用、当前硬件唯一最优的 Serving 架构。

随着前沿 Coding Agent 在代码编写与工具调用能力的持续跃进，这种“人类负责语义定义，Agent 负责系统实现与性能下沉”的开发模式，正在成为多模态大模型走向实时化落地的关键加速器。
