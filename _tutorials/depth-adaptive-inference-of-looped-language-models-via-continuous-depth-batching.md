---
layout: default
title: "CDB：给循环大模型装上“变速箱”，逼近99%理论加速极限"
description: "来自帝国理工学院、慕尼黑工业大学和波茨坦大学等机构的研究团队，首次提出了全流程落地的 连续深度批处理（Continuous Depth Batching, CDB） 系统。"
arxiv_id: "2608.09444"
paper_published: "2026-08-10"
published_at: "2026-10-04T13:15:08.348415+08:00"
topics:
  - "AI理论"
tags:
  - "Boundary stages"
  - "CDB"
  - "Depth-adaptive inference"
  - "LMs"
  - "Loop-level scheduling"
  - "One-step-ahead exit decisions"
related_tutorials:
  - "cost-aware-retrieval-augmentation-reasoning-models-with-adaptive-retrieval-depth"
  - "\u03c0mathbfr2-reactive-real-time-flow-policies"
  - "lets-verify-step-by-step"
  - "kalypso-relational-llm-serving"
seo_title: "Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching"
---

<p class="paper-original-title" lang="en">Depth-adaptive Inference of Looped Language Models via Continuous Depth Batching</p>

<img src="/images/2608.09444v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

大语言模型推理解码天然受制于显存带宽瓶颈。为了在单次前向传播中摊薄权重加载的开销，业内普遍采用批处理（Batching）策略。近两年来，学术界开始关注一种独特的架构分支：循环大模型（Looped Language Models，如 Ouro、Huginn）。这类模型通过复用一组共享参数的 Transformer 核心层，对隐层状态进行多次循环迭代更新。

> ArXiv URL：https://arxiv.org/abs/2608.09444v1

循环架构最诱人的潜力，在于其支持**按 Token 自适应深度（Depth-Adaptive Inference）**：对容易预测的词，循环 1-2 次便提前退出；遇到结构复杂的词，再跑满预设的最大循环次数（如 4 次或 16 次）。这种设计在理论上可以大幅压缩无谓的浮点计算量（FLOPs）。

然而，这一理想长期停留在算法纸面上，在工程落地时遭遇了巨大阻碍。现有的高性能推理系统（如 vLLM）均基于连续批处理（Continuous Batching, CB），调度粒度卡在 Token 边界上。一旦同一个批次内的不同序列要求循环不同的次数，统一的前向传播便立刻破碎——提前退出的 Token 必须走非循环的后处理层（Coda/LM Head），而未完成的 Token 必须继续留在核心层循环。传统引擎无法在一次前向传播“内部”剔除或重排 Token，这导致自适应深度的实际部署举步维艰。

来自帝国理工学院、慕尼黑工业大学和波茨坦大学等机构的研究团队，首次提出了全流程落地的**连续深度批处理（Continuous Depth Batching, CDB）**系统。通过在单次循环迭代的微观粒度上重构推理流水线，CDB 在 Ouro 1.4B 与 Huginn 3.5B 上实现了高达 **99% 的理论最大 FLOPs 加速上限**，在离线吞吐上斩获 **1.5 至 1.9 倍的真实提升**，同时在动态在线服务负载下将归一化延迟降低了 **45% 至 90%**。

### 为什么循环模型的动态深度难以批处理？

要理解连续深度批处理的突破，需要先拆解循环大模型的内部结构。典型的循环架构并非 100% 的循环层，而是由三段构成：

1. **前奏阶段（Prelude, $P_{\theta}$）**：非循环层，包括输入 Embedding，有时也包含若干固定 Transformer 层。

2. **循环核心（Recurrent Core, $R_{\theta}$）**：参数共享的重复块，负责多次迭代提炼隐层表征。

3. **尾奏阶段（Coda, $C_{\theta}$）**：非循环层，通常包含最终的归一化与 LM Head 输出层，用于采样下一个 Token。

在前向计算中，一个 Token 的路径表现为：




{% raw %}$$ h^{(0)}_{t}=P_{\theta}(x_{t})\;,\;h^{(r+1)}_{t}=R_{\theta}\!\left(h^{(r)}_{t},\mathcal{K}^{(r)}_{<t}\right)\;,\;p(x_{t+1}\mid x_{\leq t})=\mathrm{softmax}\!\left(C_{\theta}(h^{(r_{t}\leq r_{\max})}_{t})\right) $${% endraw %}



这种分段逻辑引发了三个严苛的系统级矛盾：

首先是**阶段异构性（Stage Heterogeneity）**。连续批处理假设一个 Step 内所有序列执行相同的网络层，但循环模型中，当 Token A 在第 2 次循环提前退出时，它必须进入 Coda 采样下一个词；而此时 Token B 正在经历第 3 次循环。两者的操作完全不同、执行频率各异、批大小动态分流。

其次是**关键路径上的退出判定（Critical Path Exits）**。早期退出门控（Early-exit Gate）只有在当前循环步 $r$ 执行完毕后才能得出判断，而这一判断决定了谁有资格进入步 $r+1$。如果 CPU 调度器在每一步后都要等待 GPU 返回结果并重新打包张量，GPU 算力将长时间陷入空转。

最后是**参差不齐的 KV Cache（Ragged KV）**。循环模型在核心层迭代时，每一层循环步都会产生对应的 Key-Value 缓存。当 Token A 在步长 $r=1$ 退出时，它在更深步数（如 $r=2, 3$）处留下了空白。后续依赖多步循环的 Token 在做跨步注意力计算时，就会面临缓存槽位断裂的困境。

### 连续深度批处理的系统架构

针对上述难点，CDB 抛弃了“以 Token 为中心”的调度范式，下沉至“以单步循环（Loop Iteration）为中心”的微观粒度，设计了完整的软硬件协同方案。

#### 1. 解耦的多队列调度器与优先级规则

CDB 将系统的生成路径彻底解耦为四个独立的优先级队列：等待 Prefill 的请求队列、等待 Prelude 的解码队列、等待下一步循环核心的 Recurrent 队列，以及退出循环等待 Coda 的解码队列。调度器在每个 Tick 时刻从队列中抓取就绪 Token 进行打包。

由于各个阶段被剥离为独立的 GPU Kernel 调用，调度顺序变得极其灵活。为了兼顾推理延迟与整体吞吐，CDB 确立了严格的调度优先级：

1. **最高优先级赋予 Coda**：尽早产出已完成 Token，推进整体序列闭环并释放显存。针对 Coda 包含较多独立 Transformer 层的情况，系统引入最小 Coda 批大小阈值，避免为了单个 Token 单独触发昂贵的尾部计算。

2. **次高优先级赋予 Prefill**：在显存容量允许的上限内优先摄入新请求，为主循环填充足够多的并发 Token。

3. **主干计算交替运行 Prelude 与 Recurrent Core**：在前奏处理完成后直接将数据无缝衔接至循环核心。

系统支持两种模式：**无填充（No-refill）**模式下，提早退出的 Token 仅等待整批序列完成再统一跑 Coda，批次逐步缩减；而**填充（Refill）**模式则在某个 Token 提早退出的瞬间，将队列中新的 Token 填入空出的核心运算槽位中，始终维持高饱和度的计算密度。

#### 2. 前瞻门控消除设备空闲

为了斩断退出判定阻塞 GPU 计算图的死锁，CDB 引入了**前瞻门控（Lookahead Gate）**机制。

常规的退出门控是在步 $r$ 结束后，判断该 Token 是否需要做步 $r+1$。CDB 将判断逻辑前移：步 $r$ 输出的状态直接用于预测该 Token 是否参与步 $r+2$。这种“隔步决策”赋予了 CPU 宿主机整整一个循环步的时间窗口，在 GPU 执行步 $r$ 时，CPU 就能在后台异步构建步 $r+1$ 的批处理张量，并利用 CUDA Graph 投递执行。

这一设计将 Token 的最少执行步数限制为 $r_{\min}=2$，但换来的是几乎完全被掩盖的调度开销。在未引入静态 Shape 与前瞻门控时，GPU 的单步空闲率高达 40%；在全套异步流与前瞻机制加持下，GPU 设备空闲率被直接压缩至 **0.67%**，调度开销几乎归零。

#### 3. 共享 KV Cache 解决空槽难题

面对非均匀深度造成的 KV Cache 碎片问题，团队对比了两种方案。一种是“最后退出缓存（Last-exited Cache）”，即把退出步的状态强制复制到后续所有未跑步数的槽位中。这种方案不仅浪费了 $r_{\max}$ 倍的显存，还引入了大量的张量搬运开销。

最终 CDB 采用了更具工程优势的**共享缓存（Shared Cache）**策略：为每一层循环只分配单一的 KV Cache 槽位。每当 Token 进行下一次循环时，直接原地覆写（Overwrite）该槽位，将表征逐步提炼细化。无论先前生成的 Token 是在第几步退出的，后续 Token 都能统一访问到前序 Token 最新、最成熟的隐层状态。这彻底抹平了断裂槽位的问题，使显存占用降低至原先的 $1/r_{\max}$，并且天然兼容 PagedAttention 等前沿注意力算子。

### 理论上限与硬件 Roofline 分析

为了科学衡量加速效率，本文建立了严谨的算力消耗边界。设单 Token 执行一次循环核心的 FLOPs 为 $F_r$，执行前后非循环边界（Prelude + Coda）的 FLOPs 为 $F_0$。在平均退出深度为 $\bar{d}$、最大深度为 $r_{\max}$ 时，自适应深度相比固定深度的理论算力加速上限为：




{% raw %}$$ \frac{F_0 + r_{\max} F_r}{F_0 + \bar{d} F_r} \leq \frac{r_{\max}}{\bar{d}} \leq \frac{r_{\max}}{r_{\min}} $${% endraw %}



公式直观揭示了模型架构对推理提速的制约关系：

- 对于像 **Ouro 1.4B** 这样的模型，其非循环边界极其轻量（仅有 Embedding 与 LM Head，$F_0 \ll F_r$），$r_{\max}=4$，单步提前退出就能砍下接近 25% 的整词计算量，加速比极限紧贴 $r_{\max}/\bar{d}$。

- 而对于 **Huginn 3.5B**，其架构为 2-4-2 的切分方式（前奏 2 层、核心 4 层、尾奏 2 层），非循环边界层占据了较大比重（$F_0 \approx 1.2 F_r$），最大循环次数 $r_{\max}=16$。单步退出节省的计算占比相对稀释，且较重的 Coda 会频繁打断核心流水线。

结合硬件的 Roofline 延迟模型，可以进一步看清 Refill 模式的价值空间：在小 Batch 的**显存受限区（Memory-bound Regime）**，模型加载权重的耗时占据主导，单纯缩减批次大小（No-refill）并不能减少实际物理耗时。此时只有通过 Refill 将空闲槽位填满新 Token，才能在相同的物理延迟内消化更多计算。而在大 Batch 的**算力受限区（Compute-bound Regime）**，计算延迟与总工作量严格线性相关，无论是否补充新槽位，减少计算量都能直接转化成耗时下降。

### 实验结果与性能表现

研究团队在配备单卡 NVIDIA H100 80GB 的环境中，基于 Alpaca（短指令）和 ShareGPT（长上下文交互）两大基准数据集，对比了基准连续批处理（标准 CB，固定跑满深度）、无填充的 CDB（No-refill）以及全功能 CDB（Refill）。

#### 离线吞吐：压榨出 99% 的理论红利

在所有请求并发提交的离线场景下，CDB 展现了极强的吞吐释放能力。

对于架构轻盈的 Ouro 1.4B，理论 FLOPs 加速上限约为 $1.6\times$。实验表明，CDB (Refill) 在整个 Batch Size 扫描区间内均保持极其强势的表现，**达到了理论加速上限的 94% 至 99%**，获得了 $1.5\times$ 至 $1.58\times$ 的绝对端到端吞吐提升。即便在不触发任何提前退出（Full Depth）的极端对照组中，CDB 相对原生 CB 的吞吐损耗也控制在 2% 以内，证实其调度基础设施的高效与极低额外开销。

对于拥有厚重边界层的 Huginn 3.5B，虽然受限于 Coda 处理延迟，CDB 依然实现了 **$1.9\times$ 的吞吐跃升**，达到了理论上限的 73% 至 92%。对比实验还印证了硬件模型的预测：在 Batch 较小时，Refill 相对 No-refill 优势显著；而在大 Batch 饱和状态下，两者性能趋近一致。

#### 在线服务：延迟全线大幅收窄

在模拟真实用户动态请求到达的在线服务测试中，CDB 同样优势明显。在不同请求速率（Request Rate）下，CDB 将归一化服务延迟（Normalized Latency）降低了 **45% 到 90%**。

在高提前退出比例的工况下，由于较容易的 Token 迅速腾出显存与计算资源，整个系统的排队积压现象被有效抑制。尤其是在以长文本为代表的 ShareGPT 测试中，持续解码的 Token 储备更为充裕，CDB 的插槽补充流水线几乎从未停转，系统始终稳定运行在极低延迟区间。

### 循环模型与推理系统的协同设计

这篇研究给大模型架构设计带来了极具实践价值的反思。

首先，**非循环边界层的体量直接决定了自适应深度的落地成色**。近期的模型设计倾向于在循环核心外增加非共享层以提升训练指标，但这直接拖累了推理系统的调度流畅度。边界层越重，Token 退出的调度代价越高，系统不得不引入积攒批次的策略，进而反噬吞吐。未来面向高效推理的循环模型，应尽可能压缩甚至完全去除外部非循环层。

其次，**自适应深度在本质上是测试期计算（Test-time Compute）在微观层面的投影**。推理思维链模型通过增减思考 Token 的数量来适配问题难度，而循环模型则通过增减单 Token 的网络深度实现算力分配。两者的技术逻辑互为表里。

CDB 的出现填补了循环模型在高效服务层面的拼图，证明了只要调度细粒度到位，动态深度的浮点节省就能真实变现为线下的吞吐飞跃与线上的毫秒级响应。随着大模型体系向端侧部署与超高并发推理解码演进，这一机制或将成为循环架构走向工程实用的底层基石。
