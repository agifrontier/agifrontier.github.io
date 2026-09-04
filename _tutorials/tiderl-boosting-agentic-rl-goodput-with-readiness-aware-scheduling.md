---
layout: default
title: "清华等提出 TideRL：就绪感知破除 Agent 强化学习死等，训练吞吐提升 5.6 倍"
description: "大语言模型（LLM）正迅速从“单轮对话的文本生成器”迈向“多轮自主行动的智能体（Agent）”。在浏览网页（WebArena）、操控操作系统（OSWorld）或执行复杂代码任务时，模型需要不断输出动作指令、调用工具或系统API，并在等待外部环境反馈后将新的观察结果追加进历史上下文，开启下一轮推理。"
arxiv_id: "2608.10402"
published_at: "2026-09-04T11:26:52.370742+08:00"
topics:
  - "AI Agent"
  - "强化学习"
tags:
  - "CTB"
  - "ERS"
  - "KV cache"
  - "RA^2P"
  - "RL training goodput"
  - "TideRL"
related_tutorials:
  - "part-ii-roll-flash-accelerating-rlvr-and-agentic-training-with-asynchrony"
  - "verltool-towards-holistic-agentic-reinforcement-learning-with-tool-use"
  - "a-practitioners-guide-to-multi-turn-agentic-reinforcement-learning"
  - "coda-coordinating-the-cerebrum-and-cerebellum-for-a-dual-brain-computer-use-agen"
---

<p class="paper-original-title" lang="en">TideRL: Boosting Agentic RL Goodput with Readiness-Aware Scheduling</p>

<img src="/images/2608.10402v1/A__title.webp" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

大语言模型（LLM）正迅速从“单轮对话的文本生成器”迈向“多轮自主行动的智能体（Agent）”。在浏览网页（WebArena）、操控操作系统（OSWorld）或执行复杂代码任务时，模型需要不断输出动作指令、调用工具或系统 API，并在等待外部环境反馈后将新的观察结果追加进历史上下文，开启下一轮推理。这一多轮交互范式虽然赋予了模型解决真实世界复杂任务的能力，却给底层强化学习（RL）基础设施带来了严峻挑战。

> ArXiv URL：https://arxiv.org/abs/2608.10402v1

传统基于 PPO 或 GRPO 的大模型强化学习系统（例如开源的 VeRL 等）大多沿用单轮问答或推理数学题时的同步执行机制。在这些框架中，采样生成（Rollout）与参数更新（Train）严格交替。然而在多轮 Agent 场景下，每条轨迹与环境交互的轮次、每轮等待网络和环境响应的时间极度不均，少数“长尾交互任务”会彻底卡死全局同步屏障，导致价值连城的 GPU 集群陷入漫长的空转等待。更致命的是，即使采用异步解耦架构，传统推理引擎在面对多轮间歇性等待时，也会频繁将未完成任务的 KV 缓存（KV Cache）踢出显存，导致模型每次拿到环境新反馈后，都必须耗费巨额算力从头做前缀重计算（Prefill）。

来自清华大学、Z.AI 以及中关村实验室的研究团队针对这一痛点，提出了专为多轮 Agent 强化学习设计的就绪感知弹性调度系统 **TideRL**。

该系统抛弃了以单一硬件利用率（Raw GPU Occupancy）为核心的传统视角，转向以单位时间内真正转化为模型参数更新的有效 Token 吞吐量（Goodput）为核心优化目标。通过在任务生命周期、计算图调度和集群资源弹性这三个维度进行协同重构，TideRL 在纯文本 Agent 任务上相比同步基线实现了高达 5.6 倍的训练有效吞吐提升，在多模态 Agent 任务上将总训练时间大幅缩减了 62.2%，同时彻底化解了困扰异步强化学术界与工业界已久的长尾气泡、模型频繁切换抖动以及 KV 缓存雪崩三大架构级难题。

<img src="/images/2608.10402v1/concept_arch.webp" alt="Agent 强化学习不同执行策略的数据流与工作流对比" style="width:85%; max-width:450px; margin:auto; display:block;">

### Agent 强化学习的“算力陷阱”：三大系统级脱节

要理解 TideRL 的革新，必须先看清当前分布式强化学习系统在面对多轮交互任务时为何步履维艰。典型的 LLM 强化学习流水线包含 Rollout（环境交互生成轨迹）、Reference（参考模型前向计算基准概率以计算 KL 散度）、Reward（奖励评估）和 Actor（策略模型梯度更新）四个计算阶段。

最直接的做法是同步调度（Synchronous RL），即全量 GPU 先共同充当 Rollout Worker，待所有任务执行完毕后，集体切换为 Trainer 执行训练。这种设计在面对环境延迟高度不可预测的多轮 Agent 任务时迅速崩溃：哪怕只有 5% 的任务因为网络延迟或复杂步骤耗时较长，其余 95% 已经跑完的 GPU 算力就只能闲置等待，造成巨大的硬件浪费。

为了突破这一瓶颈，学术界近期转向了异步强化学习架构（如 StreamRL 和 AReaL）。它们将 GPU 划分为两大阵营：一部分 GPU 专门充当生产者（Rollout Worker），持续与环境交互产生轨迹并推入全局缓冲区；另一部分 GPU 则充当消费者（Trainer Worker），将 Actor 模型与 Reference 模型部署在同一物理卡上，不断从缓冲区拉取数据计算梯度并更新权重。这理论上实现了长尾任务与短任务训练的重叠。但在实际的多轮 Agent 高并发场景下，这种朴素的解耦方式暴露了三处严重的系统级脱节：

第一处脱节在于**间歇性交互导致的 KV 缓存频繁被抢占**。现有的高性能推理后端（如 vLLM 或 SGLang）均采用“请求级连续批处理”（Request-Level Continuous Batching）。在单轮场景下这非常高效，但在多轮 Agent 场景中，任务是由一系列带有时间间隔的依赖请求构成的。当 Agent 发出点击或搜索指令后，模型在等待外部系统返回结果时是静止的。推理引擎贪婪的内存管理机制会将该任务的 KV 缓存判定为空闲，并将其从显存中驱逐以服务其他并发请求。一旦环境观察返回，系统必须以灾难性的 Cache Miss 重新做全量 Prefill 重计算。随着对话轮次增加、上下文膨胀，GPU 算力大量沦为对过去历史上下文的徒劳重算。

<img src="/images/2608.10402v1/bg_kvcache_hit_rate.webp" alt="WebShop 训练单步中的 KV 缓存命中率雪崩现象" style="width:85%; max-width:600px; margin:auto; display:block;">

第二处脱节在于**参考模型推理带来的流水线停顿与显存抖动（Thrashing）**。在分布式训练中，为了防止显存溢出，Trainer 端通常需要让 Reference 模型与 Actor 模型共用 GPU 显存。现有异步系统要么采用等待全量全局批次（Global Batch）就绪后再集体训练，这让 GPU 在单步初期出现长达 80% 以上时间的闲置气泡；要么采用微批次（Micro-batch）流式消费，但这就要求 GPU 每算一个微批次，就必须在 Reference 权重和 Actor 权重之间做一次 PCIe 换入换出或上下文切换。微批次越多，模型反复换入换出的系统抖动开销越恐怖，甚至超过了实际前向计算本身。

第三处脱节在于**静态资源划分与动态系统瓶颈之间的错配**。Agent 任务的多样性导致计算瓶颈在 Rollout 端和 Trainer 端之间极速动态漂移。当遇到高度复杂的长上下文任务时，Rollout 端产出极慢，Trainer 端长时间处于无数据可吞的饥饿状态；而当遇到短步骤任务时，Rollout 端瞬间倾泻大量轨迹，形成积压，系统瓶颈瞬间转移到了 Trainer 端。现有异步框架在集群启动时就静态固定了 Rollout 与 Trainer 的 GPU 数量配比，根本无法响应这种宏观工作负载的剧烈变化。

### TideRL 的破局核心：用“就绪感知”打通端到端调度

面对上述三个相互耦合的系统难题，TideRL 给出的根本解法是：**放弃孤立优化单一模块，以“数据就绪状态”（Readiness）作为统一信号，将多轮显存管理、流水线计算图以及物理 GPU 资源分配三者深度协同联合调度**。

<img src="/images/2608.10402v1/sys_arch.webp" alt="TideRL 系统的整体架构与工作流" style="width:85%; max-width:600px; margin:auto; display:block;">

整个系统由三大核心创新构成：

1. **连续任务批处理（Continuous Task Batching, CTB）**：重塑 Rollout 端的内存生命周期管理，从请求级粗暴驱逐升级为面向 Agent 任务语义的保真调度；

2. **资源感知 Ref-Actor 流水线（$\text{RA}^2\text{P}$）**：根据数据到达的快慢与积压量，在“解耦流式”与“同置聚合”两种计算图模式间自适应切换；

3. **弹性资源伸缩（Elastic Resource Scaling, ERS）**：利用参数同步边界实现零停机开销的 GPU 身份动态迁移。

这三个组件通过统一的反馈闭环紧密锁死：CTB 决定了轨迹何时、以何种速率变为就绪状态；$\text{RA}^2\text{P}$ 向上暴露不同数据输入速率下最优的消费代价；而 ERS 则据此在 Rollout 与 Trainer 之间动态腾挪 GPU，始终将算力压向当前拖慢训练有效吞吐的那一侧。

### 显存生命周期的语义重塑：连续任务批处理（CTB）

为了彻底根除多轮交互间隙无休止的 Prefill 重计算，TideRL 在 Rollout 端设计了连续任务批处理（CTB）。CTB 的根本转变在于把每一个数据并行（DP）节点视为一个统一的 Token 预算池，以“任务（Task）”而非单轮“请求（Request）”作为最小准入与驱逐单位。

在任务准入层面，CTB 实时监控各 GPU 节点的活跃 Token 开销，只有当显存空间足以容纳任务随轮次增长预期的显存水位时，才允许新任务准入，防止突发上下文激增导致显存耗尽。当节点面临显存压力时，CTB 坚决摒弃 LRU 等忽略任务语义的盲目置换算法，而是引入了一个基于任务语义的优先级计算函数：




{% raw %}$$P(t)=\omega_{1}\cdot\mathbb{I}_{\mathrm{eval}}+\omega_{2}\cdot G_{\mathrm{completion}}+\omega_{3}\cdot\mathbb{I}_{\mathrm{active}}+\omega_{4}\cdot L_{\mathrm{context}}$${% endraw %}



这一打分机制精准切中了强化学习的算法特性：它赋予必须按时汇报的评估任务（Eval）最高权重；对已经接近完成的 GRPO 采样组（$G_{\mathrm{completion}}$）给予次高权重，因为一个采样组如果差最后一两条轨迹未完成就无法计算相对优势（Advantage），前序计算就全是无效浪费；同时，当前处于活跃交互状态的任务和已经积累了较长历史上下文（$L_{\mathrm{context}}$）的任务也会被优先保留在显存中。

更巧妙的是，当被暂停（Paused）的任务等待外部环境响应结束、系统显存通过其他任务完成而释放出配额时，CTB 会优先恢复这些暂停任务，并严格执行“工作节点亲和性（Worker Affinity）”调度，强制将它们调度回最初生成过前缀的物理 GPU 上。这一举措极大保留了基座模型原本缓存的 KV Cache，使多轮交互中的 Cache 命中率提升了 1.58 倍，从源头消除了昂贵的重复 Prefill 算力黑洞。

### 终结气泡与抖动：$\text{RA}^2\text{P}$ 的双模动态调度

当轨迹数据进入全局缓冲区后，Trainer 端如何以最小开销将其“消化”并转化为梯度？现有方案在“全量等待的大气泡”与“微批次流式的反复模型权重换入换出”之间陷入两难。TideRL 提出了 $\text{RA}^2\text{P}$（Resource-Aware Ref-Actor Pipelining），它定义了两个极具辨识度的运行时数据就绪指标：

* **启动即就绪量（Ready-at-Start, RAS）**：当前训练步刚开始时，缓冲区内已经堆积的微批次数量；

* **单批次就绪间隔（Time Per Ready Micro-batch, TPRM）**：后续新微批次由 Rollout 端生成的平均到达间隔。

根据这两个指标，$\text{RA}^2\text{P}$ 在两种互补的执行模式间自适应切换：

<img src="/images/2608.10402v1/rap_ds.webp" alt="解耦流式模式（Decoupled Mode）下的 $\text{RA}^2\text{P}$ 执行流水线" style="width:90%; max-width:700px; margin:auto; display:block;">

第一种是**解耦流式模式（Decoupled Mode）**。当 RAS 极高或 TPRM 极短（即数据供应非常充沛）时触发。该模式打破了 Actor 与 Reference 强行挤在同一张 GPU 上的传统假设，将部分卡分配给 Reference，部分卡分配给 Actor。为了让流水线彻底跑满而不产生通信等待，TideRL 对计算图做出了深度改造：它将 Actor 节点上的 Loss 计算从传统的 Forward 阶段剥离，推迟并内联到 Backward 阶段中执行。这一精巧的重组使得 Reference 的前向传播、Actor 的前向传播以及 Actor 的反向传播能够以微批次为粒度在物理隔离的 GPU 上完美交叠（Overlap），同时彻底抹平了模型换入换出的 PCIe 开销。

针对流水线最后阶段普遍存在的排空停顿（Pipeline Flush Bubble），$\text{RA}^2\text{P}$ 还采用了一种非对称批次分配策略：前期微批次尽量聚合大尺寸以打满计算吞吐，而全局批次的最后一个微批次则刻意保持在极小的 Token 规模，并将其分配给最后执行反向传播的节点，让末尾气泡被压缩到忽略不计。

第二种是**同置聚合模式（Colocated Mode）**。当 RAS 较低且 TPRM 较长（即数据到达稀疏、Rollout 处于瓶颈）时触发。此时拆分独立的卡给 Reference 会造成严重的算力闲置。因此，$\text{RA}^2\text{P}$ 让 Reference 与 Actor 重新同置于相同的 GPU 上，但它不会来一个微批次就切换一次模型，而是结合共享内存零拷贝（Zero-Copy Shared Memory）技术，对当前已就绪的若干微批次进行聚合处理，将模型权重的切换开销平摊到一整个突发批次（Burst）上，在低速数据流下依然维持极高的显存与计算效率。

### 零停机迁移：基于就绪信号的弹性资源伸缩（ERS）

有了 CTB 稳定高效的 Rollout 生成，以及 $\text{RA}^2\text{P}$ 灵活的双模消费引擎，整个系统的最后一击落在了最宏观的物理资源分配上：**弹性资源伸缩（Elastic Resource Scaling, ERS）**。

传统的分布式深度学习系统极度畏惧运行中动态调整节点分配，因为重构通信域（NCCL Group Rebuild）、暂停数据流和跨卡搬迁模型状态通常会带来长达数十秒的停顿。而 ERS 敏锐地抓住了在策略强化学习（On-Policy RL）的本质属性——每一次梯度更新步结束时，Rollout 端由于必须保证策略不过期，本来就强制要求进行一次全量模型参数广播同步（Sync Params），并伴随 Rollout 端的缓存清空。

<img src="/images/2608.10402v1/ers_arch.webp" alt="ERS 在参数同步边界隐藏模型切换延迟的机制" style="width:85%; max-width:450px; margin:auto; display:block;">

ERS 巧妙地将 GPU 角色的动态迁移动作“寄生”在这一原本就必须存在的同步窗口内：

* 当监测到缓冲区 RAS 极低且 TPRM 过长（Trainer 饿死）时，ERS 判定瓶颈在 Rollout，调度执行同置模式并将多余的 Trainer 卡降级为 Rollout 节点。在降级过程中，节点趁着 Actor 反向传播等待间隙，在后台将 Reference 模型卸载到 CPU 内存，并无缝加入下一轮参数广播，摇身一变成为新的 Rollout 节点开始生成数据；

* 反之，当数据产生过剩、Trainer 成为瓶颈时，ERS 调度切换至解耦流式模式，并将部分 Rollout 节点升级为 Reference 节点。此时被转型的节点直接丢弃旧的 Rollout 权重，在集群广播参数的时间窗口内，重叠利用主机 PCIe 带宽把 Reference 权重拉入 GPU 显存。

整个角色切换过程完全隐藏在通信关键路径之外，既没有停止全局流水线，也没有打碎分布式通信原语，真正实现了**零额外停顿开销（Zero-Overhead Elasticity）**的动态算力伸缩。

### 扎实的实验落地：吞吐暴增与收敛一致性

为了验证系统在真实复杂负载下的工业级可用性，作者团队在配备 32 张 NVIDIA H100 GPU（分为 4 个计算节点，NVLink 互联，并辅以 JuiceFS 分布式存储）的物理解耦集群上展开了全面评估。实验覆盖了包括 WebShop、OSWorld 在内的多样化高难度 Agent 任务，横跨纯文本大模型（Qwen2.5-7B）与大模态视觉 Agent（Qwen-3-VL 4B）。

<img src="/images/2608.10402v1/thrp_text.webp" alt="文本 Agent 任务上的训练吞吐量与基线对比" style="width:85%; max-width:450px; margin:auto; display:block;">

<img src="/images/2608.10402v1/perf_text.webp" alt="文本 Agent 任务上的学习曲线收敛表现" style="width:85%; max-width:600px; margin:auto; display:block;">

在纯文本 Agent 基准测试中，TideRL 展现出了统治级的效率提升。相比于传统的同步强化学习框架 VeRL，TideRL 将训练有效吞吐量（Training Goodput）提高了 1.8 至 7.0 倍，使整体训练收敛时间直接减少了 51.1%。更具说服力的是与前沿异步系统（StreamRL 和 AReaL）的对比：在相同的硬件集群上，TideRL 的吞吐量依然稳稳高出异步基准 33% 以上。

更关键的洞见在于模型的学习质量。异步强化学习最被学者诟病之处在于“策略过期”（Policy Staleness）——当生成端跑得比消费端快太多时，Trainer 消费的往往是落后好几步的陈旧轨迹，从而破坏强化学习的数学稳定性。由于 TideRL 拥有极高的全链路流转效率，绝大多数由 Rollout 产生的有用轨迹都能被 Trainer 在零延迟（Zero-Staleness）窗口内迅速消化。从收敛曲线（Figure 8）可以看出，TideRL 的奖励上升轨迹几乎与严格同步的基准重合，在享受异步流水线数倍加速的同时，完全规避了传统异步方案因策略陈旧导致的性能坍塌。





在环境交互更深、单步包含高分辨率多模态观察的 OSWorld 任务上，挑战进一步加剧。如图 10 和 11 所示，TideRL 不仅在 Best-of-N（BoN）奖励与最终任务通过率上持续超越所有异步系统，而且仅耗费了同步基线 37.8% 的物理墙钟时间（Wall-Clock Time）就完成了相同的训练步数，整体训练时间缩减达 62.2%。

消融实验进一步清晰地剥离了每一个模块的价值支撑：

* 单独引入 $\text{RA}^2\text{P}$ 流水线，消除了无节制的模型切换，使训练步耗时显著降低高达 44.3%；

* 叠加 ERS 弹性伸缩后，系统根据就绪度信号自适应腾挪卡位，吞吐量在 $\text{RA}^2\text{P}$ 基础上再次提升 35.7%，将 Rollout 与 Trainer 的双向空闲等待时间整体削减了 68.6% 至 77.6%；

* 引入 CTB 调度器后，得益于对长上下文与完整采样组的显存兜底，系统在启动初期包含密集评估负载的极限压力下，取得了 54.8% 的阶段吞吐跃升。

### 启示与未来图景

TideRL 的诞生，标志着大模型强化学习系统正从“将单轮推理引擎与单轮训练引擎简单拼装”的原始阶段，正式迈入“面向复杂交互场景的联合协同设计”时代。

它向整个社区传递了一个非常明确的系统设计哲学：**在大模型 Agent 的长程交互链路中，孤立地榨取单卡的算力利用率往往只是虚妄的假象；如果不解决请求级切分造成的 KV 缓存雪崩、不根据数据到达的离散特征动态排布流水线，算力就会大量浪费在无意义的重复 Prefill 和显存换入换出中。** 唯有将推理后端的内存调度、训练后端的计算图切分以及集群维度的 GPU 编排统一在“就绪感知”这一主线下，才能在根本上实现大模型在复杂现实世界中飞速进化所需的极致“有效吞吐”。

随着 Agent 任务从几十轮交互进一步向长达数百步的代码开发、系统运维演进，类似 TideRL 这样兼顾算法对齐窗口与底层硬件流式特性的弹性系统，必将成为下一代自主智能体基础设施的标准技术底座。
