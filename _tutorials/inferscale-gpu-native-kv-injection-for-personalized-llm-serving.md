---
layout: default
title: "InferScale：跳过文本预填充，首字延迟降低79%的个性化大模型推理"
description: "针对这一系统瓶颈，东北大学（Northeastern University）的研究团队提出了 InferScale 。InferScale 抛弃了在文本层（Token Layer）机械拼接记忆的做法，改为直接在注意力层（Attention Layer）进行原生 KV 注入（KV Injection）。"
arxiv_id: "2607.27090"
paper_published: "2026-07-29"
published_at: "2026-09-26T13:15:07.411720+08:00"
topics:
  - "推理"
  - "AI工程"
tags:
  - "Chunked RoPE"
  - "Context-Window Encoding"
  - "GPU-native KV injection"
  - "InferScale"
  - "Reusable KV state"
  - "RoPE"
related_tutorials:
  - "specbox-speculative-sandbox-scheduling-for-efficient-llm-agent-serving"
  - "toktier-exact-stateful-tokenization-for-agentic-llm-serving"
  - "improving-context-fidelity-via-native-retrieval-augmented-reasoning"
  - "kv-skill-forging-expertise-in-the-models-native-language"
seo_title: "InferScale：跳过文本预填充，首字延迟降低79%的个性化大模型推理"
---

<p class="paper-original-title" lang="en">InferScale: GPU-Native KV Injection for Personalized LLM Serving</p>

<img src="/images/2607.27090v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在大模型走向个性化助手、长程伴侣和复杂智能体的落地过程中，系统往往需要为每个用户维护一个持续增长的个性化记忆库。现有工业界主流方案（例如 Mem0、Zep、Letta/MemGPT 等）普遍采用“检索-拼接”范式：先从用户的历史记忆中检索出前 $k$ 条相关的记忆事实（Facts），然后将其序列化为自然语言文本，拼接到当前用户 Query 的 Prompt 前方，送入大模型引擎执行 Prefill。

> ArXiv URL：https://arxiv.org/abs/2607.27090v1

这种在 Token 层面做 Prompt 注入的做法，正逐渐成为生产环境中的严重瓶颈。随着业务希望模型记忆更加全面、检索量（Top-$k$）逐渐放大，服务引擎每收到一次用户请求，哪怕检索出来的记忆事实与上一轮完全一致，也必须将这批几千甚至上万 Token 的记忆文本重新计算一遍 Key-Value（KV）投影与注意力矩阵。这一过程导致首字延迟（TTFT, Time-to-First-Token）随着检索量线性甚至二次方膨胀，原本高复用率的静态记忆成了每次计算都要重新支付的“税收”。

针对这一系统瓶颈，东北大学（Northeastern University）的研究团队提出了 **InferScale**。InferScale 抛弃了在文本层（Token Layer）机械拼接记忆的做法，改为直接在注意力层（Attention Layer）进行原生 KV 注入（KV Injection）。该方案无需修改模型权重，也无需对大模型服务引擎进行侵入式魔改，通过 vLLM 标准的 KV-connector 插件即可无缝部署。在 LoCoMo 基准评测中，当检索预算 $k=50$ 时，InferScale 将 TTFT 降低了 72% 至 79%（实现 3.6 到 4.8 倍加速），并在高并发场景下达成 3.7 到 4.5 倍的系统吞吐量。

### 为什么现有方案无法有效复用记忆？

在大模型推理架构中，Prefill 阶段针对输入序列做自注意力计算，具有 $\mathcal{O}(n^2)$ 的计算复杂度。现有的 KV 复用技术（例如 vLLM 引领的 Prefix Caching）之所以难以直接解决个性化记忆的 Prefill 成本，根本原因在于其**前缀连续性假设**。

Prefix Caching 要求复用的 Prompt 必须以完全相同的内容与位置出现在最前端。但在真实的个性化检索系统中，针对不同的用户 Query，检索算法召回的 Top-$k$ 记忆子集是动态组合的：记忆条目的组合方式、排列顺序在每次请求中都不尽相同。一旦顺序或组合发生改变，传统前缀缓存直接失效，引擎只能被迫从头执行完整的 Prefill。

从计算因果律来看，这种重复 Prefill 是极其冗余的。在因果解码器（Causal Decoder）中，静态记忆事实自身的 Key 与 Value 状态仅由其自身 Token 决定，完全独立于后续到达的用户 Query。如果把记忆 Tokens 记为 $m$，Query Tokens 记为 $q$，传统 Prompt 注入每次请求需支付 $\mathcal{O}((m+q)^2)$ 的注意力开销；而如果能将这 $m$ 个记忆的 KV 状态直接固定并注入缓存，每次请求的注意力计算将直接缩减为 $\mathcal{O}(q(m+q))$，彻底抹掉那笔重复支付的 $\mathcal{O}(m^2)$ 计算税。

### 从 Prompt 注入到 KV 注入：核心机制解析

InferScale 的核心思想是实现从“检索文本再 Prefill”到“检索 KV 并直接注入”的转变。为了让这一构想在工业级推理框架中成立，设计者必须克服位置编码漂移、上下文关联损失以及显存层级管理三道难题。

#### 1. Chunked RoPE：解决动态组合下的位置编码漂移

现代开源大模型（如 Llama 系列）广泛采用旋转位置编码（RoPE）。RoPE 的机制是在生成 Key 向量后，根据 Token 所在的绝对位置 $p$ 对其应用正交旋转矩阵 $R_p$。如果直接将离线预计算好的 KV 存下来，一旦在在线阶段这批事实被重新排列并组装到新的相对位置上，其携带的旋转角度就会错位，导致自注意力计算完全失真。

InferScale 提出了 **Chunked RoPE** 机制。其核心洞察在于：旋转操作发生在 Key 投影线性变换 $W_K h$ 之后，因此可以在离线生成记忆时截获并缓存未施加旋转矩阵的原始 Key（Pre-RoPE Keys）以及原始 Value。

当在线检索出 Top-$k$ 条记忆后，系统将这些记忆块逻辑拼接为长度为 $m$ 的虚拟上下文，紧接着在内存中根据它们最终在请求中分配的虚拟连续位置（从 $0$ 到 $m-1$），调用模型自身的旋转表，动态在 GPU 内部施加对应的旋转矩阵 $R_p$。研究在数学上严格证明了其等价性：通过 Pre-RoPE 存储结合装配时在线旋转，得到的 KV 矩阵在后续 Query 视角的注意力计算中，与将整段文本一次性传入模型执行原生 Prefill 产生完全相同的隐藏状态与输出概率分布。

#### 2. Context-Window Encoding：弥补跨记忆注意力的语义稀疏

将记忆切分为独立 Fact 并分别进行离线 KV 编码，虽然赋予了记忆极高的组装灵活性，但带来了一个隐性代价：牺牲了跨 Fact 的上下文交互。

在常规的长 Prompt Prefill 中，位于后方的文本可以通过自注意力“看到”前面的背景上下文，从而消除代词歧义或补全语义指代。如果每个 Fact 都在完全孤立的单句环境下独立编码，生成的 KV 缺乏对话语境，一旦在推理期被强行拼接，模型的检索回答准确率会出现可感知的退化。

InferScale 设计了 **Context-Window Encoding** 来填补这一差距。在离线构建记忆库时，系统在抽取特定 Fact 后，不会只将该 Fact 单独送入模型，而是将其连同它在原始对话中前面的 $w$ 轮上下文窗口一起送入模型执行前向传播。而在获取该层的隐藏状态与投影时，系统**仅提取并存储目标 Fact 对应的未旋转 KV 张量**，抛弃前面的上下文 KV。这种做法既保留了 Fact 内部对前序对话语境的语义感知，又让最终产出的 KV 单元保持精简与解耦，在不需要在线重新计算或微调模型的前提下，抹平了独立编码导致的准确率折损。

#### 3. GPU 原生双索引协同与零引擎侵入架构

在系统实现层面，InferScale 构建了一个紧密贴合硬件的端到端检索与推理流水线。每个记忆单元具有全局唯一的 `fact_id`，并拆分为两个 GPU 原生索引：

- **检索空间**：基于轻量语义向量模型计算 Embeddings，载入 GPU 原生图近似最近邻索引（Jasper），负责高效的 Top-$k$ 向量检索。

- **注入空间**：保存对应 Fact 的 Pre-RoPE KV 张量，驻留在 GPU 存储池（或主机内存）中，负责生成推理所需的状态缓存。

当用户请求到达时，由于向量检索与 KV 装配全部在 GPU 设备端闭环，检索到的记忆内容完全不需要走慢速的 PCIe 总线向 CPU 往返传输，唯有短小的 Query Tokens 需要从客户端送达 GPU。随后，InferScale 借助 vLLM 官方提供的 `KVConnectorBase_V1` 扩展接口接入推理运行时：在调度层拦截请求并声明该前缀已“外部就绪”，避免引擎为记忆分配 Prefill 计算；在计算前通过单卡内部的高速 GPU-to-GPU 内存散布复制（Scatter Copy），把拼接并应用了 Chunked RoPE 的 KV 张量直接填入 PagedAttention 的 Block 槽位中。整套机制没有改动一行 vLLM 内部核心代码，保持了对现代服务引擎生态的高度兼容。

### 实验结果与性能评估

作者在个性化长上下文基准数据集 LoCoMo 上，选取了以 Llama-3.1-8B-Instruct 为代表的开源模型，将 InferScale 与工业级方案 Mem0 进行了系统维度的对比。

评测显示出的最显著特征是：**InferScale 成功将引擎的首字延迟与检索记忆规模解耦**。

在传统 Mem0 架构下，随着检索数量从 $k=5$ 增加到 $k=50$，需要注入的文本大幅增加，Llama-3.1-8B 的引擎 TTFT 从 33.2 毫秒暴增 106% 至 68.3 毫秒。而采用 InferScale 后，由于这 50 条记忆对应的 KV 已经被直接复制进显存，GPU 只需要对短短的 Query 执行 Prefill，其 TTFT 仅从 16.6 毫秒微幅浮动到 17.3 毫秒（变化率仅 4%）。在 $k=50$ 的重度记忆依赖场景下，InferScale 取得了 72% 至 79% 的 TTFT 缩减，换算成加速比即为 3.6 到 4.8 倍。

在回答准确率方面，由于 Context-Window Encoding 充分补偿了独立编码带来的上下文损失，InferScale 在 $k=50$ 时达到了 60.3% 的准确率，与完整执行 Prompt 文本重算注入的 Mem0（63.3%）仅有微小的差距，而在中小检索预算下其准确率与基线基本持平。

更为关键的是高并发承载能力。在高负载服务环境下，反复 Prefill 大量记忆文本会迅速榨干 GPU 的计算单元，引发严重的请求排队。InferScale 将绝大多数计算转换为显存带宽友好的轻量张量搬运，在 100 个并发用户的压力测试中展现出近乎线性的吞吐扩展性，最终并发处理吞吐量达到了 Mem0 的 3.7 到 4.5 倍。针对大显存占用的顾虑，研究团队还验证了分层卸载方案：将数个 GB 的个性化 KV 存储在主机内存（Host DRAM）中，检索后按需经 PCIe 流式传入 GPU，在 $k=50$ 下端到端延迟仅增加约 3 毫秒，依然比 Mem0 快 2.3 倍，大幅拓宽了单机能够承载的用户记忆规模。

### 总结与展望

InferScale 的技术路径表明，对于强依赖历史记忆、知识图谱切片或长程个性化配置的大模型应用，**基于 Token 的文本拼接正在逼近效率极限，而基于 KV Cache 的语义状态注入才是高吞吐、低延迟服务的演进方向**。

通过 Chunked RoPE 解决位置多变性，利用 Context-Window Encoding 守护上下文质量，再通过 GPU 原生索引闭环消除总线开销，InferScale 在理论精确性与系统工程落地之间找到了一个兼顾两者的平衡点。这一范式不仅为个性化 Agent 和长程对话系统的工业落地提供了极具性价比的工程模版，也为未来多模态上下文缓存、动态外部知识注入等方向的系统级协同优化打开了新的思路。
