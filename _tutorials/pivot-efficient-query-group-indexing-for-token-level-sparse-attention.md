---
layout: default
title: "腾讯PIVOT：攻克DSA长文本检索瓶颈，索引算子提速4倍且精度无损"
description: "腾讯团队针对这一痛点提出了 PIVOT（Proxy Indexing Via One full-prefix Traversal），直接跳出以往在 Token 轴、Head 轴或 Layer 轴做局部裁剪的传统思维，首次将目光投向了长期被忽视的 Query 轴。"
arxiv_id: "2607.24593"
paper_published: "2026-07-27"
published_at: "2026-10-03T13:15:07.852414+08:00"
topics:
  - "模型优化"
  - "AI工程"
tags:
  - "DSA"
  - "PIVOT"
  - "PIVOT-Refine"
  - "PIVOT-Reuse"
  - "full-prefix traversal"
  - "group-based prefix scan"
related_tutorials:
  - "specattn-speculating-sparse-attention"
  - "trainable-log-linear-sparse-attention-for-efficient-diffusion-transformers"
  - "tree-training-accelerating-agentic-llms-training-via-shared-prefix-reuse"
  - "sekai2-from-world-exploration-to-interactive-world-modeling"
seo_title: "PIVOT: Efficient Query-Group Indexing for Token-Level Sparse Attention"
---

<p class="paper-original-title" lang="en">PIVOT: Efficient Query-Group Indexing for Token-Level Sparse Attention</p>

<img src="/images/2607.24593v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在大语言模型向数十万乃至百万级长上下文演进的进程中，全量注意力计算的二次方复杂度一直是推理落地的核心阻碍。为了撕开长文本计算的算力枷锁，以 DeepSeek-V3.2 与 GLM-5.1 为代表的生产级开源大模型普遍引入了 Token 级别的稀疏注意力机制——DeepSeek Sparse Attention（DSA）。DSA 的核心构想非常优雅：既然一个 Query 真正需要关注的上下文信息极其有限，那么只要在计算下游注意力前，先利用一个轻量化的索引器（Indexer）打分挑出最关键的 Top-$k$ 个 Token，后续昂贵的 Sparse MLA 算子就能从 $O(L^2)$ 骤降至 $O(Lk)$。

> ArXiv URL：https://arxiv.org/abs/2607.24593v1

然而，工业界的工程优化往往伴随着瓶颈的转移。下游的主注意力计算确实变轻了，但原本被视为前置轻量级步骤的索引器，却悄然演变成了新的系统性能黑洞。在长达 200K 的上下文窗口下，这个轻量索引器在 Prefill 阶段甚至会吃掉高达 81% 的端到端推理时延，Decode 阶段也占用了 41% 的耗时。腾讯团队针对这一痛点提出了 PIVOT（Proxy Indexing Via One full-prefix Traversal），直接跳出以往在 Token 轴、Head 轴或 Layer 轴做局部裁剪的传统思维，首次将目光投向了长期被忽视的 Query 轴。

<img src="/images/2607.24593v1/motivation.webp" alt="PIVOT 加速长上下文索引" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

该方案无需任何微调或重训练，以纯无损、即插即用的方式对 DSA 索引机制进行了结构性重构。在 DeepSeek-V3.2 与 GLM-5.1 两大主流基准测试中，PIVOT 成功将索引算子加速高达 4 倍，带来高达 1.6 倍的端到端推理提速，同时在超长文本基准上彻底抹平了传统稀疏优化常见的精度衰减。

### 为什么 DSA 稀疏注意力会被索引器卡住脖子？

理解 PIVOT 的核心贡献，首先需要拆解 DSA 索引器在大序列长度下的数学困局。DSA 包含两个核心组件：轻量级索引器与下游的稀疏多头潜变量注意力算子（Sparse MLA）。对于位置处于 $t$ 的 Query 和处于 $s$ 的历史 Token，索引器利用 $H^I$ 个轻量头计算二者的相关性得分：




{% raw %}$$ I_{t,s} = \sum_{j=1}^{H^I} w^I_{t,j} \ \mathrm{ReLU}\left(\mathbf{q}^I_{t,j} \cdot \mathbf{k}^I_s\right) $${% endraw %}



其中 $\mathbf{q}^I_{t,j}$ 和门控权重 $w^I_{t,j}$ 是从 Query Token $t$ 投影得到的，$\mathbf{k}^I_s$ 是从 Key Token $s$ 投影得到的。计算完当前 Query 对整个历史前缀的打分后，索引器挑选出得分最高的 $k$ 个 Token 构成索引集合 $\mathcal{T}_t = \mathrm{TopK}(I_{t,:}, k)$，随后下游的 Sparse MLA 仅在这个大小为 $k$ 的精简集合上执行注意力计算。

从公式可以看出，当下游注意力享受着 $O(Lk)$ 的低复杂度红利时，前端索引器为了挑出这 $k$ 个 Token，依然必须让序列中的每一个 Query 和其之前的所有 Key 进行完整的点积匹配与打分。当序列长度为 $L$ 时，单层索引器的总复杂度依然严格受制于 $O(L^2)$。随着上下文长度成倍膨胀，这部分二次方开销的边际增长率极高。此前业界尝试过多种缓解手段：例如 HISA 尝试构建块级到 Token 级的两级分层检索（Token 轴），MISA 将索引头当作混合专家系统并根据 Query 动态激活子集（Head 轴），IndexCache 则试图直接复用相邻层的检索结果（Layer 轴）。

但这三条技术路线始终默认了一个前提：每个 Query 都必须独立发起一次针对完整前缀的扫描。即便把单次调用的计算量压缩得再轻，调用的总次数依然随着 Query 数量线性增加，全前缀扫描的总次数没有发生本质改变。腾讯团队重新审视了这一过程，提出一个根本性的疑问：相邻的 Query 在寻找关键上下文时，真的需要各自独立、互不相干地把整个前缀从头到尾扫一遍吗？

### 关键实验观察：Query 轴隐藏的巨大冗余

为了验证 Query 之间的相关性，研究人员深入剖析了 DeepSeek-V3.2 在 128K 超长上下文、检索预算 $k=2048$ 时的内部打分行为，发现了三个极为关键且稳定的经验事实。

第一个事实是**局部 Query 的检索结果高度重合**。分析序列中相邻位置的 Top-$k$ 集合发现，相邻两个 Query 所选取的 Top-$k$ Token 存在极为明显的交集。无论是在模型的浅层、中层还是深层，相邻 Query 之间的 Top-$k$ 重合率中位数普遍保持在 65% 到 78% 之间，部分层甚至达到 80% 到 90%。哪怕将范围扩大到一个包含 4 个相邻 Query 的小分组，彼此之间的重合比例依然稳定在 60% 至 80%。这种现象源于因果自回归模型底层的局部平滑性：相邻 Query 共享绝大部分相同的前缀上下文，隐层表征的变化是渐进且连续的，投射出的索引 Query 向量自然高度接近，导致它们所感知的关键历史信息几乎一致。

第二个事实是**一组 Query 的 Top-$k$ 并集增长极其缓慢**。如果一个由 $g$ 个 Query 组成的小组内部完全互斥，其所需的候选 Token 总数理论上最高可达 $g \times k$；如果完全一致，则仅需 $k$。实测统计表明，当组大小 $g=4$ 时，整组所有 Query 所挑选出的 Top-$k$ Token 的实际并集大小仅为 $1.3k$ 到 $1.5k$；即使将组大小扩大至 $g=16$，并集大小也仅仅扩展到 $1.7k$ 到 $2.4k$。这意味着一整组 Query 关注的上下文核心极其收敛，完全不需要为每个成员准备完全独立的候选空间。

第三个事实是**索引器得分沿 Key 轴呈现极其显著的长尾分布**。由于 DSA 索引器在头维度打分时引入了单侧抑制的 $\mathrm{ReLU}$ 激活函数，点积结果为负的交互全部被置为零。只有那些在多个轻量头上均能形成强正向对齐的 Key，其累加得分才会显著大于零。这一特性促使 Key 的得分分布产生了极端的分化：少数关键 Token 迅速累积了绝大多数的打分权重，而绝大多数前缀 Token 的得分在截断后迅速衰减至零。累积质量曲线在极早期便迅速抬升并进入平缓期，意味着一个略大于 $k$ 的候选集合，就足以包揽全前缀扫描中绝大多数真正重要的 Token。

这三项事实直接奠定了 PIVOT 的理论基础：既然相邻 Query 的目标高度重合，且并集规模远小于独立累加的上限，那么完全可以将一组 Query 聚合成单一的“代理 Query”，只执行一次完整的前缀遍历，从而打破每个 Query 各自扫描前缀的固有架构。

### PIVOT 的核心架构：一次遍历与双重变体

PIVOT 的全称即为“基于单次全前缀遍历的代理索引”（Proxy Indexing Via One full-prefix Traversal）。它完全保留了 DSA 既有的输入输出接口，下游的 Sparse MLA 算子和底层的 KV Cache 完全感受不到任何变动。整个执行流被清晰地划分为两步：粗粒度的全局共享扫描，以及细粒度的组内 Top-$k$ 派生。

<img src="/images/2607.24593v1/main.webp" alt="PIVOT 架构设计" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### 代理 Query 构建与共享全前缀扫描

对于任意一个已划定好的 Query 组 $G$，PIVOT 首先将其内部所有成员的索引 Query 向量和门控权重按头维度计算均值，聚合出一个兼具组内代表性的代理 Query：




{% raw %}$$ \bar{\mathbf{q}}^I_j = \frac{1}{g} \sum_{t' \in G} \mathbf{q}^I_{t', j}, \qquad \bar{w}^I_j = \frac{1}{g} \sum_{t' \in G} w^I_{t', j} $${% endraw %}



随后，这个代理 Query 代替全组执行唯一一次全局扫描打分：




{% raw %}$$ \bar{I}_s = \sum_{j=1}^{H^I} \bar{w}^I_j \ \mathrm{ReLU}\left(\bar{\mathbf{q}}^I_j \cdot \mathbf{k}^I_s\right), \qquad s \leq t $${% endraw %}



这里将因果截断位置设定在组内首个 Token 的位置 $t$，以严格确保因果掩码对组内所有成员均合法有效。单次扫描的计算复杂度仅为 $O(L)$，原本整组所需的 $O(gL)$ 扫描成本被直接压缩到了单倍前缀级别。在获得全局代理打分 $\{\bar{I}_s\}$ 之后，PIVOT 针对计算速度与精准度的不同权衡，提供了两个派生变体。

#### 变体一：PIVOT-Reuse

追求极限低时延的场景下，可以完全跳过细粒度重排阶段。PIVOT-Reuse 直接从代理打分向量中截取前 $k$ 个最高分对应的 Token，并将这套索引集合毫无保留地分发给组内的每一个 Query：




{% raw %}$$ \mathcal{T}_t = \mathrm{TopK}\left(\{\bar{I}_s \mid s \leq t\}, k\right), \quad \forall t \in G $${% endraw %}



由于临近 Query 的重合度本就在 80% 上下，这种直接复用策略在绝大多数任务上已经能够捕获充足的核心上下文。整组的总索引复杂度直接降至 $O(L)$，彻底消除了所有逐 Query 的计算开销，速度优势最为明显。

#### 变体二：PIVOT-Refine

在超长上下文推理或对细微逻辑依赖极强的任务中，单纯共享同一个 Top-$k$ 会抹去不同 Query 之间的个体差异。PIVOT-Refine 引入了两级检索机制：

1. **全局粗筛**：根据代理得分从全长前缀中提取出一个容量为 $c$（例如 $c = 2k = 4096$）的紧凑共享候选池 $\mathcal{C} = \mathrm{TopK}(\{\bar{I}_s \mid s \leq t\}, c)$。得益于先前的并集观测，大小仅为 $2k$ 的候选池足以完整包裹整组 $g$ 个 Query 真正需要的 Token。

2. **局部精排**：让组内的每一个 Query 回归自身的精确投影参数，只在这个极小的候选池 $\mathcal{C}$ 内执行精确的单 Query 索引打分 $I_{t,s}$，并独立选出属于自己的 $\mathcal{T}_t = \mathrm{TopK}(\{I_{t,s} \mid s \in \mathcal{C}\}, k)$。

这种设计巧妙地规避了全量扫描的高额负担：整组的计算成本从原本 DSA 的 $O(gL)$ 转变为 $O(L + gc)$。在长上下文场景下，由于候选集大小 $c \ll L$，二次方项依然被成功削减为单倍前缀遍历，而组内各 Query 又重新拿回了个性化的精确打分，在系统性能与模型精度之间达成了极致平衡。

### Prefill 与 Decode 统一范式：自然融合 MTP

PIVOT 的另一大工业级精妙之处，在于它用同一套算法内核，优雅且天然地统筹了长文本推理中截然不同的两个阶段：Prefill（预填充）与 Decode（自回归解码）。两者的唯一区别，仅仅在于组的物理构建方式。

在 Prefill 阶段，由于输入 Prompt 的所有 Token 在序列维度是天然并发可见的，PIVOT 只需要沿着序列维度，将 Query 划分为连续且固定长度为 $g$（默认取 4）的切片区间。一个大小为 $L$ 的长序列被切解为 $L/g$ 个独立组，直接将 Prefill 索引复杂度从 $O(L^2)$ 降低到了接近 $O(L^2/g)$。

而在自回归 Decode 阶段，传统推理通常是一次吐出一个 Token，单步生成似乎根本不存在所谓的“并发 Query 组”，这使得很多并发优化算法在解码阶段彻底失效。然而，现代前沿长文本模型如 DeepSeek-V3.2 与 GLM-5.1，为了提高解码吞吐量，在底层已全面集成了多 Token 预测技术（Multi-Token Prediction, MTP）。在每一个解码步中，主干模型除了生成当前位置的 Query $q_t$，挂载的 MTP 预测头还会自回归地推演投机预测未来的 $d$ 个 Query（在这两款模型中均为 $d=3$）。

这些处于同一解码时隙内并发评估的 Draft Query，天然构成了一个完美的分组：




{% raw %}$$ G^{\mathrm{D}}_t = \{q_t, \tilde{q}_{t+1}, \dots, \tilde{q}_{t+d}\}, \qquad g = d + 1 = 4 $${% endraw %}



这意味着 PIVOT 在解码阶段完全不需要打破原本的计算图，也没有引入任何跨步缓存的状态同步负担，直接“零成本”复用了 MTP 机制在单步生成中构筑的 Query 批次。PIVOT 与 MTP 形成了极其罕见的协同加速闭环：MTP 提高了步进跨度，而 PIVOT 则极大地卸载了 MTP 伴生的多 Query 索引惩罚，使两者的加速倍率形成了实质性的复合乘积。

### 实验结果与精度验证：长文本检索无损保留

研究团队在配备 NVIDIA H20 GPU 的工业级集群上，利用 vLLM 框架对 DeepSeek-V3.2 与 GLM-5.1 展开了详尽评测。评测覆盖了真实多领域长文本基准 LongBench 以及长度可控的超长上下文针状检索基准 RULER（4K 至 128K）。

在 LongBench 综合基准中，由于上下文长度尚未达到极端极限，各基准方法与全量密集 DSA 索引器之间的得分差距基本在 0.5 个百分点以内。PIVOT-Refine 在 DeepSeek-V3.2 上取得了 56.18% 的宏平均分，甚至微弱反超了全量密集 DSA 基准。细分任务表现出高度合理的结构性特征：在代码补全（Code Completion）与少样本学习（Few-Shot Learning）等具备强局部结构规律的任务上，PIVOT 展现出了非常坚挺的性能，证明代理均值 Query 能够极其精准地表征局域语义；在依赖多处跳跃推理的 Multi-Document QA 上，均值代理稍有平摊，但由于 Refine 机制的存在，精度依然得到了完整挽回。

真正的分水岭出现在 64K 到 128K 的 RULER 极限压力测试下。此时，基于 Token 轴粗暴剪枝的 HISA 算子在 128K 长度下性能发生剧烈跳水，得分相比密集 DSA 暴跌了 19 至 28 个百分点，核心原因在于块级粗筛过早地丢弃了后续长程 Query 实际依赖的细粒度关键 Token。基于 Head 轴动态稀疏的 MISA 也产生了 6 到 18 个百分点的衰退。

与此形成鲜明对比的是，PIVOT-Refine 在 4K 至 128K 的整个区间内，精度曲线全程与全量密集 DSA 保持贴合，即使在 128K 的最严苛测试下也几乎毫无精度损耗。与此同时，纯复用模式 PIVOT-Reuse 在中等长度表现稳健，但在 128K 极端超长场景下开始出现轻度下滑。这一发散趋势恰好印证了团队的设计假设：随着跨度拉长，局部 Query 的关注点差异会逐步显现，在关键长文本任务上，先粗筛再逐 Query 精排的 Refine 机制，是守住模型精度的核心防线。

如果将 PIVOT 与层间复用方案 IndexCache 叠加（+IC），系统在保持极低精度波动的状态下，还能进一步斩获复合加速增益，这充分证明了在 Query 轴上做优化与其他各维度的正交性。

### 效率剖析与消融洞察：开销从何处被剥离？

在端到端耗时与算子级加速比方面，PIVOT 展现了极强的长程优势。

随着上下文窗口的递增，由于共享全前缀扫描剥离了占据主导的二次方项，PIVOT 的索引算子加速比呈现持续上扬的态势。在 256K 超长上下文配置下，索引算子本身的执行速度被大幅提升了接近 4.8 倍，端到端实际推理延迟也实现了高达 1.6 倍的加速。对于极短序列（如 4K 级别），由于此时索引在全局开销中微不足道，而精排重新计算候选打分反而存在轻微的前置开销，PIVOT 在工程实现中设计了安全回退机制（Guardrail）：当输入长度低于设定阈值时，直接平滑退回原始的密集扫描，确保在任何部署尺度下都绝不劣于原生 DSA。

为了验证各个组件的内在合理性，消融实验提供了更具说服力的微观佐证：

其一是**代理 Query 的均值聚合优势**。对比实验表明，采用组内均值池化（Mean Pooling）合成代理向量的方案，其效果显著优于直接挑选组内第一个或最后一个 Query 作为代表。在 128K 的极端长文本下，使用端点 Query 作为代理会导致 RULER 得分骤降 10 分以上。单个 Query 只能反映局部的瞬间注意力偏置，随着序列延伸，组内成员的微小分歧会被端点代理成倍放大；而多头均值聚合恰恰抽离出了 O1 观测所证明的“公共关注核心”，表现出极高的鲁棒性。

其二是**候选池预算与组大小的收益平衡**。实验表明，组大小设定在 $g=4$、精排候选池 $c$ 设定在 $2k$（即 4096）是一个极其优异的黄金分割点。在此参数下，候选集能够捕获全组超过 95% 以上的原生注意力质量，继续堆高候选预算 $c$ 对精度的边际贡献已经趋近于零，反而会徒增精排阶段的小核计算负载。

### 总结与启示

PIVOT 的成功为大模型长上下文推理系统的设计提供了一个极具普适性的启示：长序列注意力的优化，绝不只有沿着 Token 轴删减 KV Cache、沿 Head 轴稀疏化、或是沿 Layer 轴跨层复用这几条路。因果模型中由自回归机制带来的“Query 局部渐变性”，同样孕育着极为庞大的冗余压缩空间。

更为难得的是，PIVOT 并没有停留在理论玩具阶段，而是紧扣当前大模型前沿生产架构的脉搏。它既能够无缝嵌合 DSA 这一当前被头部模型广泛采纳的稀疏范式，又极其自然地将 MTP 投机推演带来的并发 Draft Query 变废为宝，以零训练、零侵入、数学完备的方式，化解了超长文本场景下索引模块反客为主的性能死锁。随着上下文窗口继续向数百万甚至千万级别迈进，这种沿 Query 维度重构检索并发粒度的思想，势必将成为高性能长文本推理引擎不可或缺的底层支柱。
