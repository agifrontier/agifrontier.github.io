---
layout: default
title: "Kimi K3：混合注意力与稳定LatentMoE，扩展效率提升2.5倍"
description: "在大语言模型（LLM）的发展进程中，扩展定律（ScalingLaws）长期主导着预训练阶段的资源投入。然而，随着推理模型和强化学习在测试时计算（Test-timecomputation）上的突破，行业逐渐演化出两条平行的扩展轴线：一是继续扩大预训练基础模型的参数规模，二是在测试阶段通过强化学习、多步骤推理和长。"
arxiv_id: "2607.24653"
published_at: "2026-08-20T13:15:22.144485+08:00"
topics:
  - "模型优化"
tags:
  - "1M-token context window"
  - "2.8T-parameter model"
  - "Agentic RL"
  - "Attention Residuals"
  - "KDA"
  - "Kimi K3"
related_tutorials:
  - "kimi-linear-an-expressive-efficient-attention-architecture"
  - "a-systematic-study-of-model-merging-techniques-in-large-language-models"
  - "a-systematic-survey-on-large-language-models-for-evolutionary-optimization-from-"
  - "accelerate-speculative-decoding-with-sparse-computation-in-verification"
---

<p class="paper-original-title" lang="en">Kimi K3: Open Frontier Intelligence</p>

<img src="/images/2607.24653v1/A__title.jpg" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

在大语言模型（LLM）的发展进程中，扩展定律（Scaling Laws）长期主导着预训练阶段的资源投入。然而，随着推理模型和强化学习在测试时计算（Test-time computation）上的突破，行业逐渐演化出两条平行的扩展轴线：一是继续扩大预训练基础模型的参数规模，二是在测试阶段通过强化学习、多步骤推理和长周期代理（Agentic）任务来扩展计算量。近年来，尽管开源社区在第二条轴线上（如测试时推理和复杂强化学习）取得了长足进步，但在第一条预训练轴线上却进展缓慢，绝大多数开源模型仍停留在百亿到千亿（1T以下）参数级别。当日益复杂的强化学习方法被应用于规模停滞的预训练底座时，开源模型与最顶尖闭源系统之间的能力差距面临进一步拉大的风险。

> ArXiv URL：https://arxiv.org/abs/2607.24653v1

Kimi K3 的发布正是为了同时在这两条轴线上突破现有的开源边界。作为一个原生多模态混合专家（MoE）模型，Kimi K3 拥有高达 2.8 万亿（2.8T）的总参数量和 1040 亿（104B）的激活参数，并支持高达 100 万 Token 的超长上下文窗口。通过在网络序列长度、模型深度和网络宽度三个维度上的架构创新——结合 Kimi Delta Attention、Attention Residuals 以及 Stable LatentMoE，Kimi K3 相比上一代 Kimi K2 实现了约 2.5 倍的整体扩展效率提升。

广泛的评估表明，Kimi K3 在长周期代码编写、代理任务、知识问答、复杂推理和视觉任务上均达到了前沿水平。虽然其综合表现略逊于目前最强大的闭源模型（如 Claude Fable 5 和 GPT-5.6 Sol），但 Kimi K3 始终优于测试集中的其他开源及闭源模型。Moonshot AI 团队开源了 Kimi K3 的完整模型权重，这不仅为超大规模多模态模型提供了极具价值的研究实体，也标志着开源人工智能正式踏入 3T 参数级别的深水区。

<img src="/images/2607.24653v1/benchmark0727.jpg" alt="Kimi K3核心评测结果对比" style="width:85%; max-width:600px; margin:auto; display:block;">

### 架构设计哲学：三维度的信息流扩展

Kimi K3 架构的核心设计理念是全面扩展信息在神经网络中的流动能力。现代 LLM 面临的根本瓶颈在于如何高效地在极长的序列、极深的网络层以及极宽的专家通道中传递和整合特征。为此，Kimi K3 沿三个互补的维度进行了架构重构：

在**序列长度维度**上，模型采用了混合注意力机制（Hybrid Attention），将擅长长序列高效混合的 Kimi Delta Attention (KDA) 与保留全局精确交互的 Gated MLA 周期性交织。

在**网络深度维度**上，模型引入了注意力残差（Attention Residuals, AttnRes），打破了传统残差连接逐层累加的马尔可夫式瓶颈，允许每一层选择性地跨层读取历史特征。

在**模型宽度维度**上，Kimi K3 部署了 Stable LatentMoE，将路由专家的特征空间投影至低维隐空间，在激活多达 16 个专家的同时控制了通信成本和计算开销。

此外，Kimi K3 的多模态能力并非通过后期的对齐网络拼接而来，而是依托于从头预训练的 MoonViT-V2 视觉编码器，实现了原生级别的视觉与文本特征融合。结合针对注意力投影头的 Per-Head Muon 优化器，这些组件共同构成了一个极其稳定且高效的万亿参数训练底座。

### 序列维度的突破：KDA与Gated MLA的混合编排

为了在 100 万 Token 的上下文中保持计算效率与信息召回率的平衡，Kimi K3 采用了一种逐层混合的注意力策略。在每个基础块（Block）中，模型连续堆叠 3 层 Kimi Delta Attention (KDA)，随后接入 1 层 Gated MLA，形成 $3:1$ 的混合比例。这种周期性的局部递归与全局注意力交替，既大幅降低了长文本的自注意力计算复杂度，又保证了关键信息不会在递归状态中衰减。此外，模型主干的最后一层固定为 Gated MLA 层，以确保最终输出前能够进行一次完整的全局上下文聚合。

#### KDA：带有下界约束的 Delta 规则

Kimi Delta Attention 是对传统 Delta 规则递归的扩展，核心在于引入了通道级别的遗忘门（Forget gate）。对于序列中的隐状态 $\mathbf{x}_{t}\in\mathbb{R}^{d}$，KDA 首先通过带有 $\operatorname{Swish}$ 激活的短卷积（ShortConv）生成查询（Query）、键（Key）和值（Value）向量。为了防止由于点积带来的数值漂移，查询和键向量还会经过 L2 归一化处理。

KDA 的递归状态更新遵循以下公式：




{% raw %}$$ \mathbf{S}_{t}=\left(\mathbf{I}-\beta_{t}\mathbf{k}_{t}\mathbf{k}_{t}^{\top}\right)\operatorname{Diag}(\mathbf{\alpha}_{t})\mathbf{S}_{t-1}+\beta_{t}\mathbf{k}_{t}\mathbf{v}_{t}^{\top} $${% endraw %}



其中，$\mathbf{\alpha}_{t}$ 是通道级别的衰减因子。为了在现代 GPU 上高效计算，KDA 采用分块并行（Chunkwise parallel）的形式，将序列切分为固定大小的块，通过块内（Intra-chunk）的局部计算和块间（Inter-chunk）的状态传递来加速。然而，分块并行在计算块内累积衰减的倒数（$1/\mathbf{\Gamma}_{[t]}^{1\rightarrow C}$）时，极易引发数值溢出问题，因为连续乘以 $(0,1)$ 之间的衰减因子会导致除数趋近于零。

Kimi K3 针对这一致命的数值不稳定性进行了优雅的改造——引入**下界衰减（Lower-bounded decay）**。有别于此前模型中无下界的负 Softplus 映射，Kimi K3 通过一个缩放的 Sigmoid 函数，将对数衰减限制在一个明确的安全区间内：




{% raw %}$$ \mathbf{g}_{t}^{h} =g_{\min}\operatorname{Sigmoid}\!\left(e^{A_{h}}\mathbf{z}_{t}^{h}\right)\in(g_{\min},0)^{d_{k}} $${% endraw %}






{% raw %}$$ \mathbf{\alpha}_{t}^{h} =\exp(\mathbf{g}_{t}^{h})\in\left(e^{g_{\min}},1\right)^{d_{k}} $${% endraw %}



通过设定明确的衰减下界（如 $e^{-5}$），Kimi K3 彻底消除了倒数爆炸的风险。这一机制的工程意义极为重大：在数值安全的保障下，KDA 能够放心地将所有对角线分块（Diagonal tiles）的因果交互直接转化为密集的 Tensor Core 矩阵乘法，而无需像以前那样退化为缓慢的显式位置对（Position-pair）计算，从而彻底打通了块内计算的硬件性能瓶颈。

#### 全秩门控与 Gated MLA 的精度修正

除了 KDA 的递归优化，Kimi K3 在输出门控上也做了升级，将低秩参数化改为依赖于输入的全秩投影（Full-rank gate），配合 RMSNorm 进一步稳定了每一层的方差。

在 $3:1$ 混合比例中的 Gated MLA 层，Kimi K3 保留了 DeepSeek-V2 引入的多头隐式注意力机制，将键值对压缩为低维隐向量 $\mathbf{c}_{t}=\mathbf{W}_{c}\mathbf{x}_{t}$ 以成倍缩减 KV Cache。但在工程实现上，研究团队发现 Flash Attention 在低精度下会产生有偏的舍入误差。为此，Kimi K3 的训练内核被重新设计，在训练期间强制将注意力输出保留在 FP32 精度。为了掩盖 FP32 带来的额外片上内存开销，团队将训练内核的访存流水线从重叠查询块（Query tile）调整为重叠 KV 暂存缓冲区，从而在不牺牲吞吐量的前提下保障了全局注意力的数值保真度。

### 深度维度的重构：从线性累加到注意力残差

传统的残差连接是 Transformer 架构的基础，它通过 $\mathbf{h}_{l} = \mathbf{h}_{l-1} + f(\mathbf{h}_{l-1})$ 的形式，将整个网络的历史信息压缩到一个单一的隐状态中逐层传递。随着网络深度的增加，这种方式类似于 RNN 在时间维度上的信息压缩，不可避免地会产生信息衰减或特征干扰。

Kimi K3 引入了 **注意力残差（Attention Residuals, AttnRes）**，将 Transformer 在序列维度上成功的“选择性注意力”机制，完美复刻到了网络深度维度上。在这一机制下，第 $l$ 层不再仅仅接收上一层的输出，而是主动“查询”之前所有层的表示：




{% raw %}$$ {\alpha_{i\to l}}=\frac{\phi\left(\mathbf{q}_{l},\mathbf{k}_{i}\right)}{\sum_{j=0}^{l-1}\phi\left(\mathbf{q}_{l},\mathbf{k}_{j}\right)},\qquad\mathbf{h}_{l}=\sum_{i=0}^{l-1}{\alpha_{i\to l}}\cdot\mathbf{v}_{i} $${% endraw %}



这种全注意力残差（Full Attention Residuals）虽然极大地拓宽了信息的跨层流动路径，但要求在内存中保留所有前序层的输出，导致 $O(Ld)$ 的内存和流水线并行通信开销。为了在性能和开销之间取得平衡，Kimi K3 采用了**块级注意力残差（Block Attention Residuals）**。

模型将近百层的网络划分为 8 个模块（Block），每个模块包含 12 层。在模块内部，各层的输出仍通过加法融合成一个单一的块级表示 $\mathbf{b}_{n}$；但在跨模块交互时，当前层可以向之前所有模块的块级表示发起 Attention 查询。这种粗粒度的跨层注意力机制，既能让网络在极深处直接抽调浅层的原始特征，又将内存开销控制在了可接受的范围内。

### 宽度维度的扩展：极致稀疏的 Stable LatentMoE

为了在不显著增加推理计算量的前提下扩充模型的知识容量，Kimi K3 将 MoE 架构推向了极致：总计配备了 896 个路由专家，并在每个 Token 上激活其中的 16 个。这高达 56 倍的稀疏度赋予了模型极大的参数表征空间。

然而，传统的 MoE 在面对如此庞大的专家基数时会遭遇严重的通信瓶颈，因为每一个被选中的专家都需要接收完整的 $d$ 维 Token 向量。Kimi K3 采用的 **LatentMoE** 机制通过降维化解了这一难题：每个 Transformer 层依然保留 2 个全宽度的共享专家处理共性特征，而 896 个路由专家则在一个紧凑的低维隐空间（Latent space，宽度为 $\ell$）中运行。

在 2.8T 的超大参数规模下，MoE 的训练稳定性面临严峻挑战。Kimi K3 在 LatentMoE 的基础上引入了三大关键维稳机制，构建了 **Stable LatentMoE**：

第一，**归一化 LatentMoE（Normalized LatentMoE）**。原始设计中，路由专家的聚合输出会直接进行向上投影，导致不同专家组合带来的尺度波动极大。Kimi K3 在专家聚合结果与向上投影之间插入了 RMSNorm，这一改动不仅平息了训练过程中的尺度漂移，还在验证集损失和下游评测上带来了持续的增益。

第二，**SiTU-GLU 激活函数**。传统的 SwiGLU 激活函数在面对巨大的正向输入时，其输出呈现无界增长，极易诱发梯度爆炸。Kimi K3 独创了 SiTU-GLU（Sigmoid Tanh Unit GLU）激活函数：




{% raw %}$$ \operatorname{SiTU-GLU}(\mathbf{x})=\left[\beta_{1}\tanh\!\left(\frac{\mathbf{W}_{g}\mathbf{x}}{\beta_{1}}\right)\odot\operatorname{Sigmoid}(\mathbf{W}_{g}\mathbf{x})\right]\odot\left[\beta_{2}\tanh\!\left(\frac{\mathbf{W}_{u}\mathbf{x}}{\beta_{2}}\right)\right] $${% endraw %}



通过在门控分支和向上投影分支中引入 $\tanh$ 缩放，SiTU-GLU 在输入趋近原点时表现得与 SwiGLU 几乎一致，保留了良好的非线性特征；但当输入值极大时，其输出会被严格限制在 $\beta_{1}\beta_{2}=100$ 的上限内。这种有界激活设计像一道安全阀，从根本上阻断了异常激活值的无序传播。

第三，**分位数平衡（Quantile Balancing）路由机制**。过去 MoE 负载均衡高度依赖于辅助损失函数（Auxiliary Loss），但这往往会干扰主语言建模目标。Kimi K3 摒弃了辅助损失，采用通过动态偏置调整的分位数平衡算法。在路由决策时，不仅参考专家的原始得分，还会加上一个动态计算的专家偏置 $b_j$。

在分布式集群和极大的全局 Batch Size 下，精确计算每个专家的得分分布分位数是不现实的。Kimi K3 实现了一种基于直方图估计（Histogram estimation）的分布式分位数统计算法：每个计算节点只需维护数百个得分区间的频数直方图并进行一次 All-reduce 操作，即可在极低的通信成本下，还原出具有全局视野的 Token 路由边界。

### 原生多模态：从零预训练视觉底座

Kimi K3 是一个原生多模态模型，文本、图像和视频共享同一个 100 万 Token 的上下文。这一特性的基础在于其视觉编码器 **MoonViT-V2**。

与行业内普遍采用 SigLIP 等经过对比学习（Contrastive learning）预训练的模型作为视觉特征初始化的做法不同，Kimi K3 的多模态训练做出了一个大胆且关键的决策：**完全放弃对比学习预训练权重，直接用语言模型的 Next-token prediction 目标从头（From scratch）训练视觉编码器。**

研究团队发现，当强行将具备全局语义偏好的预训练对比编码器接入生成式 LLM 时，联合优化极不稳定，梯度范数居高不下且频繁出现毛刺尖峰。而随机初始化的 MoonViT-V2 在整个预训练过程中表现出了卓越的平滑度与收敛性。不仅如此，由纯自回归语言目标塑造出来的视觉表征，天然更契合细粒度的文本对齐和结构化信息提取任务。配合取消线性投影偏置项等细节优化，这套 27 层的从头训练视觉塔在下游评测中完全追平甚至超越了基于 SigLIP 初始化的基线版本。这种原生的深度融合使得 Kimi K3 能够在一个连续的数据流中自主编写代码、渲染 UI 界面并根据视觉反馈迭代修改，中间无需任何跨模型的数据交接。

### 测试时扩展与强化学习基础设施

拥有了 2.8T 的强悍底座后，Kimi K3 的后训练（Post-training）阶段全面转向了基于超长上下文的代理任务强化学习。模型在涵盖多推理难度、通用代理和高阶代码编写的环境中进行了大规模探索。

Kimi K3 的训练环境不仅仅是简单的问答，而是包含可验证搜索、内核级代码优化、包含视觉反馈的工具调用、网页开发等复杂的长期交互任务。在这些任务中，模型往往需要执行数百次工具调用，累计生成并阅读上百万个 Token，形成了一个完整的“推理、行动、观察、验证、适应”闭环。不同垂直领域和推理深度训练出的策略，最终通过多教师在策略蒸馏（Multi-teacher on-policy distillation）技术，被完美融合进单一的 Kimi K3 模型权重中。

支撑这一切的，是团队为了多万亿参数模型量身打造的软硬协同基础设施。针对 KDA，开发了融合算子和跨设备的上下文并行机制；针对超大规模 MoE，设计了 MoonEP 系统以实现零拷贝通信和极致的专家并行负载均衡；针对百万 Token 代理级强化学习，部署了支持微型虚拟机沙盒驻留和外部 KV 缓存保留的混合系统环境。

### 总结与影响

Kimi K3 的发布，不仅是一次算力和参数规模的暴力堆叠，更是对超大型语言模型信息流动机制的一次系统性重构。混合注意力机制平衡了长序列的效率与质量，注意力残差打破了深层网络的马尔可夫屏障，而稳定化的 LatentMoE 则以极小的通信代价激活了海量的专业知识。

这些在网络深度、宽度和序列长度上的底层创新，结合从头训练的原生视觉编码器和百万 Token 级别的强化学习范式，使得 Kimi K3 成功确立了新的开源基准。它以 2.8T 的庞大身躯和 2.5 倍的扩展效率提升，证明了开源社区同样具备向最前沿智能体形态和复杂多步推理系统发起冲击的技术底蕴。通过释放完整模型权重，Kimi K3 无疑将极大加速学术界与工业界在下一代通用人工智能路线上探索的步伐。
