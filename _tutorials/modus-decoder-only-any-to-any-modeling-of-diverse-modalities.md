---
layout: default
title: "Modus：单模型统一15种模态，Decoder-Only如何打破Any-to-Any壁垒？"
description: "由 Apple、香港中文大学（CUHK）、EPFL 等机构联合提出的 Modus ，直接打破了这一工程惯例。它证明了： 不需要为每个模态定制专门的预测头，也不需要外挂专有损失函数或复杂的任务管道，完全基于统一的单解码器（Decoder-Only）架构。"
arxiv_id: "2607.25948"
paper_published: "2026-07-28"
published_at: "2026-09-12T13:15:08.779692+08:00"
topics:
  - "基础模型"
tags:
  - "Modus"
  - "any-to-any multimodal modeling"
  - "chained cross-modal generation"
  - "cross-modal self-verification"
  - "decoder-only architecture"
  - "modality-agnostic conditioning"
related_tutorials:
  - "encoder-decoder-or-decoder-only-revisiting-encoder-decoder-large-language-model"
  - "causal-reasoning-favors-encoders-on-the-limits-of-decoder-only-models"
  - "nextflow-unified-sequential-modeling-activates-multimodal-understanding-and-gene"
  - "seedance-2-0-advancing-video-generation-for-world-complexity"
---

<p class="paper-original-title" lang="en">MODUS: Decoder-Only Any-to-Any Modeling of Diverse Modalities</p>

<img src="/images/2607.25948v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在大模型技术演进的历程中，**多模态**通常遵循两条截然不同的路线：一条是以视觉语言模型（VLM）为主导的“理解型”路线，通过视觉编码器将图像压缩成视觉词元，输送给自回归因果语言解码器，但这类模型大多只能输出文本；另一条是以扩散模型或流匹配（Flow Matching）为主导的“生成型”路线，专精于从文本或几何先验生成图像或特定空间图谱，但往往难以承载复杂符号逻辑与开放式推理。

> ArXiv URL：https://arxiv.org/abs/2607.25948v1

如果想要构建一个通用的全对全（Any-to-Any）多模态系统——任意输入一种或多种模态组合，任意输出另一模态——现有的解决方案通常依赖编码器-解码器（Encoder-Decoder）架构，或者借助大语言模型（LLM）作为文本中转站。前者通常需要从零开始联合训练所有模态的跨模态对齐，计算开销巨大，且无法直接继承顶级语言大模型已经学到的丰富通用先验；后者（如 NExT-GPT、AnyGPT）则陷入了“文本中心主义”的陷阱：当模型尝试从深度图生成 RGB 图像时，必须先将深度图描述成文本，再从文本合成图像，细粒度的三维几何与空间拓扑结构在文本中转过程中几乎损耗殆尽。

由 Apple、香港中文大学（CUHK）、EPFL 等机构联合提出的 **Modus**，直接打破了这一工程惯例。它证明了：**不需要为每个模态定制专门的预测头，也不需要外挂专有损失函数或复杂的任务管道，完全基于统一的单解码器（Decoder-Only）架构，就能在同一个模型中原生处理包括文本、RGB图像、深度图、表面法线、边缘轮廓、分割掩码、目标检测边框以及 DINOv2、CLIP 特征在内的 15 种模态**。更重要的是，Modus 无需从零冷启动，而是通过巧妙的序列编排和训练配方，直接继承预训练解码器的先验，在仅耗费 5,664 GH200 卡时的渐进式训练下，达到了比肩甚至超越专科模型的跨模态生成与理解水平。

<img src="/images/2607.25948v1/Method_v21.webp" alt="Modus 架构与统一序列建模全景" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 对称式 Token 化：让所有模态在同一张序列画布中对齐

要让一个 Decoder-Only 架构同时接纳离散符号和连续物理场，核心的第一步是如何将本质相异的模态抽象为统一的上下文（Context）。过去许多系统之所以架构臃肿，是因为针对检测任务加检测头、针对深度图加卷积上采样头、针对文本用交叉熵损失，导致模型结构碎片化。Modus 抛弃了这些特定任务头，将所有的输入与输出一视同仁地转化为统一的词元序列（Token Sequence）。

在 Modus 的设计中，现实世界的多模态数据被清晰划分为两大数学表达类别：**1D 序列模态**与 **2D 空间模态**。

对于 1D 序列模态，涵盖了传统自然语言文本、目标检测与 Grounding 边框坐标，以及全局和局部的特征词元（例如 DINOv2、CLIP 和 ImageBind 特征）。这类模态天然具有离散或因果属性，被统一映射为 1D 离散或连续词元，其建模逻辑是标准的自回归下一个词元预测（Next-Token Prediction, NTP），通过最大化条件似然来学习序列概率分布。

对于 2D 连续空间模态，包括 RGB 彩色图像、深度图（Depth）、表面法线图（Surface Normals）、Canny 边缘图、语义分割掩码以及 SAM 掩码。这类模态蕴含着密集的二维空间几何结构，Modus 并未强行将其粗暴量化为离散的 VQ 编码，而是采用了双表征路径：高维语义层面，提取来自 ViT 的连续特征以保留宏观概念；细粒度重建层面，则通过预训练变分自编码器（VAE）将物理图像投射到紧凑的潜在空间（Latent Space）。当 2D 模态作为输入条件时，直接将干净的无噪 VAE 特征拼入上下文；当其作为生成目标时，模型在潜在空间中施加流匹配（Flow Matching）目标函数，学习从标准高斯噪声到目标模态的连续速度场向量。

这两种计算范式在 Modus 的单一解码器内部通过“混合 Transformer 专家”（Mixture-of-Transformers）机制共存：1D 专家负责处理 NTP 损失下的符号因果依赖，2D 专家负责预测流匹配的速度场。在模态内部，1D 模态遵循因果注意力，2D 模态在自身空间内则展开完全的双向自注意力；而在跨模态层面，所有词元依然维系着由左至右的共享全局因果上下文。任意模态都能作为上下文的一部分充当条件，也都能被放置在序列末端作为生成目标。

### 根治多模态混淆：流匹配中为何必须抛弃 Logit-Normal 采样？

在将生成范式切换为潜在流匹配时，研究团队遭遇了一个致命的系统性隐患：**模态混淆（Modality Confusion）**。

在常规的单模态图像生成研究（如各类开源扩散模型和连续流生成器）中，研究者通常偏爱使用 **Logit-Normal 时间步采样策略**。这种策略在数学上倾向于在中等扩散时间步（$t \in [0.3, 0.7]$ 附近）集中采样更多的训练样本，因为在纯视觉图像生成中，中间时间步往往决定了图像的主要结构与纹理过渡，过早或过晚的噪声对最终生成质量的边际贡献相对较低。

然而，一旦把这一经验套用到包含 15 种模态的 Any-to-Any 联合训练中，灾难便发生了：模型开始严重违背指令提示。例如，输入一张室内 RGB 图像并明确下达指令要求预测“深度图”，模型最终生成的输出却往往掺杂了“表面法线图”的五彩色彩分布，或者在生成 Canny 边缘时夹带了语义分割块。

<img src="/images/2607.25948v1/figure3_modality_mix_v13_cam.webp" alt="Logit-Normal 与 Uniform 时间步采样在模态对齐上的机制对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

深入剖析流匹配的动力学轨迹，作者发现了多模态生成独特的物理规律：**多模态生成在时间步的权责分配上具有高度的阶段非对称性**。在逆向生成的初始阶段（即大时间步 $t \to 1$ 的极早噪声阶段），速度场预测的主要任务并非细化轮廓，而是根据输入的上下文提示词元，在多个完全不同的流分布流形之间做出“方向性抉择”——确定自己究竟是要流向深度空间的单通道渐变流形，还是要流向法线空间的三通道正交向量流形。一旦跨过这个阶段，进入中间和后期时间步（$t \to 0$），向量场的作用退化为局部的几何平滑与细节锐化。

传统的 Logit-Normal 采样大幅压缩了极早时间步的监督权重，使模型在决定流形归属的起点缺乏充分的监督信号，导致轨迹在起点处就发生漂移。为了彻底纠正这种跨模态漂移，Modus 果断舍弃了扩散社区惯用的中间偏向采样，全面采用**均匀时间步采样（Uniform Timestep Sampling）**。通过赋予早期、中期与晚期时间步均等的梯度反传权重，模型得以在起点牢牢锁定由指令指定的模态物理空间，彻底消除了不同模态间的特征污染现象。

### 2900万多维样本与渐进式分期训练

全对全多模态模型的另一大瓶颈在于数据的全维度对齐。在现有的公开数据集中，图像往往只与文本配对，深度图只与 RGB 图像配对，检测标注则集中在 COCO 风格的特定数据集。如果只用两两配对的孤立数据集进行训练，模型很容易将特定任务与特定上下文格式绑定，无法学会通用的概念流转。

为此，研究团队构建了包含 2900 万样本的 **Modus-Dataset**。他们以 BLIP-3o 的高质量图像与长文本标注为底座，利用当下一线顶尖的单模态专家模型，对每一张图像展开了密集的离线标注提取：

- 利用 DepthAnything 提取高质量绝对/相对深度；

- 利用 Marigold 提取细粒度的表面法线矢量；

- 利用 Grounded-SAM 与标准 SAM 生成稠密的像素级实例与语义分割掩码；

- 利用 Canny 算子与边缘模型生成多粒度的边缘线稿；

- 利用 GLaMM 与 EVA-02 驱动的 ViTDet 生成接地坐标与目标检测框；

- 利用 DINOv2、CLIP 及 ImageBind 编码器，同时提取全局表征向量与局部网格表征。

所有模态被严丝合缝地对齐在同一个场景原语之上。这意味着，在 Modus 的训练过程中，不仅能进行常规的“文本到图像”或“图像到深度”，还可以任意抽取极罕见甚至前所未见的模态转换组合，比如“Canny 边缘 $\rightarrow$ 表面法线”、“深度图 $\rightarrow$ 局部 DINOv2 特征”，或是多重条件组合如“深度图 + 局部线稿 $\rightarrow$ 语义分割”。

为了稳妥地驱动庞大的联合参数，Modus 采用了三阶段课程训练策略（Curriculum Learning）：

第一阶段主要基于预训练语言-视觉基础模型，初步适配多模态基础词元输入；

第二阶段逐步引入几何、结构与稠密语义模态，并在样本中随机屏蔽和暴露不同的模态输入，强迫自回归因果上下文理解跨模态关联；

第三阶段全面放开多条件（Multi-Conditioning）混合训练，随机采样多模态子集作为上下文，并将生成目标扩展至全模态。在 64 块 GH200 上累计耗时仅不到 90 个小时，便成功产出了覆盖 14B 与 77B 架构的完整统一权重。

### 为什么语义和几何不可偏废？ViT 与 VAE 特征的双轨互补

对于任意生成系统，一个长期存在的争论是：到底应该依赖高层语义特征（如 ViT Token），还是低层重建潜空间（如 VAE Latent）？

Modus 针对这一问题给出了详尽的实证消融分析。当模型需要从 RGB 图像生成几何深度或进行空间变换时，作者对比了三种不同的条件输入策略：仅输入 ViT 语义词元、仅输入 VAE 连续潜变量，以及二者兼备的混合表征。

<img src="/images/2607.25948v1/fig_representation_v2_cam.webp" alt="ViT 与 VAE 条件输入对几何与语义保真度的影响对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

实验结果清晰地揭示了各自的局限性：

如果在条件中**仅仅使用 ViT 特征**，模型能够敏锐地捕捉到画面的全局概念，例如清晰识别出画面中央是一把椅子，但在局部几何形状上会出现显著的扭曲和失真。深度图上的椅背厚度、椅腿的支撑结构往往被模糊化或产生形变，无法做到像素级的空间对齐。

反之，如果**仅仅使用 VAE 特征**，局部像素的高频几何边缘被完整保留下来，但模型极易丧失全局物理语义的先验认知。在论文展示的典型样例中，墙壁上悬挂的黑色显示器因为颜色极深，在纯 VAE 局部线索下被模型错误地推断为一个深陷进墙体内部的“空洞凹陷”，完全违背了现实物理世界的常识。

有趣的是，这种“把黑色屏幕误判为凹洞”的几何理解缺陷，在近期针对闭源商业巨头 GPT-4o 的视觉几何评测中也被反复观察到。这间接暗示了主流纯自回归多模态模型内部极度偏向高层抽象表征、忽视细粒度低层几何潜空间的通病。Modus 将 ViT 高层语义与 VAE 低层连续几何完美缝合，在高层认知约束下填充精确的局部物理场，彻底解决了这一失效难题。

### 链式生成与跨模态自验证：单模型闭环的独特威力

当一个系统真正具备对称的 Any-to-Any 泛化能力后，它所能带来的应用延展将超越单任务模型的简单拼凑。Modus 展示了两项无需任何额外模块或重训练就能原生执行的高阶能力：**链式生成（Chained Generation）**与**跨模态自验证（Cross-Modal Self-Verification）**。

<img src="/images/2607.25948v1/modus_grid_chained.webp" alt="Modus 的全对全链式多模态生成矩阵" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

所谓**链式生成**，是指用户无需一次性穷尽所有生成意图，而是可以将前一步生成的产物作为新的上下文词元，动态追加至 KV-Cache 尾部，持续推演下一模态。例如，用户可以仅输入一段极其抽象的草图轮廓，先让 Modus 推导出语义分割掩码，接着利用“草图 + 分割掩码”推导深度场，再联合已生成的全部物理几何线索，一步步渲染出细节丰满、物理规律完全自洽的高保真 RGB 真实照片。在整个生成流中，所有中间产物始终紧密锚定在同一个潜在场景中，完全规避了传统级联流水线中各模型标准不一致导致的累积漂移。

更为惊艳的是**跨模态自验证**机制。在以往的大模型最佳候选采样（Best-of-$N$）中，通常需要额外训练专门的奖励模型（Reward Model）或判别器来给生成的候选图像打分。但在 Modus 框架下，模型本身就是全模态的闭环系统。

在文本到图像生成测试中，Modus 可以针对同一段提示词并行生成 4 张候选图像；紧接着，模型将这些生成的图像重新送入自身，反向生成与提示词对应的 Grounding 检测边框或执行视觉问答（VQA）。通过对比自身反向推导出的目标置信度与条件提示词的契合程度，Modus 能够完全自主地评判哪一张图像真正准确地还原了指令内容。在 GenEval 基准测试中，Modus 仅凭借这一完全无外部辅助的自闭环验证机制，便直接将生成达标分数从 0.81 强力拉升到了 0.84。

### 结语

长期以来，多模态社区在 Decoder-Only 的扩展上步履维艰，要么退化为仅负责语言交互的被动应答机，要么不得不妥协于复杂的扩散中继网络。Modus 的成功揭示了一条清晰且优雅的演进路径：**统一的自回归与流匹配分工、均匀平衡的时间步采样机制，搭配多模态完全对齐的大规模合成训练数据，足以让单一 Decoder-Only 架构原生地驾驭任意维度的现实世界信号**。

无论是从离散的符号语义跃迁至连续的几何物理世界，还是让模型通过反思自身生成的模态来审视推断质量，Modus 都展现出了通用具身基座和下一代通用世界模型该有的雏形。随着该框架全部代码、权重与 2900 万样本数据集的开源，多模态生成研究正迈向一个不再区分“理解”与“生成”、不再设置模态特权的全新阶段。
