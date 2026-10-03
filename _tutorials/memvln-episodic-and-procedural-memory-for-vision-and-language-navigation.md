---
layout: default
title: "MemVLN：用金字塔情景与程序记忆，让连续视觉语言导航跑出14 FPS"
description: "针对这一矛盾，来自香港中文大学、香港科技大学、香港大学以及 Lightspeed 团队的研究人员提出了全新导航框架 MemVLN 。该工作从人类认知系统的双重记忆结构中汲取灵感：一方面引入 金字塔分辨率的情景记忆（Episodic Memory） ，在完整保留二维空间结构的前提下。"
arxiv_id: "2607.23504"
paper_published: "2026-07-26"
published_at: "2026-10-03T13:15:07.852414+08:00"
topics:
  - "知识系统"
  - "多模态&视觉"
tags:
  - "Atomic Mid-level Actions"
  - "Autoregressive Decoding"
  - "Episodic Memory"
  - "LLM"
  - "MemVLN"
  - "Procedural Memory"
related_tutorials:
  - "memrl-self-evolving-agents-via-runtime-reinforcement-learning-on-episodic-memory"
  - "remember-me-refine-me-a-dynamic-procedural-memory-framework-for-experience-drive"
  - "latent-learning-episodic-memory-complements-parametric-learning-by-enabling-flex"
  - "flashdrive-flash-vision-language-action-inference-for-autonomous-driving"
seo_title: "MemVLN：用金字塔情景与程序记忆，让连续视觉语言导航跑出14 FPS"
---

<p class="paper-original-title" lang="en">MemVLN: Episodic and Procedural Memory for Vision-and-Language Navigation</p>

<img src="/images/2607.23504v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在具身智能领域，连续环境下的视觉语言导航（Vision-and-Language Navigation in Continuous Environments, VLN-CE）一直被视为极具挑战的真实世界交互任务。与传统预设拓扑图节点之间“瞬移”的离散图导航不同，连续导航要求智能体在未知的 3D 物理空间内，完全依靠自身的第一人称视觉传感器输入，实时将自然语言指令转化为低层电机控制信号（如前移、左转、右转、停止）。

> ArXiv URL：https://arxiv.org/abs/2607.23504v1

随着多模态大模型（LVLM）的爆发，学术界开始尝试将视觉感知、语言理解与动作生成统一到端到端大模型框架中。然而，连续空间导航直接撞上了大模型架构的致命痛点：**智能体既需要维持足够长的时序视觉历史以避免路径漂移，又必须保证极低的推断延迟来应对即时物理控制。** 当前的端到端方案要么粗暴地均匀抽样少量帧（如全局仅抽 8 帧），丧失了连续控制所需的时间密度；要么采用 Token 合并（Token Merging）来压缩序列，这不仅抹杀了精细的空间几何细节，更破坏了标准的二维网格结构，与现代视觉语言模型广泛采用的多模态旋转位置编码（M-RoPE）产生直接冲突。再加上大语言模型底层的自回归逐 Token 解码机制，高昂的延迟让导航系统难以在真实物理帧率下顺畅运行。

<img src="/images/2607.23504v1/teaser_cropped.webp" alt="MemVLN 模拟人类情景记忆与程序记忆的概念总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

针对这一矛盾，来自香港中文大学、香港科技大学、香港大学以及 Lightspeed 团队的研究人员提出了全新导航框架 **MemVLN**。该工作从人类认知系统的双重记忆结构中汲取灵感：一方面引入**金字塔分辨率的情景记忆（Episodic Memory）**，在完整保留二维空间结构的前提下，对远期历史实施渐进压缩、对近期感知保留高清输入；另一方面引入**面向快速动作的程序记忆（Procedural Memory）**，将控制决策从多步自回归解码转化为紧凑中层动作词表的单步预测（One-shot Prediction）。

实验结果显示，MemVLN 在仅使用单目 RGB 摄像头、不依赖任何深度传感器与外部里程计的前提下，大幅刷新了 R2R-CE 和 RxR-CE 两大基准的性能。相比 Qwen3-VL-4B 基准，MemVLN-4B 在 R2R 和 RxR 上的成功率（SR）分别提升了 5.8% 和 9.7%，推断延迟降低为原来的七分之一，系统端到端推断速度稳定达到 14 FPS，真正让大模型导航兼顾了全局轨迹一致性与实时控制响应。

### 为什么现有长序列多模态大模型在导航中频频受挫？

连续环境导航的本质是一个长时程的时空状态决策过程。智能体在走廊、房间之间穿梭时，每秒都在产生新的高分辨率观察帧。为了判断“穿过客厅后左拐进入第二个门”中的“第二个门”，模型必须能召回数十秒前的视觉地标信息，这对应着极其庞大的视觉上下文。

<img src="/images/2607.23504v1/observation_analysis.webp" alt="视觉观察的时空特性与注意力分布分析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

常规处理视频长上下文的方式主要有两种，但都难以适配高频连续控制。

第一种方式是固定帧率或均匀降采样。例如无论轨迹多长，都只保留固定数量的几帧图像。这种策略对静态长视频问答或许可行，但在连续物理移动中，智能体极容易在转弯、穿门等关键微操节点丢失关键视差信息，导致碰撞或动作振荡。

第二种方式是利用 Token Merging（如 ToMe）等池化机制按注意力动态压缩视觉 Token。这一方案在分类和图文检索任务中表现优异，但在具身导航中引发了不可忽视的结构性副作用。首先，随机合并或基于相似度压缩 Token 会直接打乱图像规整的 2D 网格结构，丧失了细粒度空间几何定位能力。其次，当前顶尖的多模态模型（如 Qwen2-VL、Qwen3-VL）高度依赖多模态旋转位置编码（M-RoPE）来精确对齐三维时空坐标，非均匀压缩 Token 会破坏位置编码的时空网格假设，引发表征崩溃。

更严重的问题隐藏在动作生成阶段。通用大模型在输出指令时，通常以自然语言文本格式逐词生成（例如输出包含多个字符的动作名称或坐标），自回归解码的多次前向传播极大地拖慢了推理循环，导致智能体面对连续环境变化时出现严重的动作滞后。

### 金字塔情景记忆：规整网格下的时空保真压缩

人类在物理空间中行动时，不会在脑海中以同等像素精度回放过去半小时的所有画面。人们对刚刚经过的几步保留着极其清晰的局部细节（用于避障和微调角度），而对几分钟前的路径则只保留粗粒度的语义地标与方向概略。

<img src="/images/2607.23504v1/method_cropped.webp" alt="MemVLN 整体系统架构与双记忆机制流程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

MemVLN 将这一认知机制抽象为**金字塔分辨率的情景记忆管理模块**。模型在每个时间步 $t$ 接收当前帧 $v_{t-1}$ 与历史帧序列 $\mathcal{H}_{t-1} = \{v_0, \dots, v_{t-2}\}$。金字塔机制并没有粗暴删除远期帧，也没有打碎 Token 的空间排布，而是根据时序距离将历史划分为不同时间区间，赋予差异化的输入分辨率：




{% raw %}$$ v'_{t}=\begin{cases}\text{Rescale}(v_{t},r_{imm}),&T-B_{short}<t\leq T-1\\ \text{Rescale}(v_{t},r_{short}),&T-B_{long}<t\leq T-B_{short}\\ \text{Rescale}(v_{t},r_{long}),&0\leq t\leq T-B_{long}\end{cases} $${% endraw %}



其中 $r_{imm} > r_{short} > r_{long}$。对于紧邻当前决策的即时感知（Immediate Percepts），系统分配最高分辨率 $r_{imm}$，确保视觉编码器 $\mathcal{E}_{vis}$ 能提取精准的局部空间结构；对于中期历史则下采样至 $r_{short}$；而对于远期深层记忆，则缩放到极低分辨率 $r_{long}$。

<img src="/images/2607.23504v1/pyramidalvstome.webp" alt="金字塔分辨率策略与传统 Token 压缩（ToMe）的对比机制" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

这一设计的精妙之处在于两点：

1. **显存与计算开销的可控性**：低分辨率直接在图像空间完成重采样，使得每个远期帧切分出的视觉 Token 数量呈平方级骤减，让有限的上下文窗口能够容纳跨度极长的历史轨迹。

2. **完美兼容 M-RoPE 空间位置编码**：由于只是调整图像分辨率尺寸，所有生成的视觉 Token 在空间维度上依然严格满足均匀的 2D 矩形网格，位置编码的时空连续性得到完整保留，避免了 ToMe 导致的网格破损问题。

<img src="/images/2607.23504v1/variants_cropped.webp" alt="金字塔分辨率变体与模型注意力权重分布特征" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

注意力分布的实验结果清晰印证了这一设计的合理性：模型对距离当前决策点最近的高清帧赋予了最高的注意力权重，而随着时序拉长，远期低分辨率帧提供的粗粒度全局表征足以维系整体轨迹的方向判断，既避免了无谓的计算浪费，又防止了路径遗忘。

### 程序记忆与快速动作：绕过自回归瓶颈的单步决策

即便解决了长时程视觉历史的高效输入，大模型固有的自回归解码延迟依然是将导航推向实时的硬屏障。标准的大模型策略通常需要模型生成一串文本序列，甚至多个 Token 来表达动作概率，每生成一个 Token 就意味着调用一次数十亿参数的大模型前向计算。

在认知心理学中，**程序记忆（Procedural Memory）**负责管理人类熟练技能的无意识、自动化执行（如骑自行车或敲击键盘时的肌肉记忆），这类行为完全不需要经过深思熟虑的高层语言组织，属于瞬间反射式的输出。

MemVLN 据此设计了面向控制的**快速动作机制（Fast Action）**。研究团队没有采用多步自回归生成方案，而是重新定义了动作解码流程：

1. **构建紧凑的增强动作词表 $\mathcal{A}_{aug}$**：将底层控制指令映射为离散且高信息密度的原子中层动作原语，或者短时域内的动作基元序列。

2. **复用词表单 Token 投影**：从原预训练大模型的词表空间中直接复用一组单 Token 来代表这些增强动作，使得策略网络仅需一次前向传播（Single forward pass），就能在这一紧凑输出空间中完成单步分类预测。

这一改动彻底绕过了自回归循环的等待延迟，将原本可能长达数百毫秒的迭代推理过程压缩为一次纯粹的单步判决。在多项延迟消融实验中，该设计展现了压倒性的效率优势，直接推动 MemVLN 达成 14 FPS 的端到端推理性能，满足了连续物理仿真与真实机器人控制的基本实时门槛。

### 统一端到端训练与多源轨迹扩增

在训练阶段，MemVLN 采用标准的多模态监督微调（SFT）机制，将金字塔视觉特征序列 $\mathbf{F}_t$ 与文本导航指令 $\mathcal{I}$ 进行跨模态融合，直接对真实的动作目标 $a_t^* \in \mathcal{A}_{aug}$ 计算交叉熵损失：




{% raw %}$$ \mathcal{L}=-\frac{1}{T}\sum_{t=1}^{T}\log P(a_{t}^{*}\mid\mathbf{F}_{t},\mathcal{I};\theta) $${% endraw %}



为了让模型在复杂的未见（Unseen）环境中具备极高的泛化能力，研究团队构建了涵盖仿真与真实轨迹的大规模多源训练数据集：

*   **基础环境数据**：囊括标准的 R2R、RxR 及其通过 EnvDrop 增强生成的多种空间轨迹。

*   **空间先验扩增**：随机抽样来自 HM3D 与 Gibson 逼真环境的 416K 对 ScaleVLN 数据，以丰富空间拓扑的多样性。

*   **连续交互与纠错数据**：引入 557K 条人类真实连续导航轨迹，同时结合 DAgger 算法合成的 278K 条在线交互轨迹。

数据消融结果表明，DAgger 产生的交互式策略纠错数据对连续控制的鲁棒性起到了决定性作用。相比于被动的静态路径回放，带有偏差纠正特征的数据显著改善了智能体在偏离主路线时的自愈（Self-Correction）能力，使模型在 R2R 上的探索度（OS）提升了 2.4%，在多语言、长时程的 RxR 基准上更是带来了 4.6% 的 OS 和 3.5% 的成功率（SR）跃升。

### 基准评测：单目 RGB 击败多传感器方案

在标准的 Habitat 仿真平台上，MemVLN 在两大最具挑战性的连续导航未见场景验证集（Val-Unseen）上接受了严格检验：R2R-CE（短指令、短距离）与 RxR-CE（多语言细粒度长指令、平均轨迹更长更复杂）。

评测基准通常划分为两类派系：一类是依赖特定航路点预测器（Waypoint Predictor）的层级方法，它们往往强依赖环境先验甚至仿真器特有的深度/全景拼接；另一类则是直接基于原生视角的端到端连续控制。

在端到端非航路点预测体系中，MemVLN-8B 刷新了连续环境下的性能高度：

*   **R2R-CE 基准**：达到了 **65.3% 的探索度（OS）** 与 **58.4% 的导航成功率（SR）**。

*   **RxR-CE 基准**：在长轨迹连续导航中实现了 **66.0% 的成功率（SR）** 与 **57.3% 的路径加权成功率（SPL）**。

*   **跨数据集零样本迁移**：若仅在 R2R 上训练，MemVLN 在完全未见的 RxR 跨域评测中依然斩获了 47.9% 的 OS，表现出极强的自然语言地标泛化能力。

尤为值得关注的维度在于**传感器依赖与鲁棒性**。以往大量具有竞争力的基准方案（如航路点预测模型）通常必须依赖全景图像拼接（Panoramic Views）、深度传感器（Depth）或高精里程计（Odometry）。然而在现实世界中，深度图极易受到光照反光和遮挡的干扰，里程计也会随着机械打滑产生难以消除的累积漂移。

MemVLN **仅依赖单目 RGB 视频流与纯文本指令**，便全面超越了诸多装备了全景或深度硬件的旧有基线。这证明通过金字塔时序压缩与单步快速策略，纯视觉大模型已具备自主重构三维空间几何关系、抗衡传感器缺失的内在泛化能力。

### 总结与启示

回顾 MemVLN 的设计架构，其核心贡献并不是单纯堆叠多模态模型参数，而是从认知科学中提炼出了行之有效的计算架构取舍：

首先，长时序多模态具身建模不能盲目依赖破坏空间几何排布的 Token 剪枝手段。金字塔分辨率机制证明，在图像输入层保持规整几何网格的降采样，能以极小的语义代偿代价换取超长历史帧的兼容性，为适配现代多模态 RoPE 架构指明了方向。

其次，对于高频物理交互而言，自回归大模型的“逐 Token 生成”范式属于天然的效率杀手。将语言模型的庞大容量限制在单步中层动作原语的高维映射上，是解决实时物理控制延迟的一剂良方。

从 14 FPS 的实际推断速度到纯单目 RGB 的极简传感器配置，MemVLN 展现出大模型赋能具身物理实体的可行路线：只有在系统层面解开显存历史开销与动作推理时延的双重枷锁，大语言模型蕴藏的世界常识才能真正转化为现实物理世界中敏捷、连贯的自主导航能力。
