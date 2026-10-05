---
layout: default
title: "JoyAI-Video-Edit：单卡720p 30帧，两步蒸馏攻克流式视频编辑"
description: "JoyAI-Video-Edit 提出的 长时程自回归蒸馏（Long-Horizon Autoregressive Distillation, LHAD） ，巧妙地解决了长程稳定性与显存占用的矛盾。LHAD 采用了分段优化（Segmented Optimization）机制。"
arxiv_id: "2608.03974"
paper_published: "2026-08-04"
published_at: "2026-10-05T13:15:07.965576+08:00"
topics:
  - "多模态&视觉"
  - "模型优化"
tags:
  - "16B-parameter model"
  - "720p 30 FPS streaming editing"
  - "JoyAI-Video-Edit"
  - "Long-Horizon Autoregressive Distillation"
  - "SA-DMD"
  - "autoregressive diffusion"
related_tutorials:
  - "voyager-an-open-ended-embodied-agent-with-large-language-models"
  - "crayotter-learning-long-horizon-video-editing-agents-via-group-relative-preferen"
  - "mixture-of-contexts-for-long-video-generation"
  - "streamarena-toward-continuous-interactive-and-long-horizon-agentic-streaming-vid"
seo_title: "JoyAI-Video-Edit：单卡720p 30帧，两步蒸馏攻克流式视频编辑"
---

<p class="paper-original-title" lang="en">JoyAI-Video-Edit: Real-Time Open-Ended Video Editing with Autoregressive Diffusion</p>

<img src="/images/2608.03974v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在大模型视频生成的竞争进入白热化之后，视频编辑（Video Editing）正经历从“离线后期合成”向“实时流式交互”的关键范式转移。现存的顶尖商业系统（如 Runway、Kling 或各类前沿扩散模型）大都基于离线全局设计：模型需要预先加载整段视频，在双向全注意力和数十步去噪迭代下完成计算。这种架构在处理几秒到十几秒的固定片段时固然能保证全局一致性，但其计算量、显存开销随视频长度呈超线性膨胀，更致命的是它依赖“未来帧”，根本无法应用在直播推流、实时交互、音视频通话以及具身智能连续感知等必须即拍即得的场景中。

> ArXiv URL：https://arxiv.org/abs/2608.03974v1

流式视频编辑（Streaming Video Editing）的核心矛盾在于：如何在无法获知未来信息的前提下，维持严格因果输出与恒定的低延迟？而在长时间推演中，自回归模型不可避免地要吞入自己先前生成的非完美帧，微小的色彩偏差、形变误差会像滚雪球一样扩散，最终引发严重的“长时程累积漂移（Accumulated Temporal Drift）”与画面崩溃。

<img src="/images/2608.03974v1/teaserv7.webp" alt="JoyAI-Video-Edit 实时流式视频编辑示例" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

针对这一技术死结，Joy Future Academy 近期开源了参数量达 16B 的自回归扩散流式视频编辑框架 **JoyAI-Video-Edit**。该工作通过系统级的算法创新与全流程工程压榨，实现了在单张 Nvidia B200 GPU 上以大约 30 FPS（实际达到 30.19 FPS）的吞吐量，完成端到端 720p 的无上限长度实时视频编辑。它既无需依赖预设的视频总长度，也不受未来帧约束，同时在生成质量与指令遵循度上逼近顶尖的离线双向编辑模型。本文将系统拆解该系统的架构设计、核心训练机制及其工程实现。

### 从双向编辑到分块因果架构的蜕变

JoyAI-Video-Edit 的底层骨干由三部分组成：用于解析文本指令与多模态交互的多模态大语言模型（MLLM）条件编码器、因果时空视频 VAE，以及一个 16B 参数的多模态扩散 Transformer（MM-DiT）。整个系统不仅支持单纯的文本驱动视频编辑（V2V），还兼顾基于参考图像引导的多模态视频编辑（IV2V）。

将一个重型扩散模型改造成流式编辑器的第一步，是打破全局双向注意力的计算范式。传统的离线模型通过在整个时间轴上展开全自注意力来获取上下文，但这会导致两点缺陷：显存与算力随着帧数剧增，且无法在线流式解码。团队采取了“分块自回归适配（Chunk-wise Autoregressive Adaptation）”策略，将连续输入视频沿时间轴切分成离散对齐的源视频块（Source Chunk）与目标视频块（Target Chunk）。在当前实现中，每个 Chunk 对应潜在空间中的一个潜帧（对应像素空间的 8 帧）。

在注意力掩码设计上，模型采用了“块内双向、跨块因果”的混合注意力模式。在一个 Chunk 内部，所有的空间与时间 Token 允许双向交互，以保证单块画面的结构完整性与纹理锐利度；而在 Chunk 之间，注意力被严格施加因果约束，当前活动块只能检索当前和历史信息，对未来序列完全屏蔽。

<img src="/images/2608.03974v1/framework.webp" alt="JoyAI-Video-Edit 框架与三阶段训练流程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了让计算与显存开销彻底解耦于视频推流时长，模型引入了受限滑动窗口机制。每一个正在生成的活动块仅保留一个固定长度的最近历史 Chunk 窗口，同时额外将视频的“第一块”（First Chunk）作为全局汇聚槽（Global Sink）长久驻留。这个 Sink 机制如同一个持久的语义与外观锚点，防止自回归模型在推进数十秒乃至几分钟后彻底“遗忘”最初的主体特征。

更为关键的是训练阶段与推理阶段的分布对齐。传统的自回归模型在训练时普遍使用教师强迫（Teacher Forcing），即历史上下文使用的是完全真实、干净的目标视频帧。然而，在实际部署推流时，模型必须消耗自身在前一步生成的、带有轻微噪声和重建瑕疵的预测结果。这种“训练-推理不匹配”（Train-Inference Mismatch）正是自回归漂移的温床。JoyAI-Video-Edit 引入了重采样强迫（Resampling Forcing）策略，训练时不向历史槽位输入纯净目标，而是先通过单步去噪 Rollout 生成一个脱离梯度的自身估计值作为上下文历史。这一操作提前让网络适应推理阶段那类略带缺陷的历史输入分布，从根本上增强了模型对自身微小瑕疵的鲁棒性。

### SA-DMD：源锚定蒸馏破解两步生成难题

即使实现了因果流式计算，多步扩散迭代（如通常所需的 20 到 50 步采样）依然无法满足 30 FPS 的毫秒级实时交互要求。要做到真正的实时视频流处理，必须将去噪步数压缩至极限——2 步以内。现存的分布匹配蒸馏（DMD）方法通常利用模式寻优的逆向 KL 散度，指导学生网络用极少步数逼近多步教师网络的真实数据分布。然而，直接将通用 DMD 应用于因果自回归视频编辑，会引发致命的“源漂移（Source Drift）”：在多轮推演中，学生网络会过度依赖自己有偏差的生成历史，导致画面的主体身份、姿态或背景脱离输入的原始源视频，最终产生幻觉。

为了在极致压缩步数的同时锁死输入源信息，本文提出了**源锚定分布匹配蒸馏（Source-Anchored DMD, 简称 SA-DMD）**。该方法在蒸馏结构中包含了因果生成器学生模型 $G_{\theta}$、可训练的假分布评分模型 $F_{\psi}$，以及冻结的真实分布评分教师模型 $R_{\phi}$。三者共享主干 LoRA 权重，并从前序阶段初始化。

SA-DMD 的精妙之处在于对真实教师评分实施了“解耦式双轴无分类器引导（Classifier-Free Guidance, CFG）”。传统 CFG 仅沿文本轴放大条件强度，而 SA-DMD 将引导方向分解为独立的文本控制轴与源保真度控制轴：




{% raw %}$$ v_{\phi}^{g} = v_{\phi}^{\mathrm{cond}} + w_{\mathrm{txt}}\big(v_{\phi}^{\mathrm{cond}} - v_{\phi}^{-\mathrm{txt}}\big) + w_{\mathrm{src}}\big(v_{\phi}^{\mathrm{cond}} - v_{\phi}^{-\mathrm{src}}\big) $${% endraw %}



公式中，$v_{\phi}^{\mathrm{cond}}$ 为全条件下的速度场预测，$v_{\phi}^{-\mathrm{txt}}$ 为丢弃文本指令的无文本预测，而 $v_{\phi}^{-\mathrm{src}}$ 则是剔除了时间对齐源潜变量块 $S^k$ 的无源预测。超参数 $w_{\mathrm{src}}$ 成为了调节生成历史连续性与原始视频保真度之间博弈的关键天平。

通过这种显式引导，教师模型在向学生传递分布知识时，给出的目标是一个“源特征高度锐化”的后验分布。更关键的工程优势在于，**这种双重源引导仅作用于离线蒸馏的目标函数中，并完全被单步/两步生成器参数所吸收**。在部署推理时，系统不需要并行计算多个无条件分支，仅凭单次前向传递就能稳定保留未编辑区域的背景、几何与光影，从而兼顾了极致的低延迟与高保真度。

### 长时程自回归蒸馏：内存受限下的长程一致性

单靠短片段内的蒸馏并不能彻底解决长时间自回归漂移。许多误差往往在连续推演 30 到 50 个 Chunk 后才会显露出来，如果蒸馏训练仅局限于短序列，模型就永远无法学到如何从深层累积误差中纠偏。然而，直接拉长训练 Rollout 会迅速耗尽显存，导致反向传播计算图发生 OOM。

JoyAI-Video-Edit 提出的**长时程自回归蒸馏（Long-Horizon Autoregressive Distillation, LHAD）**，巧妙地解决了长程稳定性与显存占用的矛盾。

LHAD 采用了分段优化（Segmented Optimization）机制。在对超长序列进行推演时，系统将 $m$ 个 Chunk 的长时程 Rollout 切割为若干较短的连续片段。模型按片段顺序计算 SA-DMD 的反向传播梯度，在计算完单个片段的梯度后立刻释放当前激活图与计算图，仅在优化器层面积累跨片段的梯度，最后统一执行单次优化器权重更新。这一机制使得模型可以在恒定的、极为紧凑的显存预算内，直接接收来自超长时程状态的监督信号。

长程蒸馏面临的另一个现实问题是高保真长视频训练数据的稀缺。为了给长达几十甚至上百个 Chunk 的推演提供持续的源条件，团队开发了动态镜像回环（Dynamic Mirror Looping）策略。该方法在源视频到达尾部时，以交替的正向与反向时间序列拼接上下文。相比简单的周期性循环（从最后一帧硬跳回第一帧），动态镜像回环避免了语义和画面的突兀跳变，在不凭空增加显存占用的情况下，构建了平滑、连续的超长源信号，显著提升了流式模型在长视频中的耐受极限。

### 软硬件协同部署：单卡 30 FPS 的底层工程实现

算法层面的两步生成只是跨入实时门槛的前提，要将端到端延迟打入毫秒级，还必须对整个推理流水线实施系统级手术。

<img src="/images/2608.03974v1/pipeline.webp" alt="JoyAI-Video-Edit 在 B200 上的运行时性能剖析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从部署全貌来看，视频流以 8 帧为一个时间块切分送入系统。整套流水线经历了四层深度的系统加速：

1. **混合精度与算子级优化**：在整个 DiT 主干与 VAE 中全面引入 FP8 量化，搭配算子融合（Operator Fusion）与计算图编译（Graph Compilation），极大地压榨了新一代 GPU 的 Tensor Core 算力。

2. **因果 VAE 编解码优化**：因果视频 VAE 的自回归特性通常是瓶颈所在。团队通过专门编译的 VAE 执行路径与自适应调优，并利用伪编码器（Pseudo Encoder）提供单帧上下文，使得每 8 帧的编解码延迟大幅降低。

3. **KV Cache 恒定约束**：基于 Chunk-wise 注意力结构，生成完成后的干净 Key-Value 状态被即时写入缓存，由于始终保持“全局 Sink + 滑动局部窗口”，KV 显存占用与计算耗时在流式过程中保持严格水平线，不会因播放时间推移而下降。

4. **异步流水线重叠（Pipelined Overlapping）**：主机侧的视频帧预处理、数据拷贝，与设备侧的 DiT 两步去噪以及 VAE 解码被重叠在不同的执行流中，消除了所有空泡时间。

测试数据显示，在单张 Nvidia B200 GPU 上，JoyAI-Video-Edit 的端到端全链路吞吐达到了 30.19 FPS（对应分辨率为 $720 \times 1280$）。这一指标比闭源的流式竞品 XMax-X2.0（在更低分辨率下运行）还要快 44.4%，更是此前开源流式方案 SANA-Streaming 吞吐量的两倍以上。该系统首次证明了：**百亿参数规模的高画质扩散模型，完全可以在单卡环境下支撑高分辨率、真 30 帧的实时视频编辑**。

### 实验评测：流式性能全面领先，逼近离线工业旗舰

为了全面评测模型性能，作者团队不仅在开源的短视频基准 OpenVE-Bench 上进行了涵盖全局风格、局部修改、背景替换、物体增删五大维度的自动评测，还针对长程推演构建了全新的长视频评测集 **LongV2VBench**。基准对比覆盖了两大阵营：流式编辑器阵营（StreamDiffusionV2、SANA-Streaming、LiveEdit、XMax-X2.0）与离线旗舰阵营（包括 VACE、OpenVE-Edit、UniVideo 等开源方案，以及 PixVerse V6、Runway Aleph、Kling-3.0 Omni、Kling-O1 等工业界顶级系统）。

<img src="/images/2608.03974v1/GSB.webp" alt="人类偏好盲测对比结果" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

自动评估与人类盲测（GSB 评估）呈现出了高度一致的趋势：

在流式方法对比中，JoyAI-Video-Edit 展现出明显的断层式优势。此前的流式系统（如 StreamDiffusionV2 和 SANA-Streaming）往往为了追求帧率而牺牲了模型容量，使用极小尺寸的骨干网，导致它们在面对复杂指令（如精细的局部物体替换或风格迁移）时常出现语义失效和画面崩坏。在人类偏好盲测中，JoyAI-Video-Edit 相对于 SANA-Streaming 和 LiveEdit 取得了压倒性的胜率。

更为惊艳的是与离线重量级模型的横向较量。离线模型由于掌握全局时空信息且去噪步数充裕，通常被认为是视频质量的上限。但在 OpenVE-Bench 与 LongV2VBench 上，仅需两步去噪的 JoyAI-Video-Edit 表现出了令人意外的竞争力。尽管离线旗舰在极其复杂的长程物理运动模拟上仍微弱占优，但得益于 SA-DMD 带来的源锚定特性以及 LHAD 的防漂移控制，JoyAI-Video-Edit 在背景保留度、局部修改精准性以及长视频前后的时间一致性上，已经能够与诸多离线系统并驾齐驱，甚至在风格统一性得分上超过了部分两阶段离线方案。

消融实验进一步印证了各个模块的必要性：当剥离 SA-DMD 的源锚定项时，随着 Rollout 步数增加，人物面部特征与背景几何很快出现严重漂移；而如果去掉 LHAD 分段长时程蒸馏，模型在推演几十秒后便会陷入不可逆的色彩暗化与纹理模糊。只有三者结合，才能在极少步数与长时推演的双重压力下维持平衡。

### 范式重塑：视频交互从“渲染等待”迈向“即拍即得”

JoyAI-Video-Edit 的核心价值在于，它标志着视频大模型从“离线后期工具”向“在线交互基础设施”迈出了决定性的一步。

过去，AI 视频编辑更像是一种非实时的“盲盒渲染”：创作者输入提示词，提交任务至服务器队列，等待数分钟才能看到生成或修改后的样片；如果不满意，只能修改提示词并重新完整渲染一次。这种离散的工作流将创作者隔绝在反馈回路之外。

当单卡 720p 30 FPS 的自回归因果编辑成为现实，这种交互范式被彻底重构。用户可以在推流的同时即时调整提示词，画面可以在保持原视频姿态、动作与未动区域的前提下，毫秒级响应新的风格化渲染或元素增删。这不仅极大地缩短了数字内容创作的迭代周期，也为实时虚拟主播（Digital Humans）、直播特效无感渲染、沉浸式元宇宙视频通话乃至游戏画面的实时重绘带来了可落地的技术支点。

开源代码与全套架构的释放，为社区展示了如何通过 Chunk-wise 因果改造、解耦式分布匹配蒸馏与长程分段优化，驯服 16B 级别庞然大物的工程路径。流式扩散与自回归的结合，正在重新定义人机视觉交互的响应边界。
