---
layout: default
title: "VideoXAgent：告别离线预处理，纯在线长视频Agent将上下文降至15%"
description: "针对这一两难困境，百度团队在一项最新研究中提出了名为 VideoXAgent 的纯在线长视频智能体框架（Purely Online Video Agent Harness）。"
arxiv_id: "2609.12818"
paper_published: "2026-09-11"
published_at: "2026-09-15T13:15:08.126145+08:00"
topics:
  - "AI Agent"
  - "多模态&视觉"
tags:
  - "VideoXAgent"
  - "budget-aware control"
  - "context rot"
  - "expert tools"
  - "long video understanding"
  - "multimodal evidence aggregation"
related_tutorials:
  - "a-survey-on-agentic-multimodal-large-language-models"
  - "a-survey-on-multimodal-large-language-models"
  - "mixture-of-contexts-for-long-video-generation"
  - "budget-aware-tool-use-enables-effective-agent-scaling"
seo_title: "VideoXAgent：告别离线预处理，纯在线长视频Agent将上下文降至15%"
---

<p class="paper-original-title" lang="en">Online Video Agent Harness for Long Video Understanding</p>

<img src="/images/2609.12818/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

理解一部长达数小时的视频，对多模态大模型而言往往是一场“大海捞针”式的考验。真正回答用户提问的关键证据，通常只极其稀疏地分布在几个特定时间片段中；而为了捕捉这些线索，业界的传统做法要么是用高采样率把数千帧画面直接塞进上下文窗口，要么是耗费高昂算力预先将全片切片、解析并建立层次化检索索引。前者容易引发多模态大模型的上下文退化（Context Rot）与巨大的推理开销，后者则脱离了具体提问，极易在预处理阶段漏掉特定细节。

> ArXiv URL：https://arxiv.org/abs/2609.12818

针对这一两难困境，百度团队在一项最新研究中提出了名为 VideoXAgent 的纯在线长视频智能体框架（Purely Online Video Agent Harness）。该方案不依赖任何针对视频全片的离线预建库操作，而是从原始视频文件和具体 Query 出发，按需调用多模态异构专家工具，以渐进搜寻证据的方式解决长视频推理。在 Video-MME-Long、LVBench 和 MINERVA 等长视频基准测试中，VideoXAgent 在平均约 50k tokens 的轻量智能体上下文开销下，取得了比肩顶尖闭源原生多模态模型的成绩；在 MINERVA 复杂推理基准上，它仅消耗了 1,024 帧密集拼帧基线约 15% 的输入上下文。

这一研究揭示出一个关键趋势：解决超长视频理解的核心并不在于把模型上下文无节制地拉长到数百万 tokens，而在于如何通过严谨的智能体控制架构，将非结构化视频逐步拆解为按需验证的多模态证据链。

### 从“全量拼帧”与“离线建库”走向纯在线搜寻

长视频理解在模型层面上始终面临两类路线的天然缺陷。第一类是原生多模态长上下文模型（如 Qwen2.5/3-VL、Kimi-K2.5、Seed-2.0 以及 Gemini 系列）。这类模型通常依靠均匀降采样把画面铺满窗口，但只要采样间隔稍大，转瞬即逝的动作或细小文字就会丢失；如果拉高采样密度，大量无关背景帧又会干扰注意力分布，造成严重的上下文衰减。近期虽然出现了 VideoThinker、LongVT 等尝试分段探索的方案，但其底层仍然主要依赖单次上下文内的视觉帧堆叠，对语音、屏幕 OCR 等非视觉模态的结构化利用相对受限。

第二类则是以 DVD、HAVEN、DrVideo 为代表的视频 Agent 路线。为了避免把数小时的视频硬塞给单个模型，这类系统通常在接收用户 Query 之前，先运行一段庞大的离线预处理流水线：将视频按固定时长分段、抽取音频转录文本、提取关键帧特征，最后在数据库中构建多粒度索引或层次化树状记忆。这种设计的硬伤在于“提问不可知”——离线阶段的均匀采样和粗粒度总结很可能直接滤掉用户特定提问所需的微观线索；此外，如果用户只问一个简单问题，系统却不得不先预计算整部数小时的电影，在实际落地中成本极难承受。

<img src="/images/2609.12818/overview.webp" alt="两种视频智能体处理范式对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

如上图所示，VideoXAgent 彻底打破了先建库再查询的范式，确立了纯在线（Purely Online）的运作机制。系统在推理开始前只接收原始视频路径 $V$ 与用户自然语言提问 $Q$。主控 Agent 将输入视频视为一个黑盒，将其置于环境之中，自身则保持轻量状态，仅在需要时通过 FFmpeg 脚本实时切出目标片段或提取音轨，调用特定领域工具，完成“假设—检验—证据聚合”的闭环。

在形式化定义上，系统将视频推理表述为 $A = \operatorname{Agent}(V, Q, T, B)$，其中 $T = \{t_1, \ldots, t_M\}$ 为可用专家工具集合，$B$ 为执行预算。在第 $k$ 个推演步，Agent 维护当前状态 $s_k = (Q, H_k, b_k)$，由积累的历史动作与多模态证据 $H_k$ 及剩余预算 $b_k$ 共同驱动下一步决策。整个过程完全由 Query 条件引导，消除了脱离具体任务的无效计算。

### 基于人类专家轨迹的原子能力挖掘与工具空间

许多 Agent 系统的工具箱往往依赖开发者主观拍脑袋决定，极易产生工具功能重叠、调用边界模糊或关键能力缺失的问题。复杂视频问答往往交织着时序定位、空间局部放大、语音语义核验以及细粒度多步逻辑推演，传统的粗粒度任务标签无法指导细粒度工具的构建。

为了建立系统化、完备的工具体系，研究团队采取了数据驱动的反向挖掘方法。他们收集了人类专家在复杂长视频基准 MINERVA 上的真实解题思考轨迹，深入分析人类专家在面对复杂时序关联、长程因果推理和微小视觉目标时分解出的底层动作，进而归纳出一套原子能力分类法（Atomic Capability Taxonomy）。

<img src="/images/2609.12818/capability_mining.webp" alt="基于专家轨迹的原子能力挖掘与工具映射流水线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

通过这套自底向上的归纳流程，团队将长视频理解所需的底层动作划分为感知、定位、提取与高级推理等多重维度，并据此设计了包含 60 多个独立工具的异构工具库，横跨 14 种成熟算法后端：

- 视觉与目标级感知：集成 Grounding-DINO、DINO-X、SAM/SAM2 以及人脸分析模型，专门用于解决“某人在何处出现”或“微小物品在画面中的空间追踪”；

- 文本与语音多模态信号提取：集成 PaddleOCR 用于捕捉画面中微小的计分牌、街景招牌、文字提示，集成 WhisperX 与 PyAnnote 处理语音听翻及说话人日志；

- 时序概览与检索：依托轻量 CLIP 检索与定制 FFmpeg 抽帧脚本，实现自适应候选时间窗定位；

- 通用语义检验：采用小型或前沿 VLM（如 Gemini-3.1-Pro-preview）作为单次、短跨度的场景级检验单元。

这些工具对外暴露标准化的媒体作用域（Media Scope）、参数契约与结构化输出接口，底层的执行细节被统一抽象。即便工具返回空值或执行异常，报错信息也会作为明确观测反馈回上下文中，供 Agent 进行反思纠错，而不是直接导致调用崩溃。

### 严谨的智能体控制架构：抑制幻觉与控制死循环

当一个系统集成了数十个异构工具并运行于长时序环境下时，很容易暴露出 ReAct 架构的两个经典弊端：第一，模型在多轮交互中容易顺应自身提问产生主观臆断；第二，当某一步工具检索未返回理想结果时，Agent 容易陷入盲目重复调用的死循环。针对这两点，VideoXAgent 在控制层（Harness）中构建了多重防御屏障。

首先是针对多模态 VLM 工具的“客观证据提示契约”（Objective Visual Evidence Prompting）。在多步任务分解中，主控模型派生出的子问题往往自带先验假设（例如“找出红色小车右转时的车牌号”隐含了“小车必然右转”的前置设定）。普通的视觉模型很容易在模糊画面的诱导下“顺从”这一假设，产生虚假确证。

为了彻底阻断这种错误累积，所有接入系统的 VLM 工具都被强制置于统一的客观验证规范下：子问题仅被视作待验证的假说，模型只能严格基于当前输入的图像帧或局部裁剪块作答。更重要的是，工具的返回体被严格格式化为三个分离字段：直接观测到的视觉证据（visual_evidence）、基于证据的判定结论（answer）以及不确定性说明（uncertainty）。辅助性的 OCR 或检测结果只能作为参考上下文，严禁与实际可见的视觉事实混淆。如果画面分辨率不足以辨认微小文字或动作意图，工具必须显式声明局限性，使主控模型能够有据可依地在多源证据冲突时进行权衡与二次验证。

其次是面向执行轨迹的“预算感知中间件控制”（Budget-aware Control）。长视频交互中，低质量的规划会导致无效调用，进而让冗余观测污染上下文，使模型失去长程推理方向。VideoXAgent 在消息流底层部署了轻量级中间件，执行两道硬性约束：

1. 硬步数上限与强制截断作答：单条轨迹最多允许执行 $L=125$ 个节点步骤（涵盖主控模型调用、工具请求与工具执行，折合大约 42 次工具调用交互）。当剩余预算触底时，中间件强制将 tool_choice 锁定为 none，强行要求主控模型基于已有证据输出文本答案，彻底避免因无休止探索导致的“超时无输出”。

2. 阶梯式递增预警：统计显示大部分成功样本在消耗 $65\%$ 预算内即可收敛。中间件在步数消耗达到 $65\%$、$80\%$ 和 $90\%$ 三个阈值时，会分别注入单次严重程度递增的系统级提醒，明确指示剩余额度，敦促 Agent 放弃边缘线索的重复验证、通过少数服从多数原则处理冲突，并尽快聚合现有证据收尾。

在路径路由方面，框架设计了自适应的长短视频双轨策略。针对短视频或能紧凑容纳的关键片段，优先采用 VLM 宽视野直接推演，仅在遭遇生僻字、细小人脸或密集动作关系时外挂专用工具；针对超长视频，则遵循由粗到细（Coarse-to-fine）的漏斗路径，先利用转录文本、字幕或稀疏场景摘要构建全局时序草图，定位候选区间，再下沉到特定片段进行局部密集采样与多模态专项透视，杜绝了无端遍历全片的计算浪费。

### 实验结果与效率表现

研究团队在五个极具代表性的长视频理解评测集上进行了全面检验，涵盖 LongVideoBench-Long、Video-MME-Long、LVBench、MINERVA 以及 Video-MME-v2。主控模型主要采用在复杂编排与工具调度上表现突出的 Claude-Opus-4.6，VLM 级工具后端则挂载 Gemini-3.1-Pro-preview。

下表系统展示了 VideoXAgent 与当前代表性多模态长上下文模型及视频 Agent 的能力对比：


| 方法 | 类别 | LVBench (Visual-only) | Video-MME-Long (All) | MINERVA |
| :--- | :--- | :--- | :--- | :--- |
| GPT-4o (2024-11-20) | 原生 LMM | 65.3 | 77.2 | 57.0 |
| Gemini-2.5-Pro | 原生 LMM | 66.8 | 83.1 | 59.8 |
| Gemini-3.1-Pro | 原生 LMM | 73.8 | 84.8 | 61.3 |
| Seed-2.0-pro | 原生 LMM | 73.1 | 82.5 | 62.4 |
| HAVEN | 离线 Agent | 71.7* | 75.3 | - |
| VideoSeek | Agent | 65.8 | - | 47.9 |
| **VideoXAgent** | **纯在线 Agent** | **76.5** | **85.1** | **65.7** |

注：HAVEN 的 LVBench 得分为原论文中移除字幕和音频的视觉单模态消融成绩；在 Video-MME-Long 上，VideoXAgent 取得了 $85.1\%$ 的准确率，在纯视觉评测集 LVBench 上达到 $76.5\%$，在深度推理基准 MINERVA 上达到 $65.7\%$，全面超越了此前依赖全片离线预索引的 HAVEN 与 VideoSeek，并在多个维度上优于或紧咬当前最强的闭源原生超长上下文多模态模型。

<img src="/images/2609.12818/lvbench_accuracy_comparison.webp" alt="LVBench基准准确率对比与长视频准确率/上下文开销权衡" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

<img src="/images/2609.12818/tradeoff_plot.webp" alt="上下文开销对比图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从上方的效率权衡图中可以看出，性能的突破并没有建立在无节制消耗计算资源的基础上。原生多模态大模型在处理一小时以上的视频时，为了维持全局视野通常需要密集抽帧并喂入几十万乃至上百万的上下文 token。而 VideoXAgent 在一个小时量级的样本中，平均每条轨迹仅消耗大约 50k tokens 的 Agent 上下文开销。在 MINERVA 这一强调微观多步推理的测试集中，系统达成 $65.7\%$ 准确率所使用的上下文，仅为传统 1,024 帧密集全采样基线的约 $15\%$。这充分验证了“按需在线搜集证据”相比于“全量帧前向传播”在计算经济性上的巨大优势。

更具启发性的是消融实验中的一项关键发现：当主控模型换用多模态视觉理解能力较弱的模型、甚至换用纯文本语言模型来担当调度大脑时，借助这套规范化的 Harness 框架与异构专家工具，系统依然能够在长视频基准上交出令人惊讶的高分。这从实证角度说明，长视频推理能力的突破口并不完全绑定在端到端视觉大模型的单次前向上下文容量上；只要具备严谨的推理规划、合理的假设验证机制以及细粒度的专业工具调度能力，强大的视频理解水平同样可以从智能体演进的多模态探索过程中自发涌现。

### 总结与展望

VideoXAgent 的研究给当下的多模态计算范式提供了一个极具参考价值的视角。面对信息密度极低、时空跨度极大的长视频数据，盲目扩张单体模型的上下文长度并非唯一解，更不一定是性价比最高的方案。脱胎于专家真实轨迹的原子能力工具库，配合包含客观证据约束与步数预算管控的严谨 Harness，使得系统在完全摒弃离线切片预处理的前提下，以小得多的上下文代价跑通了高难度的长视频推演。

这种纯在线、轻量级的智能体范式，不仅极大降低了长视频问答系统的推理延迟与全片预建库的存储维护成本，也为更广泛的实时音视频理解、自主长程监控分析以及复杂多模态 Agent 的工程化落地指明了切实可行的新路径。
