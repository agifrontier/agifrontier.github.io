---
layout: default
title: "LLaVA-OneVision: Easy Visual Task Transfer"
---

# LLaVA-OneVision: Easy Visual Task Transfer

- **ArXiv URL**: http://arxiv.org/abs/2408.03326v3

- **作者**: Hao Zhang; Dong Guo; Feng Li; Kaichen Zhang; Chunyuan Li; Yanwei Li; Renrui Zhang; Ziwei Liu; Yuanhan Zhang; Bo Li

- **发布机构**: ByteDance; HKUST; NTU; The Chinese University of Hong Kong

---

# TL;DR
本文提出了 LLaVA-OneVision，一个系列化的开放大型多模态模型 (Large Multimodal Models, LMMs)，它通过整合数据、模型和视觉表示的洞见，成为首个能够在单图像、多图像和视频这三个核心视觉场景下，同时达到顶尖性能的单一开源模型，并展示了通过任务迁移实现的新兴能力。

# 关键定义
*   **LLaVA-OneVision**: 本文提出的一个开放大型多模态模型家族。其核心设计理念是构建一个单一、通用的模型，能够同时在单图像、多图像和视频理解任务中取得卓越表现，打破了现有开源模型通常只专精于单一场景的局限。
*   **Higher AnyRes**: 一种处理高分辨率图像的灵活视觉表示策略。它将任意分辨率的图像分割成多个固定分辨率的图像块（crops），并为每个图像块提取视觉特征。当总的视觉 token 数量超过预设阈值时，它会通过双线性插值（bilinear interpolation）对每个图像块的特征图进行降采样，以在保持高分辨率信息的同时，有效控制计算成本。
*   **OneVision 训练 (OneVision Training)**: 一个关键的训练阶段，旨在将模型从单图像能力扩展到多场景能力。在完成了大规模单图像指令微调后，模型会进入此阶段，在一个混合了单图像、多图像和视频的数据集上进行联合训练。这使得模型能够学习跨不同场景的通用能力，并促进知识从数据丰富的图像任务迁移到数据相对稀疏的多图像和视频任务。
*   **任务迁移 (Task Transfer)**: 本文强调的一种新兴能力，指模型将在某一特定模态（如单图像）上学到的知识和技能，成功应用于其他模态（如视频或多图像）的任务中。例如，通过将图像表示为类似视频的 token 序列，模型能够零样本（zero-shot）地将图像理解能力迁移到视频理解上。

# 相关工作
当前，最先进的私有大型多模态模型（如 GPT-4V/o、Gemini）已经展现出在单图像、多图像和视频等多种视觉场景下的通用处理能力。然而，在开源社区，现有工作通常是为每个场景单独开发专用模型。大多数研究致力于提升单图像任务的性能，少数工作开始探索多图像场景，而视频 LMMs 虽然在视频理解上表现出色，却常常以牺牲图像处理性能为代价。

因此，领域内存在一个明显的空白：缺乏一个单一的、开源的、能够在上述所有三个核心视觉场景中都表现出色的通用模型。

本文旨在解决这一问题，通过提出 LLaVA-OneVision，构建一个能在多种视觉任务中达到顶尖水平的单一模型，并探索通过跨场景任务迁移来催生新的通用能力。

# 本文方法

## 模型架构
本文模型继承了 LLaVA 系列的极简设计哲学，旨在有效利用预训练的语言模型和视觉模型的现有能力，并支持在数据和模型规模上的良好扩展性。

<img src="/images/2408.03326v3/x1.jpg" alt="LLaVA-OneVision 网络架构。左：当前模型实例；右：LLaVA的通用架构形式，但已扩展以支持更多视觉信号。" style="width:85%; max-width:600px; margin:auto; display:block;">

其核心组件包括：
*   **语言模型 (LLM)**: 选择 Qwen-2 作为基础语言模型 $f\_{\boldsymbol{\phi}}(\cdot)$，因为它提供了多种模型规模，并且在当前开源模型中具有强大的语言能力。
*   **视觉编码器 (Vision Encoder)**: 采用 SigLIP 模型作为视觉编码器 $g\_{\boldsymbol{\psi}}(\cdot)$，它将输入图像 ${{\bf X}}\_{\texttt{v}}$ 编码为视觉特征 ${\bf Z}\_{\texttt{v}}=g({{\bf X}}\_{\texttt{v}})$。
*   **投影器 (Projector)**: 使用一个简单的两层 MLP 网络 $p\_{\boldsymbol{\theta}}(\cdot)$，将视觉编码器输出的图像特征投影到语言模型的词嵌入空间，生成一系列视觉 tokens ${\bf H}\_{\texttt{v}}=p({\bf Z}\_{\texttt{v}})$。

模型的选择基于团队之前的经验：更强的 LLM 通常能带来更强的多模态能力，而 SigLIP 在开源视觉编码器中能带来更高的 LMM 性能。对于一个给定的视觉输入 ${{\bf X}}\_{\texttt{v}}$ 和指令 ${{\bf X}}\_{\texttt{q}}$，模型生成答案 ${{\bf X}}\_{\texttt{a}}$ 的概率计算如下：


{% raw %}$$
p({{\bf X}}\_{\texttt{a}} \mid {{\bf X}}\_{\texttt{v}},{{\bf X}}\_{\texttt{q}})=\prod\_{i=1}^{L}p(\boldsymbol{x}\_{i} \mid {{\bf X}}\_{\texttt{v}},{{\bf X}}\_{\texttt{q},<i},{{\bf X}}\_{\texttt{a},<i})
$${% endraw %}


此处的视觉输入 ${{\bf X}}\_{\texttt{v}}$ 是一个通用形式，可以代表单图像、多图像序列中的单个图像，或视频序列中的单个帧。

## 视觉表示
视觉信号的表示方式是模型成功的关键，它涉及原始像素空间的分辨率和特征空间的 token 数量两个因素。为了在性能和成本之间取得平衡，本文采用了 **Higher AnyRes** 策略。

<img src="/images/2408.03326v3/x2.jpg" alt="视觉表示方法。上：新的 Higher AnyRes 方案，通过双线性插值处理更高分辨率的图像；下：LLaVA-NeXT 中的原始 AnyRes 方案。" style="width:85%; max-width:450px; margin:auto; display:block;">

该策略将高分辨率图像分割成 $a \times b$ 个图像块，每个图像块都以视觉编码器适合的分辨率进行处理。如果生成的总 token 数 $L$ 超过了设定的阈值 $\tau$，则会通过双线性插值减少每个图像块的 token 数量 $T\_{\text{new}}$：


{% raw %}$$
T\_{\text{new}}=\begin{cases}\frac{\tau}{(a\times b+1)}&\text{if }L>\tau\\ T&\text{if }L\leq\tau\end{cases}
$${% endraw %}


这种灵活的表示框架被应用于不同场景，并设计了相似的最大视觉 token 数量，以促进跨场景的能力迁移。

<img src="/images/2408.03326v3/x3.jpg" alt="LLaVA-OneVision 中为每个场景分配 token 的视觉表示策略。不同场景的最大视觉 token 数量被设计为相似的，确保了平衡的视觉表示，以适应跨场景能力迁移。注：729 是 SigLIP 编码一个 384×384 视觉输入的 token 数量。" style="width:90%; max-width:700px; margin:auto; display:block;">

具体策略如下：
*   **单图像**: 为了促进向视频任务的能力迁移，单张图像被分配大量的视觉 tokens，通过模拟视频的长序列表示方式来编码。
*   **多图像**: 为节省计算资源，仅使用基础分辨率处理每张图像，不进行高分辨率分块。
*   **视频**: 每个视频帧被缩放到基础分辨率进行编码，然后通过双线性插值减少每帧的 token 数量，从而能在有限的计算预算内处理更多的帧。

## 数据策略
本文强调在 LMM 训练中“质量优于数量”的原则，并将数据分为两类进行精心构建。

### 高质量知识学习
为高效地向 LMM 注入新知识，本文专注于使用高质量数据进行学习。这些数据几乎全部（99.8%）是合成的，主要来自三个方面：
*   **重标注的详细描述数据 (3.5M)**: 使用早期的 LLaVA-NeXT-34B 模型为 COCO、BLIP 等数据集的图像生成新的详细描述，实现了一种简单的自提升。
*   **文档/OCR 数据 (1.1M)**: 利用 UReader 和 SynDOG 等数据集，专注于提升模型的文本阅读和光学字符识别（OCR）能力。
*   **中文和语言数据**: 包含使用 GPT-4V 生成的 9.2 万条中文详细描述数据，以及 14.3 万条来自 Evo-Instruct 的数据，以增强模型的中文能力和通用语言理解能力。

### 视觉指令微调
视觉指令微调数据对 LMM 的能力至关重要。本文从视觉输入、语言指令和语言响应三个层次对数据进行分类和整理。

基于前期研究的洞见（更强的图像模型能更好地迁移到视频任务），本文将指令数据分为两组进行训练：
*   **单图像数据 (3.2M)**: 这是一个精心挑选和平衡的数据集合，汇集了来自学术界和社区的多个高质量数据集，涵盖了通用问答、文档/图表、数学推理、OCR 和语言等多个类别。该数据集用于模型的第一阶段微调。

<img src="/images/2408.03326v3/x4.jpg" alt="单图像数据集合 (3.2M)。左：各类别数据分布。外圈为大类分布，内圈为子集分布。右：各数据集的详细数量。" style="width:80%; max-width:300px; margin:auto; display:block;">


| 类别 (占比) | 数据集详情 |
| :--- | :--- |
| **通用 (36.1%)** | ALLaVA Inst (70.0K), AOKVQA (66.2K), Cambrian (filtered) (83.1K), CLEVR (0.7K), COCO Caption (20.0K), Hateful Memes (8.5K), IconQA (2.5K), Image Textualization (99.6K), LLaVA-158K (158.0K), LLaVA-Wild (train) (54.5K), LLaVAR (20.0K), OKVQA (9.0K), RefCOCO (50.6K), ScienceQA (5.0K), ShareGPT4o (57.3K), ShareGPT4V (91.0K), ST-VQA (17.2K), TallyQA (9.9K), Vision FLAN (186.1K), Visual7W (14.4K), VisText (10.0K), VizWiz (6.6K), VQARAD (0.3K), VQAv2 (82.8K), VSR (2.2K), WebSight (10.0K), InterGPS (1.3K) |
| **文档/图表/屏幕 (20.6%)** | AI2D (GPT4V) (4.9K), AI2D (InternVL) (12.4K), AI2D (Original) (3.2K), Chart2Text (27.0K), ChartQA (18.3K), Diagram Image2Text (0.3K), Doc-VQA (10.2K), DVQA (20.0K), FigureQA (1.0K), HiTab (2.5K), Infographic VQA (4.4K), LRV Chart (1.8K), RoBUT SQA (8.5K), RoBUT WikiSQL (75.0K), RoBUT WTQ (38.2K), Screen2Words (15.7K), TQA (1.4K), UReader Caption (91.4K), UReader IE (17.3K), UReader KG (37.6K), UReader QA (252.9K), VisualMRC (3.0K) |
| **数学/推理 (20.1%)** | MAVIS MCollect (87.4K), MAVIS Data Engine (100.0K), Geo170K QA (67.8K), Geometry3K (2.1K), GEOS (0.5K), Geometry3K (MathV360K) (9.7K), GeoMVerse (MathV360K) (9.3K), GeoQA+ (MathV360K) (17.2K), MapQA (MathV360K) (5.2K), CLEVR-Math (5.3K), Geo170K Align (60.3K), MathQA (29.8K), Super-CLEVR (8.7K), TabMWP (45.2K), UniGeo (12.0K), GQA (72.1K), LRV Normal (10.5K), RAVEN (2.1K), Visual Genome (86.4K) |
| **通用 OCR (8.9%)** | ChromeWriting (8.8K), HME100K (74.5K), IIIT5K (2.0K), IAM (5.7K), K12 Printing (12.8K), OCR-VQA (80.0K), Rendered Text (10.0K), SynthDog-EN (40.1K), TextCaps (21.9K), TextOCR (25.1K) |
| **语言 (14.3%)** | Magpie Pro (L3 MT) (150.0K), Magpie Pro (L3 ST) (150.0K), Magpie Pro (Qwen2 ST) (150.0K) |

*   **OneVision 数据 (1.6M)**: 在单图像训练之后，使用这个混合数据集进行进一步微调。该数据集包含 80 万单图像样本（从上一阶段高质量数据中采样）、56 万多图像数据和 35 万视频数据。此阶段的目标是赋予模型处理多图像和视频的能力，并促进跨场景的知识迁移。

<img src="/images/2408.03326v3/x5.jpg" alt="OneVision 数据集合 (1.6M)。这是一个高质量的单图像、多图像和视频数据集合。左：各类别数据分布。右：各数据集的详细数量。“MI”表示该数据集是 DEMON 提出的多图像版本。" style="width:85%; max-width:450px; margin:auto; display:block;">


| 类别 (占比) | 数据集详情 |
| :--- | :--- |
| **单图像 (31.2%)** | Magpie Pro (90.0K), Vision FLAN (filtered) (55.8K), Image Textualization (49.8K), Cauldron (40.2K), UReader (39.9K), ShareGPT4V (21.0K), ALLaVA Inst. (21.0K), Cambrian (filtered GPT4o) (24.9K), 等... |
| **多图像 (43.0%)** | NLVR (86.4K), Co-Instruct (50.0K), ScanNet (49.9K), RAVEN (35.0K), IconQA (34.6K), VIST (26.0K), ScanQA (25.6K), ContrastiveCaption (25.2K), ALFRED (22.6K), FlintstonesSV (22.3K), 等... |
| **视频 (25.9%)** | ActivityNet (6.5K), Charades (23.6K), Ego4D (0.8K), NextQA (9.5K), ShareGPT4Video (255.0K), Youcook2 (41.9K) |

## 训练策略
本文采用课程学习（curriculum learning）的原则，分阶段地对模型进行训练，逐步增加任务难度。

<img src="/images/2408.03326v3/x6.jpg" alt="table1" style="width:90%; max-width:700px; margin:auto; display:block;">


| | **Stage-1** | **Stage-1.5** | **Stage-2 (Single-Image)** | **Stage-2 (OneVision)** |
| :--- | :--- | :--- | :--- | :--- |
| **目标** | 语言-图像对齐 | 高质量知识学习 | 单图像指令微调 | 多场景能力扩展 |
| **视觉-分辨率** | 384 | 384, AnyRes (多尺度) | 384, AnyRes (多尺度) | 384, AnyRes (多尺度) |
| **视觉-#Tokens** | 729 | 最大 729×5 | 最大 729×10 | 最大 729×10 |
| **数据** | LCS (558K) | 知识数据 (4M) | 单图像指令数据 (3.2M) | OneVision 混合数据 (1.6M) |
| **可训练模块** | 仅投影器 | 全模型 | 全模型 | 全模型 |
| **学习率 (LR)** | Proj/LLM: 1e-3 | Vision: 2e-6, Proj/LLM: 1e-5 | Vision: 2e-6, Proj/LLM: 1e-5 | Vision: 2e-6, Proj/LLM: 1e-5 |
| **Epoch** | 1 | 1 | 1 | 1 |

*   **Stage-1: 语言-图像对齐**: 目标是将视觉特征与 LLM 的词嵌入空间对齐。此阶段仅训练投影器，使用较低的图像分辨率和固定的 token 数。
*   **Stage-1.5: 高质量知识学习**: 在此阶段开始全模型微调，引入精心策划的高质量知识数据，同时开始使用 AnyRes 处理更高分辨率的图像，增加视觉 token 数量。
*   **Stage-2: 视觉指令微调**: 这是能力养成的核心阶段，分为两步：
    1.  **单图像训练**: 使用 3.2M 单图像指令数据进行训练，使模型在单图像任务上获得强大的、遵循多样化指令的能力。此阶段使用更高的分辨率和更多的视觉 tokens。
    2.  **OneVision 训练**: 在上一步的基础上，使用包含单图像、多图像和视频的 1.6M 混合数据进行训练。此阶段旨在将模型的能力从单图像场景泛化到多场景，并促进新兴的跨场景迁移能力。

整个训练过程遵循渐进式原则，逐步增加序列长度、图像分辨率和视觉 token 数量，并逐步开放更多模块进行训练。视觉编码器的学习率被设置为 LLM 的 1/5，以更好地保留其预训练知识。

# 实验结论
本文对 LLaVA-OneVision 模型家族（0.5B, 7B, 72B）在单图像、多图像和视频三大类基准上进行了全面评估。


| 能力 | 基准 | LLaVA-OneVision-7B | LLaVA-OneVision-72B | GPT-4V | GPT-4o |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **单图像** | DocVQA | 87.5% | 91.3% | 88.4% | 92.8% |
| | MathVista | 63.2% | 67.5% | 49.9% | 63.8% |
| | MMBench | 80.8% | 85.9% | 75.0% | - |
| | MMMU (val) | 48.8% | 56.8% | 56.8% | 69.1% |
| | LLaVA-Wilder (small) | 67.8% | 72.0% | 81.0% | 85.9% |
| **多图像** | LLaVA-Interleave | 64.2% | 79.9% | 60.3% | - |
| | MuirBench | 41.8% | 54.8% | 62.3% | - |
| | Mantis | 64.2% | 77.6% | 62.7% | - |
| **视频** | ActivityNetQA | 56.6% | 62.3% | 57.0% | - |
| | MLVU | 64.7% | 68.0% | 49.2% | 64.6% |
| | VideoMME | 58.2% | 66.2% | 59.9% | 71.9% |

**关键结果总结如下：**

*   **通用性与高性能**: 实验结果验证了 LLaVA-OneVision 作为一个单一模型在三大视觉场景中同时取得 SOTA 性能的能力。其最大的 72B 模型在多个基准上的表现超越了 GPT-4V，并接近 GPT-4o，证明了本文提出的数据、模型和训练策略的有效性。

*   **单图像能力**: 在如图表、文档理解（DocVQA, ChartQA）和数学推理（MathVista）等需要精细感知的任务上，LLaVA-OneVision-72B 表现优异，显著超过 GPT-4V。但在更复杂的开放式真实世界聊天场景（如 LLaVA-Wilder）中，与 GPT-4o 相比仍有差距，这可能需要更强的基础 LLM 和更好的偏好学习。

*   **多图像能力**: 实验清晰地展示了 "OneVision 训练" 阶段的价值。经过该阶段的混合数据训练后，模型在多图像任务（如 LLaVA-Interleave Bench, Mantis）上的性能得到显著提升，甚至在一些子任务上大幅超越 GPT-4V，尤其是在多图像推理、差异识别和 3D 环境理解方面。

*   **视频理解能力**: 尽管是基于一个强大的图像模型迁移而来，LLaVA-OneVision 在多个视频理解基准（如 MLVU, ActivityNetQA）上也取得了极具竞争力的成绩，甚至超过了 GPT-4V，这有力地证明了从图像到视频的能力迁移是成功的。

**最终结论**：本文提出的 LLaVA-OneVision 框架，通过其极简架构、灵活的 Higher AnyRes 视觉表示、高质量的精选数据和分阶段的课程学习策略，成功地构建了一个在单图像、多图像和视频任务上均表现卓越的单一开源模型。该模型家族（特别是 72B 版本）的强大性能揭示了一条通过精心设计的数据和训练策略来打造通用视觉智能体的有效路径，为开源社区追赶乃至超越顶尖私有模型提供了宝贵的经验。