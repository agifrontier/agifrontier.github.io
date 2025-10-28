---
layout: default
title: "A Comprehensive Dataset for Human vs. AI Generated Text Detection"
---

# A Comprehensive Dataset for Human vs. AI Generated Text Detection

- **ArXiv URL**: http://arxiv.org/abs/2510.22874v1

- **作者**: Gurpreet Singh; Ritvik Garimella; Subhankar Ghosh; Aman Chadha; Shreyas Dixit; Vipula Rawte; Shwetangshu Biswas; Shashwat Bajpai; Amit Sheth; Nasrin Imanpour; 等16人

- **发布机构**: Amazon AI; BITS Pilani Hyderabad Campus; Birla Institute of Technology and Science Pilani; Gandhi Institute for Technological Advancement; Indian Institute of Information Technology Guwahati; Indraprastha Institute of Information Technology Delhi; Kalyani Government Engineering College; Meta AI; National Institute of Technology Silchar; San Jose State University; University of California Los Angeles; University of South Carolina; Vishwakarma Institute of Information Technology; Washington State University

---

# TL;DR
本文发布了一个包含超过58000条文本的大型数据集，其中包含真实的《纽约时报》文章以及由六种先进大语言模型（LLMs）生成的对应版本，并为区分人类与AI文本以及AI文本模型溯源这两个任务提供了基准性能。

# 关键定义
本文主要沿用并扩展了现有研究中的核心概念，关键定义如下：

1.  **AI生成文本检测 (AI-Generated Text Detection)**: 指识别并区分由人工智能（特别是大语言模型）生成的文本与人类创作的文本的任务。这是本文数据集旨在解决的核心问题。
2.  **模型溯源 (Model Attribution)**: 一个更细分的任务，旨在不仅检测出文本是AI生成的，还要进一步确定它是由哪个具体的大语言模型（如GPT-4o, LLaMA-8B等）生成的。
3.  **基于重写的检测 (Rewriting-based Detection)**: 本文基线方法所采用的一种检测策略。其核心假设是：相比于重写人类原创文本，大语言模型在重写由其他AI模型（尤其是其自身）生成的文本时，所做的修改会显著更少。通过衡量这种重写过程中的“编辑距离”差异，可以推断文本的来源。

# 相关工作
当前研究主要集中在AI生成文本检测和虚假新闻检测两个领域，但现有数据集存在明显不足。

在AI生成文本检测方面，虽然已有一些大型数据集（如包含600万样本的[12]和多语言数据集[13]），但它们大多依赖合成提示、学生作文或通用网络内容，缺乏高质量、真实世界的新闻语料。此外，很少有数据集系统性地在受控条件下比较多种顶尖LLM的输出。

在虚假新闻检测方面，现有数据集（如LIAR, FakeNewsNet）虽然对研究内容真实性很有价值，但它们通常关注的是简短的声明或社交媒体帖子，而非完整的、长篇的文章。

本文旨在解决上述缺陷，通过构建一个基于高质量新闻来源（《纽约时报》）的数据集来填补空白。该数据集不仅包含完整的、由人类撰写的文章，还系统地包含了由多种当前最先进的LLM生成的对应文本，从而为开发更鲁棒的检测和溯源方法提供了坚实的基础。

# 本文方法
本文的核心贡献是创建了一个新的数据集，并在此基础上建立了一个基线检测方法。

## 数据集构建

### 数据来源
数据集的基础是《纽约时报》（NYT）自2000年1月1日至今超过210万篇文章的集合。这些文章涵盖了广泛的主题，并包含丰富的元数据，如摘要、标题、关键词、出版日期等。

### 生成过程
1.  **摘要作为提示 (Abstract as Prompt)**: 提取每篇NYT文章的官方摘要，作为生成AI文本的统一提示（Prompt）。这确保了AI生成的内容与原始人类文章主题一致，具有可比性。
2.  **提取人类文本**: 通过文章的URL获取完整的人类撰写的叙事文本，称为“human story”。
3.  **生成AI文本**: 使用文章摘要作为提示，调用六种当前先进的LLM生成对应的文本。这些模型包括：Gemma-2-9b, Mistral-7B, Qwen-2-72B, LLaMA-8B, Yi-Large, และ GPT-4-o。

### 数据集结构与统计
最终的数据集以表格形式组织，包含原始摘要、人类撰写的全文以及由上述六个LLM分别生成的文本。

数据集总共包含超过58,000篇真实和合成的文章。详细的分布如下表所示，每个LLM贡献了大致相等数量的文章。


| 来源 | 数量 |
| :--- | :--- |
| Prompt | 7321 |
| Human\_story | 7295 |
| Gemma-2-9b | 7310 |
| Mistral-7B | 7316 |
| Qwen-2-72B | 7314 |
| LLaMA-8B | 7306 |
| Yi-Large | 7319 |
| GPT\_4-o | 7321 |
| **总计** | **58502** |

数据集可在 <https://huggingface.co/datasets/gsingh1-py/train> 获取。

下图分别展示了人类撰写文本和所有LLM生成文本组合后的词云图，可以看出“new york”、“united state”等关键词在两者中都很突出。

<img src="/images/2510.22874v1/wordcloud_Human_story.jpg" alt="Refer to caption" style="width:85%; max-width:450px; margin:auto; display:block;">

<img src="/images/2510.22874v1/wordcloud_combined_all_llms.jpg" alt="Refer to caption" style="width:85%; max-width:450px; margin:auto; display:block;">

### 数据集用途
该数据集能够支持多项研究，包括：
*   构建和评估区分人类与AI文本的模型。
*   研究AI生成文本特有的语言、风格和语义特征。
*   测试检测器在不同LLM之间的泛化能力。
*   为AI内容检测算法提供基准测试。
*   助力开发用于验证新闻内容真实性、抵御AI滥用的工具。

## 基线方法
本文实现了一个基于分类的基线方法，其灵感来源于Raidar方法 [20]，该方法通过“重写”来检测机器生成的内容。

### 创新点
该方法的核心思想是：一个LLM在被要求重写（Rewriting）一段文本时，如果该文本最初是由另一个LLM（尤其是它自己）生成的，那么它所做的修改会比重写人类撰写的文本要少。这种修改程度的差异可以量化并用于判断文本来源。

### 方法描述
1.  **重写**: 使用一个固定的重写模型（如GPT-3.5-Turbo）和一个标准化的指令（如“为我精简这段话并保留所有信息”）来重写输入的待测文本。
2.  **计算编辑距离**: 对于每个候选的源LLM，都让其执行重写操作，并使用莱文斯坦距离（Levenshtein metric）计算原始文本和重写后文本之间的编辑距离。
3.  **分类判决**:
    *   **模型溯源**: 产生最小编辑距离的那个候选LLM被预测为该文本的生成模型。
    *   **人 vs. AI**: 如果所有候选LLM产生的编辑距离都超过一个预设的阈值，则该文本被分类为人类撰写。

下图直观展示了这一思路：重写人类文本（上）会产生大量修改（高亮部分），而重写AI文本（下）产生的修改则少得多。

<img src="/images/2510.22874v1/x1.jpg" alt="[Uncaptioned image]" style="width:90%; max-width:700px; margin:auto; display:block;">

# 实验结论
本文在构建的数据集上定义了两个任务，并使用上述基线方法进行了测试。

*   **任务A：人类 vs. AI生成文本分类**：区分文本是由人类还是AI撰写的。
*   **任务B：AI生成文本的模型溯源**：识别出AI文本具体是由哪个模型生成的。

基线方法的性能结果如下表所示：


| 任务 | 描述 | 准确率 (Accuracy) |
| :--- | :--- | :--- |
| 任务 A | 人类 vs. AI生成文本分类 | 0.5835 |
| 任务 B | AI生成文本的模型溯源 | 0.0892 |

*   **结果分析**:
    *   在区分人类与AI文本的任务上，基线方法取得了58.35%的准确率，略高于随机猜测，但效果并不理想。
    *   在模型溯源任务上，准确率仅为8.92%，表明这项任务极具挑战性。
*   **最终结论**:
    *   本文成功构建并发布了一个高质量、大规模且具有挑战性的数据集，用于推进AI生成文本检测的研究。
    *   基线实验的低分结果凸显了这两个任务的 inherent 难度，尤其是在面对当今先进的LLM时，这表明该领域存在巨大的方法创新空间。
    *   未来的工作可以包括将数据集扩展到其他语言和模态，研究更具欺骗性的AI文本生成方法（如利用语境学习或智能体系统），以及开发更复杂的检测技术。