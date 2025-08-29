---
layout: default
title: "StarCoder: may the source be with you!"
---

# StarCoder: may the source be with you!

- **ArXiv URL**: http://arxiv.org/abs/2305.06161v2

- **作者**: Swayam Singh; Olivier Dehaene; Qian Liu; Thomas Wolf; Manuel Romero; Daniel Fried; M. Kunakov; Terry Yue Zhuo; N. Fahmy; L. V. Werra; 等57人

- **发布机构**: CSIRO; Carnegie Mellon University; Columbia University; Discover Dollar Pvt Ltd; Eleuther AI; Forschungszentrum Jülich; Hugging Face; IBM Research; Independent; Johns Hopkins University; Leipzig University; MIT; McGill University; Mila; Monash University; NYU; Northeastern University; Queen Mary University of London; Roblox; SAP; Saama; ScaDS.AI; Sea AI Lab; ServiceNow; Stanford University; Technical University of Munich; Technion – Israel Institute of Technology; Telefonica I+D; The Alan Turing Institute; Toloka; UnfoldML; University of Allahabad; University of British Columbia; University of Notre Dame; University of Vermont; Weizmann Institute of Science; Wellesley College

---

# TL;DR
本文发布了StarCoder，一个拥有155亿参数、支持8K上下文长度和代码填充能力的开源代码大语言模型；该模型在经过精心筛选和隐私信息处理的 permissively licensed 代码数据集上进行训练，其性能超越了所有现存的多语言开源代码模型，并达到了与闭源模型相当的水平。

# 关键定义
*   **StarCoder / StarCoderBase**：这是本文提出的两个核心模型。StarCoderBase 是一个拥有155亿参数的基础模型，在来自80多种编程语言、GitHub Issues、Git Commits和Jupyter Notebooks的上万亿个token上训练而成。StarCoder 是在StarCoderBase的基础上，使用350亿个Python token进行微调得到的模型。
*   **The Stack**: 一个大规模、公开可用的代码预训练数据集，包含了6.4TB、来自384种编程语言的、使用宽松许可证（permissively licensed）的源代码。该数据集提供了一套透明的数据治理框架，包括供开发者检查其代码是否被收录的工具，以及一个选择退出（opt-out）的流程。
*   **多查询注意力 (Multi-Query Attention, MQA)**：一种注意力机制的变体。与标准的多头注意力（Multi-Head Attention）中每个头都有独立的键（key）和值（value）投影不同，MQA让所有头共享同一组键和值。这极大地减少了推理过程中的内存占用和解码时间，从而实现了快速的大批量推理。
*   **代码填充 (Fill-in-the-Middle, FIM)**：一种训练目标和推理能力，使模型不仅能预测代码的后续部分，还能根据上下文填充代码的中间部分。通过使用特殊的分隔符token，模型学会了在给定前缀和后缀的情况下生成缺失的代码片段。

# 相关工作
当前，性能最顶尖的代码大语言模型（Code LLMs），如谷歌的PaLM、DeepMind的Chinchilla和OpenAI的GPT系列，大多是闭源的，仅通过付费API提供有限访问。这种封闭性限制了社区对模型安全性、内部工作原理的研究，也阻碍了外部研究者为模型改进做出贡献。

另一方面，虽然已有一些开源或开放访问（open-access）的模型，如PolyCoder、SantaCoder、CodeGen、LLaMA等，但它们通常存在一些局限：要么模型规模较小、性能不及闭源模型；要么其许可证限制了商业用途；要么其训练数据和处理过程不够透明，引发了关于版权和隐私的担忧。

因此，当前领域的核心问题是在高性能与开放性、负责任之间存在巨大鸿沟。本文旨在解决这一问题，目标是开发并发布一个性能强大、真正开放（模型权重、数据、开发过程均透明）、且负责任（处理版权、隐私问题）的代码大语言模型，以弥合闭源模型与现有开源模型之间的差距。

# 本文方法

## 数据集构建与清洗
本文的训练数据主要基于The Stack v1.2数据集，这是一个专门为代码模型构建的、包含宽松许可证源代码的大规模数据集。在此基础上，作者进行了一系列精细的数据筛选、清洗和处理。

### 数据来源与筛选
1.  **编程语言选择**：从The Stack的358种语言中，综合考虑了数据量大小、在TIOBE和Githut等流行度排行榜上的排名，最终筛选出86种语言。
2.  **人工质检**：招募社区志愿者对抽样的3万个文件（覆盖300种文件扩展名）进行人工检查，以确保代码质量，排除自动生成或非代码内容，并为后续的自动过滤规则提供依据。
3.  **专用过滤器**：
    *   **XML过滤器**：通过检测文件头部是否包含$$<?xml version=$$来剔除被错误分类的XML文件。
    *   **字母比例过滤器**：对于像MATLAB这样可能包含大量数据张量的文件，移除字母字符比例低于25%的文件。
    *   **HTML过滤器**：移除样板代码和链接过多的HTML文件，仅保留可见文本占比较高（>20%）的文件。
    *   **Jupyter Notebooks处理**: 将Jupyter Notebooks转化为两种格式：一是纯脚本格式（Jupyter – scripts），二是保留Markdown和代码块结构化配对的格式（Jupyter – structured）。
    *   **GitHub Issues和Git Commits**：对GitHub Issues中的对话进行清洗，去除机器人评论和非英文内容；对Git Commits数据应用多种启发式规则过滤，以保留高质量的代码变更记录。

<img src="/images/2305.06161v2/x1.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">
<center>图1: 标注的PII数据集中编程语言的分布。</center>

### 数据去重与加权
*   **去重**：采用MinHash和局部敏感哈希（LSH）技术对所有源代码文件和Jupyter Notebooks进行近乎重复（near-deduplication）的检测和移除，Jaccard相似度阈值设为0.7。
*   **加权**：训练数据基本按各语言和数据源的自然体积比例进行采样。唯一的例外是JSON、YAML和CSS，因为不希望模型在这些格式的数据内容上浪费过多计算资源，它们的体积被人为降低。

以下表格展示了StarCoder训练数据中各编程语言在去重、过滤和去污染后的文件数量和数据量。


| 语言 | 去重后文件数 | 去重后体积 (GB) | 过滤后文件数 | 过滤后体积 (GB) | 权重 | 百分比 |
| --- | --- | --- | --- | --- | --- | --- |
| c | 8,625,559 | 57.43 | 8,536,791 | 53.89 | 53.89 | 7.027 |
| c-sharp | 10,839,399 | 46.29 | 10,801,285 | 44.66 | 44.66 | 5.823 |
| cpp | 6,377,914 | 50.89 | 6,353,527 | 48.92 | 48.92 | 6.379 |
| java | 20,151,565 | 89.30 | 20,071,773 | 86.94 | 86.94 | 11.336 |
| javascript | 21,108,587 | 141.65 | 19,544,285 | 64.71 | 64.71 | 8.437 |
| python | 12,962,249 | 64.30 | 12,866,649 | 60.40 | 60.40 | 7.875 |
| php | 15,904,518 | 66.84 | 15,683,017 | 60.89 | 60.89 | 7.939 |
| markdown | 21,045,171 | 75.25 | 21,029,287 | 74.93 | 74.93 | 9.77 |
| html | 9,533,367 | 146.76 | 3,299,965 | 29.36 | 29.36 | 3.828 |
| typescript | 10,637,070 | 28.82 | 10,547,331 | 26.52 | 26.52 | 3.458 |
| go | 4,730,461 | 25.74 | 4,700,526 | 23.78 | 23.78 | 3.101 |
| ... | ... | ... | ... | ... | ... | ... |
| **总计** | | | **305,929,658** | **815.68** | **799.37** | **100** |

<center>表1 & 2合并简化: StarCoder训练数据概览（部分语言）。</center>

## 个人可识别信息（PII）处理
为了负责任地处理数据，本文构建了一套先进的PII（Personally Identifiable Information）移除流程。

1.  **PII数据集构建**：通过众包平台Toloka组织了1399名标注者，对来自31种编程语言的12,000个文件进行了PII标注，共得到22,950个标注实体，涵盖姓名、邮箱、IP地址、密钥、密码和用户名等类型。
2.  **StarEncoder模型**：为了更好地理解代码上下文以进行PII识别，本文首先训练了一个专用于代码的编码器模型StarEncoder。该模型基于BERT架构，在代码数据上使用MLM（Masked Language Modelling）和NSP（Next Sentence Prediction）目标进行预训练。
3.  **PII检测模型**：在StarEncoder的基础上，通过在新建的PII标注数据集上进行微调，训练了一个用于命名实体识别（NER）的PII检测模型。为了提升对密钥（key）和密码（password）等稀有类别的检测能力，还采用了伪标签（pseudo-labeling）技术，即用一个初始模型在大量未标注数据上生成高置信度的标签，再用这些伪标签数据来增强训练。最终模型的F1分数在邮箱、姓名、IP地址等类别上超过90%。


| PII类型 | 出现次数 | 召回率 | 精确率 |
| :--- | :--- | :--- | :--- |
| IP_ADDRESS | 2526 | 85% | 97% |
| KEY | 308 | 91% | 78% |
| PASSWORD | 598 | 91% | 86% |
| ID | 1702 | 53% | 51% |
| EMAIL | 5470 | 99% | 97% |
| USERNAME | 780 | 74% | 86% |

<center>表5简化: PII类型、标注数量及人工检查的质量评估。</center>

## 模型架构与训练
StarCoder模型本身集成了多项先进的架构设计，以实现高性能和新功能。

### 模型架构
*   **基础**: 一个155亿参数的解码器-仅（Decoder-only）Transformer模型。
*   **长上下文**: 支持高达8192个token的上下文长度，这得益于 FlashAttention 等高效注意力实现。
*   **多查询注意力 (MQA)**: 采用MQA来加速大批量推理时的解码速度并减少内存占用。
*   **代码填充 (FIM)**: 模型通过特殊的$$<fim_prefix>$$, $$<fim_middle>$$, $$<fim_suffix>$$ token进行训练，使其原生具备根据前后文填充代码中间部分的能力。


| 超参数 | 值 |
| :--- | :--- |
| 隐藏层大小 | 768 |
| 中间层大小 | 3072 |
| 最大位置嵌入 | 1024 |
| 注意力头数量 | 12 |
| 隐藏层数量 | 12 |
| 注意力机制 | Multi-head |
| 参数量 | ≈1.25亿 |

<center>表6: PII检测模型的基础StarEncoder架构。</center>

*注意：上表为StarEncoder，StarCoder主模型为15.5B参数。*

### 训练过程
1.  **StarCoderBase**: 在经过上述清洗、去重和PII处理后的多语言代码数据集上训练，总共处理了约1万亿个token。
2.  **StarCoder**: 在StarCoderBase模型的基础上，额外使用了350亿个Python token进行微调，以增强其在Python语言上的特定能力。

## 负责任的AI发布
本文在模型发布方式上采取了多个重要步骤，以确保其开放、安全和负责任。
*   **OpenRAIL-M许可证**：模型在OpenRAIL-M许可证下发布，这是一种“负责任的AI模型”许可证。它允许免版税的访问、使用和分发（包括商业用途），但嵌入了一系列针对已识别关键场景的使用限制，以防止滥用。
*   **归因工具 (Attribution Tool)**：随模型发布了一个集成在VSCode演示中的归因工具。该工具可以帮助用户检测生成的代码是否与训练集中的某个片段高度相似（即可能存在复制）。它通过一个轻量级的成员资格检查和后续的BM25索引搜索两步过程实现。

# 实验结论
本文对StarCoder模型进行了迄今为止针对代码大语言模型最全面的评估，涵盖了多种基准测试，并得出了以下关键结论：

*   **多语言性能领先**：在支持多种编程语言的开放代码大语言模型中，StarCoderBase的表现全面超越了所有竞争对手（如CodeGen、CodeGeeX等）。
*   **媲美闭源模型**：StarCoderBase的性能与OpenAI的$$code-cushman-001$$模型（早期版本的Codex）相当，甚至在某些方面有所超越。
*   **Python微调效果显著**：经过Python微调的StarCoder模型，其性能大幅超越了其他同样在Python上微调过的现有模型。
*   **知识保持能力**：尽管在Python上进行了专门的微调，StarCoder模型在其他编程语言上的性能并未出现明显下降，保持了强大的多语言能力。

尽管本文没有明确指出模型表现不佳的具体场景，但实验结果有力地证明，通过开放、社区驱动和负责任的开发方法，可以构建出与顶级闭源模型性能相媲美的代码大语言模型。StarCoder的发布及其配套的数据、工具和许可证，为开源AI社区树立了一个新的标杆。