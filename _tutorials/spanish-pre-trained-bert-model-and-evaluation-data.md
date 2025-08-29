---
layout: default
title: "Spanish Pre-trained BERT Model and Evaluation Data"
---

# Spanish Pre-trained BERT Model and Evaluation Data

- **ArXiv URL**: http://arxiv.org/abs/2308.02976v1

- **作者**: Hojin Kang; Jorge P'erez; Gabriel Chaperon; Jou-Hui Ho; Rodrigo Fuentes; J. Cañete

- **发布机构**: Millennium Institute for Foundational Research on Data; Universidad de Chile

---

# TL;DR
本文提出并开源了第一个完全基于西班牙语料库预训练的BERT模型（BETO），并构建了一个名为GLUES的西班牙语NLP任务评估基准，实验证明该单语模型在多数西班牙语下游任务上的表现优于同等规模的多语言BERT模型。

# 关键定义
本文主要应用和扩展了现有技术，并为西班牙语社区创建了新的资源，关键定义如下：

*   **BETO**: 本文提出的西班牙语BERT模型的名称（源于其GitHub仓库名beto）。它是一个基于BERT-Base架构，但完全使用西班牙语语料进行预训练的语言模型。模型包含12个自注意力层，每层12个注意力头，隐藏层维度为768，总参数量约1.1亿。本文发布了cased（区分大小写）和uncased（不区分大小写）两个版本。

*   **GLUES (GLUE for Spanish)**: 本文整理和汇编的一个用于评估西班牙语NLP模型性能的基准测试集。它仿照了英语领域的GLUE基准，整合了多项西班牙语下游任务，旨在为西班牙语NLP研究提供标准化的评测方案。

# 相关工作
预训练语言模型 (Pre-trained language models) 已成为NLP领域的主流范式，其典型代表是通过自监督学习在海量无标签文本上进行预训练，然后在特定下游任务上进行微调 (fine-tune)。早期的ULM-Fit采用循环神经网络，而BERT则基于强大的Transformer架构和掩码语言模型 (Masked Language Modeling, MLM) 任务，极大地推动了技术发展。

为了支持英语和中文以外的语言，研究人员发布了多语言BERT (mBERT)，它在包含100多种语言的混合语料上进行训练，并在跨语言任务上表现出色。然而，社区发现，针对单一语言（如法语、荷兰语、俄语等）训练的单语BERT模型，通常能在该语言的特定任务上超越mBERT。

尽管西班牙语是世界主要语言之一，但当时NLP社区缺乏一个高质量、公开可用的西班牙语预训练BERT模型和标准化的评估基准。本文旨在填补这一空白，为西班牙语NLP研究提供核心基础资源。

# 本文方法

### 模型架构与词表
本文提出的西班牙语BERT模型在架构上与$$BERT-Base$$保持一致，包含12个自注意力层 (self-attention layers)，12个注意力头 (attention-heads)，隐藏层维度为768，总参数量约为1.1亿。

研究人员构建了一个大小为3.2万的词表 (vocabulary)。该词表基于SentencePiece库的字节对编码 (Byte Pair Encoding, BPE) 算法生成了3.1万个subword token，并额外增加了1000个占位符token以备后用。

### 预训练数据
为了训练模型，本文收集并整合了大规模的西班牙语文本语料，总词数约30亿。数据来源主要包括：
*   西班牙语维基百科 (Wikipedia) 的全部内容。
*   OPUS项目 (Tiedemann, 2012) 中所有包含西班牙语文本的语料，如联合国和政府期刊、TED演讲、字幕、新闻故事等。
这个语料库是当时最新的西班牙语大型语料库之一，并且已被公开发布。

### 训练创新点
本文在训练过程中借鉴了RoBERTa等后续工作的成功经验，对原始BERT的训练方法进行了优化：
*   **动态掩码 (Dynamic Masking)**：与原始BERT在数据预处理阶段对每个句子只生成一个静态掩码不同，本文采用了动态掩码技术。在每个训练周期 (epoch) 中，同一个输入句子会被生成不同的掩码版本（本文使用了10x动态掩码），增强了模型的学习鲁棒性和数据利用率。
*   **全词掩码 (Whole-Word Masking, WWM)**：当一个词被分割成多个subword token时，WWM策略会确保将构成这个完整单词的所有subword token同时掩码，而不是只掩码其中的一部分。这有助于模型学习更高层次的语义表示。
*   **两阶段训练与大批量 (Two-phase Training & Large Batches)**：训练分为两个阶段。第一阶段（前90万步）使用较短的序列长度（128）和更大的批量大小（2048）；第二阶段则使用更长的序列长度（512）和较小的批量大小（256）。这种策略可以加速早期训练并让模型在后期适应更长的上下文。

### GLUES基准测试
为了系统性地评估模型性能，本文构建了GLUES基准，整合了以下七类西班牙语NLP任务：

*   **自然语言推断 (Natural Language Inference, NLI)**: 使用XNLI数据集，判断前提(premise)和假设(hypothesis)之间的关系（蕴含、矛盾、中立）。
*   **释义识别 (Paraphrasing)**: 使用PAWS-X数据集，判断两个句子是否在语义上等价。
*   **命名实体识别 (Named Entity Recognition, NER)**: 使用CoNLL-2002数据集，识别文本中的人名、组织、地点等实体。
*   **词性标注 (Part-of-Speech Tagging, POS)**: 使用Universal Dependencies (v1.4)，为每个单词标注其语法词性。
*   **文档分类 (Document Classification)**: 使用MLDoc数据集，将路透社新闻文档分为四个类别。
*   **依存句法分析 (Dependency Parsing)**: 使用Universal Dependencies (v2.2)，构建句子的语法结构树。
*   **问答 (Question Answering, QA)**: 使用多个翻译版的SQuAD v1.1数据集，包括MLQA, TAR, 和XQuAD，从给定上下文中提取问题的答案。

# 实验结论

### 结果总结
本文将训练好的西班牙语BERT模型（分为cased和uncased版本）在GLUES的各项任务上进行了微调，并与文献中报道的最佳mBERT性能进行了对比。

**表格1：分类任务结果对比 (准确率/F1分数)**

| Model | XNLI | PAWS-X | NER | POS | MLDoc |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Best mBERT | 78.50 | 89.00 | 87.38 | 97.10 | 95.70 |
| **es-BERT uncased** | 80.15 | **89.55** | 82.67 | 98.44 | **96.12*** |
| **es-BERT cased** | **82.01** | 89.05 | **88.43** | **98.97*** | 95.60 |
*注：*表示达到新的SOTA水平。文献来源：a(Wu & Dredze, 2019), b(Yang et al., 2019a)。*

**表格2：问答任务结果对比 (F1 / 精确匹配率)**

| Model | MLQA, MLQA (Train, Test) | TAR, XQuAD (Train, Test) | TAR, MLQA (Train, Test) |
| :--- | :---: | :---: | :---: |
| Best mBERT | 53.90 / 37.40 | **77.60 / 61.80** | 68.10 / 48.30 |
| **es-BERT uncased** | 67.85 / 46.03 | 77.52 / 55.46 | 68.04 / 45.00 |
| **es-BERT cased** | **68.01 / 45.88** | 77.56 / 57.06 | **69.15 / 45.63** |
*注：文献来源：c(Lewis et al., 2019), d(Artetxe et al., 2019)。*

### 优势与不足
*   **优势**: 实验结果表明，本文提出的西班牙语BERT模型（BETO）在大多数任务上显著优于多语言模型mBERT。特别是在训练数据量较大的XNLI任务上，性能提升最为明显。在POS和MLDoc这两个标准的西班牙语数据集上，BETO甚至创造了新的技术水平（State-of-the-Art, SOTA）。

*   **不足**: 
    1.  在部分问答（QA）任务设置中，BETO的表现并未超越mBERT。作者推测这可能与训练数据（如MLQA）的机器翻译质量不高有关，数据集中近一半的样本存在答案与上下文位置不匹配的问题。
    2.  作者也指出，BETO作为纯单语模型，无法像mBERT那样利用其它语言（尤其是英语）的训练数据进行跨语言迁移学习。更大规模的多语言模型，如拥有5.6亿参数的XLM-RoBERTa，通过利用多语言数据，在XNLI和NER等任务上取得了比BETO更好的成绩。

### 最终结论
本文成功地预训练并开源了一个高性能的西班牙语BERT模型（BETO），并通过构建GLUES基准对其进行了全面评估。实验证明，专为单一语言设计的模型在大多数情况下比通用的多语言模型更具优势。这项工作为西班牙语NLP社区提供了宝贵的基础设施，有望推动西班牙语国家在NLP领域的研究与应用。