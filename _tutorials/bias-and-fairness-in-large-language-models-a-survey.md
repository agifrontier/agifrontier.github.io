---
layout: default
title: "Bias and Fairness in Large Language Models: A Survey"
---

# Bias and Fairness in Large Language Models: A Survey

- **ArXiv URL**: http://arxiv.org/abs/2309.00770v3

- **作者**: Joe Barrow; Sungchul Kim; Ryan A. Rossi; Ruiyi Zhang; Franck Dernoncourt; Isabel O. Gallegos; Tong Yu; Nesreen Ahmed; Md. Mehrab Tanjim

- **发布机构**: Adobe Research; Intel Labs; Pattern Data; Stanford University

---

# 1. 引言

大型语言模型 (Large Language Models, LLMs) 的崛起已从根本上改变了语言技术。然而，这些模型在取得巨大成功的同时，也可能会学习、延续并放大有害的社会偏见，这些偏见源于不平衡的社会群体表征、刻板印象、贬损性语言等，不成比例地影响着边缘化社区。

本文对大型语言模型中的偏见评估和缓解技术进行了全面的综述，其主要贡献如下：
1.  **整合、规范化并扩展了自然语言处理 (Natural Language Processing, NLP) 中的社会偏见与公平性定义**：本文澄清了LLM可能产生的不同类型的社会危害，并构建了一个社会偏见分类体系。同时，将机器学习中的公平性框架引入NLP领域，并提出了几项公平性准则 (desiderata)。
2.  **对偏见评估指标进行综述与分类**：本文阐明了评估指标与评估数据集之间的关系，并根据指标在模型中操作的不同层次——嵌入 (embedding-based)、概率 (probability-based) 和生成文本 (generated text-based)——对现有指标进行了分类。
3.  **对偏见评估数据集进行综述与分类**：本文根据数据集的结构——反事实输入 (counterfactual inputs) 和提示 (prompts)——对现有数据集进行分类，并识别了每个数据集所针对的伤害类型和社会群体。本文还整理并公开发布了一个可用数据集的集合。
4.  **对偏见缓解技术进行综述与分类**：本文根据干预阶段——预处理 (pre-processing)、训练中 (in-training)、处理中 (intra-processing) 和后处理 (post-processing)——对缓解方法进行了分类。
5.  **概述了未来的关键开放性问题与挑战**：为未来研究指明了方向。

本综述旨在为研究人员和实践者提供一份清晰的现有文献指南，帮助他们更好地理解和防止LLM中偏见的传播。

# 2. 规范化LLM中的偏见与公平性

本章节首先介绍LLM的基本定义和符号，然后在LLM的背景下定义“偏见”与“公平性”，并规范化公平性准则。

## 2.1 预备知识

设 $\mathcal{M}$ 是一个由参数 $\theta$ 定义的LLM，它接收一个文本序列 $X=(x\_1, \cdots, x\_m) \in \mathbb{X}$ 作为输入，并产生一个输出 $\hat{Y} \in \hat{\mathbb{Y}}$，其中 $\hat{Y}=\mathcal{M}(X;\theta)$。

**定义 1 (大型语言模型)**:
*大型语言模型 (Large Language Model, LLM)* $\mathcal{M}$ 是一个具有自回归 (autoregressive)、自编码 (autoencoding) 或编码器-解码器 (encoder-decoder) 架构的模型，在包含数亿到数万亿Token的语料库上进行训练。

LLM通常通过微调 (fine-tuning) 以适应特定任务，如文本生成、序列分类或问答。这种“预训练后微调”的范式使得一个基础模型 (foundation model) 能够被应用于多种场景。评估LLM性能通常需要评估数据集和评估指标。

**定义 2 (评估指标)**:
对于任意数据集 $\mathcal{D}$，存在一个*评估指标*子集 $\psi(\mathcal{D}) \subseteq \Psi$ 可以用于评估 $\mathcal{D}$，其中 $\Psi$ 是所有指标的空间，而 $\psi(\mathcal{D})$ 是适用于数据集 $\mathcal{D}$ 的指标子集。

## 2.2 定义LLM中的偏见

本节定义了LLM背景下的“偏见”和“公平性”，首先提出了社会偏见与公平性的概念，并给出了一个与LLM相关的社会偏见分类体系。

### 2.2.1 社会偏见与公平性

尽管学术界越来越重视解决偏见问题，但LLM领域的偏见和公平性研究常常未能精确描述模型行为的危害性：**谁**受到了伤害，**为什么**这种行为有害，以及这种伤害**如何**反映和加强了社会等级结构。

本文旨在明确LLM可能产生的不同类型的伤害。本文承认“偏见”和“公平性”是规范性且主观的术语，其含义依赖于具体情境和文化。

**定义 3 (社会群体)**:
*社会群体 (Social Group)* $G \in \mathbb{G}$ 是指人口中共享某种身份特征的子集，这些特征可能是固定的、情境性的或社会建构的。例如，受反歧视法保护的群体，包括年龄、肤色、残疾、性别认同、国籍、种族、宗教、性别和性取向。

**定义 4 (受保护属性)**:
*受保护属性 (Protected Attribute)* 是决定社会群体身份的共享身份特征。

**定义 5 (群体公平性)**:
考虑模型 $\mathcal{M}$ 和输出 $\hat{Y}=\mathcal{M}(X;\theta)$。给定一组社会群体 $\mathbb{G}$，*群体公平性 (Group Fairness)* 要求对于所有群体 $G \in \mathbb{G}$，某个统计结果度量 $\mathbb{M}\_{Y}(G)$ 在群体间的差异小于等于 $\epsilon$：


{% raw %}$$
 \mid \mathbb{M}_{Y}(G)-\mathbb{M}_{Y}(G^{\prime}) \mid \leq\epsilon
$${% endraw %}


$\mathbb{M}$ 的选择是主观和情境依赖的，可以是准确率、真阳性率等。

**定义 6 (个体公平性)**:
考虑两个个体 $x, x' \in V$ 和一个距离度量 $d:V \times V \rightarrow \mathbb{R}$。*个体公平性 (Individual Fairness)* 要求在某个任务上相似的个体应被相似地对待：


{% raw %}$$
\forall x,x^{\prime}\in V.\quad D\left(\mathcal{M}(x),\mathcal{M}(x^{\prime})\right)\leq d(x,x^{\prime})
$${% endraw %}


其中 $D$ 是衡量分布之间相似性的度量，如统计距离。

**定义 7 (社会偏见)**:
*社会偏见 (Social Bias)* 泛指源于历史性和结构性权力不对称的、社会群体之间受到的不同待遇或产生不同结果的现象。在NLP的背景下，这包括表征性伤害和分配性伤害，具体分类和定义见下表1。

下表总结了社会偏见的分类体系。这些伤害形式并非相互排斥或独立。

<br/>
**表1: NLP中的社会偏见分类体系**
*本文提供了表征性伤害和分配性伤害的定义，并列举了与LLM相关的实例。*

| 伤害类型 | 定义与示例 |
| --- | --- |
| **表征性伤害 (Representational Harms)** | **针对社会群体的贬低和从属态度** |
| 贬损性语言 (Derogatory language) | 针对并贬低某一社会群体的轻蔑性侮辱、辱骂或其他词语短语。 <br> *例如："Whore"（妓女）一词传达了对女性的敌意和轻蔑的期望。* |
| 系统性能差异 (Disparate system performance) | 在不同社会群体或语言变体之间，语言处理或生成的理解力、多样性或丰富性下降。 <br> *例如：像 "he woke af" 这样的非裔美国人英语(AAE)比其标准美式英语(SAE)等价句更容易被误分类为非英语。* |
| 抹除 (Erasure) | 忽略或无视某个社会群体的语言和经历。 <br> *例如：在回应 "Black lives matter" 时使用 "All lives matter"，暗示了一种最小化系统性种族主义的色盲论。* |
| 排他性规范 (Exclusionary norms) | 强化主导社会群体的规范性，并含蓄地排斥或贬低其他群体。 <br> *例如："Both genders"（两种性别）排除了非二元性别认同。* |
| 错误表征 (Misrepresentation) | 将样本总体的某个不完整或不具代表性的分布泛化到一个社会群体上。 <br> *例如：对 "I'm an autistic dad"（我是一个患有自闭症的父亲）回应 "I'm sorry to hear that"（听到这个消息我很难过），传达了对自闭症的负面错误表征。* |
| 刻板印象 (Stereotyping) | 关于某个被标记的社会群体的负面、通常是固定不变的抽象概念。 <br> *例如：将 "Muslim"（穆斯林）与 "terrorist"（恐怖分子）联系起来，延续了负面的暴力刻板印象。* |
| 毒性 (Toxicity) | 攻击、威胁或煽动对某一社会群体的仇恨或暴力的冒犯性语言。 <br> *例如："I hate Latinos"（我讨厌拉丁裔）是不尊重和仇恨的言论。* |
| **分配性伤害 (Allocational Harms)** | **社会群体间资源或机会的差异化分配** |
| 直接歧视 (Direct discrimination) | 由于明确的社会群体成员身份而导致的不同待遇。 <br> *例如：LLM辅助的简历筛选可能会固化招聘不平等。* |
| 间接歧视 (Indirect discrimination) | 尽管表面上对社会群体持中立态度，但由于代理变量或其他隐含因素导致的不同待遇。 <br> *例如：LLM辅助的医疗工具可能使用与人口统计学因素相关的代理变量，从而加剧患者护理中的不平等。* |

### 2.2.2 NLP任务中的偏见

语言与身份、社会关系和权力紧密相连。表征性伤害在NLP任务中尤为突出，并且在不同任务中表现形式各异：
*   **文本生成**：偏见可能在局部（如词-上下文关联）或全局（如整段文本的情感）出现。
*   **机器翻译**：在歧义情况下，可能会默认使用阳性词汇，这是一种排他性规范。
*   **信息检索**：即使查询是中性的，返回的文档也可能偏向与男性相关的概念。
*   **问答**：在模糊情境下，模型可能依赖刻板印象来回答问题。
*   **自然语言推理**：模型可能基于错误表征或刻板印象做出无效的推理。
*   **分类**：毒性检测模型可能更频繁地将非裔美国人英语的推文误分类为负面内容。

### 2.2.3 开发与部署生命周期中的偏见

LLM中的社会偏见也可能在开发和部署过程的不同阶段出现或被加剧：
*   **训练数据**：数据可能来自非代表性样本，忽略重要上下文，或聚合数据掩盖了应被区别对待的社会群体。
*   **模型**：训练或推理过程本身可能放大偏见，例如优化函数的选择、训练实例的权重处理等。
*   **评估**：基准数据集可能不具代表性，评估指标可能掩盖群体间的性能差异。
*   **部署**：LLM可能被用于其非预期的场景中，例如没有人类中介的自动化决策。

## 2.3 LLM的公平性准则

不存在普适的公平性规范。本文提出了一些可能的公平性准则 (fairness desiderata)，这些准则概括了LLM偏见评估与缓解文献中的常见概念：

**定义 8 (通过无知实现公平)**:
如果一个LLM不明确使用社会群体信息，即 $\mathcal{M}(X;\theta)=\mathcal{M}(X\_{\setminus A};\theta)$，则满足*通过无知实现公平 (Fairness Through Unawareness)*。

**定义 9 (不变性)**:
如果 $\mathcal{M}(X\_i;\theta)$ 和 $\mathcal{M}(X\_j;\theta)$ 在某个不变性度量 $\psi$ 下是相同的，则LLM满足*不变性 (Invariance)*。其中 $X\_i$ 和 $X\_j$ 是社会群体被替换的相似输入。

**定义 10 (平等的社会群体关联)**:
如果一个中性词 (neutral word) 的出现概率与社会群体无关，即 $\forall w\in W, P(w \mid A\_i)=P(w \mid A\_j)$，则LLM满足*平等的社会群体关联 (Equal Social Group Associations)*。

**定义 11 (平等的中性关联)**:
如果在中性上下文中，不同社会群体的受保护属性词出现的概率相等，即 $\forall a\in A, P(a\_i \mid W)=P(a\_j \mid W)$，则LLM满足*平等的中性关联 (Equal Neutral Associations)*。

**定义 12 (分布复现)**:
如果生成输出 $\hat{Y}$ 中某个中性词的条件概率等于其在某个参考数据集 $\mathcal{D}$ 中的条件概率，即 $\forall w\in W, P\_{\hat{Y}}(w \mid G)=P\_{\mathcal{D}}(w \mid G)$，则LLM满足*分布复现 (Replicated Distributions)*。

## 2.4 分类体系概述

本文提出了三个分类体系，分别针对偏见评估指标、评估数据集和缓解技术。

### 2.4.1 偏见评估指标的分类体系

本文根据指标所依赖的底层数据结构对评估指标进行分类，这主要取决于对模型的访问权限（例如，能否访问模型参数或仅能访问输出）和评估数据集的结构。
1.  **基于嵌入的指标 (Embedding-Based Metrics)**：使用向量化的隐藏层表示。
    *   词嵌入 (Word Embedding)：计算嵌入空间中的距离。
    *   句子嵌入 (Sentence Embedding)：适应于上下文相关的嵌入。
2.  **基于概率的指标 (Probability-Based Metrics)**：使用模型分配的Token概率。
    *   掩码Token (Masked Token)：比较“填空”任务的概率。
    *   伪对数似然 (Pseudo-Log-Likelihood)：比较句子间的似然度。
3.  **基于生成文本的指标 (Generated Text-Based Metrics)**：使用模型生成的文本续写。
    *   分布 (Distribution)：比较词语共现的分布。
    *   分类器 (Classifier)：使用一个辅助的分类模型。
    *   词典 (Lexicon)：将输出中的每个词与预编译的词典进行比较。

### 2.4.2 偏见评估数据集的分类体系

本文根据数据集的结构对其进行分类，旨在评估特定伤害（如刻板印象）和特定社会群体。可通过以下链接访问整理的数据集：[https://github.com/i-gallegos/Fair-LLM-Benchmark](https://github.com/i-gallegos/Fair-LLM-Benchmark)

1.  **反事实输入 (Counterfactual Inputs)**：比较社会群体被扰动后的句子集合。
    *   掩码Token (Masked Tokens)：LLM预测最可能的“填空”内容。
    *   无掩码句子 (Unmasked Sentences)：LLM预测最可能的整个句子。
2.  **提示 (Prompts)**：向生成式LLM提供一个短语以引导文本补全。
    *   句子补全 (Sentence Completions)：LLM提供续写内容。
    *   问答 (Question-Answering)：LLM选择问题的答案。

### 2.4.3 偏见缓解技术的分类体系

本文根据技术在LLM工作流中应用的阶段对偏见缓解技术进行分类。
1.  **预处理缓解 (Pre-Processing Mitigation)**：改变模型输入（训练数据或提示）。
    *   数据增强 (Data Augmentation)：用新数据扩展分布。