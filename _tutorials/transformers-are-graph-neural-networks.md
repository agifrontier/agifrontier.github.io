---
layout: default
title: Transformers are Graph Neural Networks
---

# Transformers是图神经网络

## 1. 用于自然语言处理的Transformer

表示学习 (representation learning) 是任何机器学习任务的基础。深度神经网络将数据的统计和语义信息压缩成一个称为潜在表示或嵌入 (embedding) 的数字列表。一个关键要素是拥有一个富有表现力且可扩展的模型架构，这一点在自然语言处理 (Natural Language Processing, NLP) 领域尤为明显。

### 从RNN到Transformer

循环神经网络 (Recurrent Neural Networks, RNNs) 曾是NLP领域的常用架构。RNNs按顺序（一次一个词）构建句子中每个词的表示。然而，这种顺序性使它们难以处理长文本，因为整个句子的信息被压缩进一个固定长度的表示中。为了解决这个问题，注意力机制 (attention mechanism) 被提出，允许模型在每一步关注输入句子的不同部分。

Transformer网络建立在注意力机制之上，它允许模型并行地构建每个词的表示，而不是顺序地。通过计算每个词相对于其他所有词的重要性，并据此更新词的表示，Transformer能够捕捉长距离依赖关系，从而产生更具表现力的表示。因其出色的表现力和可扩展性，Transformer已取代RNN成为NLP及更广泛深度学习应用的首选架构。

<img src="/images/2506.22084v1/x1.jpg" alt="Representation Learning for NLP" style="width:90%; max-width:800px; margin:auto; display:block;">
**图1**：NLP中的表示学习。RNNs一次构建一个token的表示，捕捉语言的顺序性。Transformers通过注意力机制并行构建表示，捕捉词与词之间的相对重要性。

### 注意力机制

注意力机制是Transformer的核心。给定一个包含$n$个词（或token）的句子$\mathcal{S}$，每个token $i$的初始表示为$h\_i^{\ell=0} \in \mathbb{R}^d$。在从$\ell$层到$\ell+1$层的更新中，每个token $i$的表示 $h\_i^{\ell+1}$ 按如下方式更新：




$$
h_{i}^{\ell+1} = \text{Attention}\left(Q=W_{Q}^{\ell}\ h_{i}^{\ell},\ K=\{W_{K}^{\ell}\ h_{j}^{\ell},\ \forall j\in\mathcal{S}\},\ V=\{W_{V}^{\ell}\ h_{j}^{\ell},\ \forall j\in\mathcal{S}\}\right)
$$






$$
= \sum_{j\in\mathcal{S}}w_{ij}\cdot W_{V}^{\ell}\ h_{j}^{\ell}
$$



其中，$W\_{Q}^{\ell},W\_{K}^{\ell},W\_{V}^{\ell}\in\mathbb{R}^{d\times d}$ 是可学习的线性变换，分别代表查询 (Query)、键 (Key) 和值 (Value)。注意力权重 $w\_{ij}$ 表示token $i$和$j$之间的相对重要性，通过点积和softmax归一化计算得出：




$$
w_{ij} = \text{softmax}_{j\in\mathcal{S}}\left(W_{Q}^{\ell}h_{i}^{\ell}\ \cdot\ W_{K}^{\ell}h_{j}^{\ell}\right)
$$






$$
= \frac{\text{exp}\left(W_{Q}^{\ell}h_{i}^{\ell}\ \cdot\ W_{K}^{\ell}h_{j}^{\ell}\right)}{\sum_{j^{\prime}\in\mathcal{S}}\text{exp}\left(W_{Q}^{\ell}h_{i}^{\ell}\ \cdot\ W_{K}^{\ell}h_{j^{\prime}}^{\ell}\right)}
$$



<img src="/images/2506.22084v1/x2.jpg" alt="A simple attention mechanism" style="width:80%; max-width:500px; margin:auto; display:block;">
**图2**：一个简单的注意力机制。输入token $i$的表示$h\_i^\ell$和句子中其他token的表示集合$\{h\_j^\ell\}$，通过点积和softmax计算出注意力权重$w\_{ij}$，最后通过加权求和得到更新后的token表示$h\_i^{\ell+1}$。所有token并行地进行此过程。

### 多头注意力

为了提升注意力机制的表现力，Transformer引入了多头注意力 (multi-head attention)，它并行计算多组注意力权重。每个头学习自己的一套查询、键和值变换，使模型能同时关注不同类型的关系。

形式上，对于$K$个注意力头，每个头$k$在更新token $i$的表示时，计算自己的查询、键和值：




$$
Q^{k}=W_{Q}^{\ell,k}\ h_{i}^{\ell},\quad K^{k}=\{W_{K}^{\ell,k}\ h_{j}^{\ell},\ \forall j\in\mathcal{S}\},\quad V^{k}=\{W_{V}^{\ell,k}\ h_{j}^{\ell},\ \forall j\in\mathcal{S}\}
$$



其中$W\_{Q}^{\ell,k},W\_{K}^{\ell,k},W\_{V}^{\ell,k}\in\mathbb{R}^{d\times\frac{d}{k}}$是第$k$个头的可学习线性变换。每个头的输出为：


$$
\text{head}_{i}^{k} = \sum_{j\in\mathcal{S}}w_{ij}^{k}\cdot W_{V}^{\ell,k}\ h_{j}^{\ell}
$$


其中权重$w\_{ij}^{k}$的计算方式与单头注意力类似。所有头的输出被拼接并投影，得到最终更新的表示：


$$
\tilde{h}_{i}^{\ell}=\text{Concat}\left(\text{head}_{i}^{1}, \ldots,\text{head}_{i}^{K}\right)O^{\ell}
$$


其中$O^{\ell}\in\mathbb{R}^{d\times d}$是一个可学习的投影矩阵。之后，通过残差连接 (residual connection)、层归一化 (Layer Normalization, LayerNorm) 和一个逐token的前馈网络 (MLP) 进行进一步处理：


$$
h_{i}^{\ell+1} = \text{MLP}\left(\text{LayerNorm}\left(h_{i}^{\ell}+\tilde{h}_{i}^{\ell}\right)\right)
$$


这些标准组件使得构建非常深的Transformer模型成为可能。

<img src="/images/2506.22084v1/x3.jpg" alt="A Transformer layer" style="width:80%; max-width:500px; margin:auto; display:block;">
**图3**：一个Transformer层。多头注意力子层计算token间的相对重要性并更新其表示。随后，更新后的表示由一个逐token的多层感知机（MLP）子层处理。

## 2. 用于图上表示学习的图神经网络

图 (Graph) 被用来为现实世界中复杂互联的系统建模。一个属性图$\mathcal{G}=(\boldsymbol{A},\boldsymbol{H})$由一组节点$\mathcal{V}$和连接它们的边组成，其中$\boldsymbol{A}$是邻接矩阵，$\boldsymbol{H}$是初始节点表示矩阵。图结构的一个关键特性是置换对称性 (permutation symmetry)，即节点的顺序是任意的。

### 消息传递图神经网络

图神经网络 (Graph Neural Networks, GNNs) 是一类为图结构数据设计的深度学习架构。GNNs的核心是消息传递 (message passing) 原理，每个节点通过聚合其邻居节点的信息来迭代更新自身的表示。一个通用的消息传递层包含三个步骤：

1.  **消息构建 (Message construction)**：为每个节点$i$及其邻居$j \in \mathcal{N}\_i$构建消息$\boldsymbol{m}\_{ij}^{\ell}$。


    $$
    \boldsymbol{m}_{ij}^{\ell} = \psi\left(\boldsymbol{h}_{i}^{\ell},\boldsymbol{h}_{j}^{\ell}\right),\quad \forall j\in\mathcal{N}_{i}
    $$


2.  **聚合 (Aggregation)**：将来自节点$i$所有邻居的消息聚合成一个单一消息$\boldsymbol{m}\_{i}^{\ell}$。


    $$
    \boldsymbol{m}_{i}^{\ell} = \bigoplus_{j\in\mathcal{N}_{i}}\boldsymbol{m}_{ij}^{\ell}
    $$


    其中$\bigoplus$是一个置换不变的操作（如求和、均值、最大值）。
3.  **更新 (Update)**：使用聚合后的消息更新节点$i$的表示。


    $$
    \boldsymbol{h}_{i}^{\ell+1} = \phi\left(\boldsymbol{h}_{i}^{\ell},\boldsymbol{m}_{i}^{\ell}\right)
    $$


通过堆叠多层消息传递，GNN可以捕捉图中更复杂的、多跳的邻域关系。

<img src="/images/2506.22084v1/x4.jpg" alt="Representation learning on graphs with message passing" style="width:90%; max-width:800px; margin:auto; display:block;">
**图4**：使用消息传递的图表示学习。GNN通过消息传递机制构建图数据的潜在表示，每个节点从其局部邻域聚合信息。堆叠L层消息传递层可以使信息在L跳子图范围内传播。

### 图注意力网络

图注意力网络 (Graph Attention Networks, GATs) 是一类特殊的GNN，它使用注意力机制在聚合步骤中为不同的邻居加权。在GATs中，从邻居$j$到节点$i$的消息计算方式如下：




$$
\psi\left(\boldsymbol{h}_{i}^{\ell},\boldsymbol{h}_{j}^{\ell}\right) = \text{LocalAttention}\left(W^{\ell}_{Q}\ \boldsymbol{h}_{i}^{\ell}\ , \{W^{\ell}_{K}\ \boldsymbol{h}_{j}^{\ell},\ \forall j\in\mathcal{N}_{i}\}\ , \{W^{\ell}_{V}\ \boldsymbol{h}_{j}^{\ell},\ \forall j\in\mathcal{N}_{i}\}\right)
$$




$$
=\frac{\exp(W^{\ell}_{Q}\ \boldsymbol{h}_{i}^{\ell}\cdot W^{\ell}_{K}\ \boldsymbol{h}_{j}^{\ell})}{\sum_{j^{\prime}\in\mathcal{N}_{i}}\exp(W^{\ell}_{Q}\ \boldsymbol{h}_{i}^{\ell}\cdot W^{\ell}_{K}\ \boldsymbol{h}_{j^{\prime}}^{\ell})}\cdot W^{\ell}_{V}\ \boldsymbol{h}_{j}^{\ell}
$$



更新后的节点表示通过对来自所有邻居的消息进行聚合得到：


$$
\boldsymbol{h}_{i}^{\ell+1} = \boldsymbol{h}_{i}^{\ell}\ +\ \sum_{j\in\mathcal{N}_{i}}\psi\left(\boldsymbol{h}_{i}^{\ell},\boldsymbol{h}_{j}^{\ell}\right)
$$


这些方程与Transformer中的注意力机制几乎完全相同。

## 3. Transformer是全连接图上的GNN

本文的核心论点是：Transformer可以被视为在全连接图上操作的GNN。在全连接图中，每个token都是一个节点，而自注意力机制则是在所有节点对之间建模关系。

我们可以将Transformer中的多头注意力直接实例化到GNN的消息传递框架中。对于每个头，从token $j$到token $i$的消息$\psi\left(\boldsymbol{h}\_{i}^{\ell},\boldsymbol{h}\_{j}^{\ell}\right)$可以表示为全局注意力 (Global Attention)：


$$
\psi\left(\boldsymbol{h}_{i}^{\ell},\boldsymbol{h}_{j}^{\ell}\right) = \text{GlobalAttention}\left(W^{\ell}_{Q}\ \boldsymbol{h}_{i}^{\ell}\ , \{W^{\ell}_{K}\ \boldsymbol{h}_{j}^{\ell},\ \forall j\in\mathcal{S}\}\ , \{W^{\ell}_{V}\ \boldsymbol{h}_{j}^{\ell},\ \forall j\in\mathcal{S}\}\right)
$$




$$
=\frac{\exp(W^{\ell}_{Q}\ \boldsymbol{h}_{i}^{\ell}\cdot W^{\ell}_{K}\ \boldsymbol{h}_{j}^{\ell})}{\sum_{j^{\prime}\in\mathcal{S}}\exp(W^{\ell}_{Q}\ \boldsymbol{h}_{i}^{\ell}\cdot W^{\ell}_{K}\ \boldsymbol{h}_{j^{\prime}}^{\ell})}\cdot W^{\ell}_{V}\ \boldsymbol{h}_{j}^{\ell}
$$


这与GAT的注意力计算公式唯一的区别在于，这里的注意力是在**所有token集合$\mathcal{S}$上**进行归一化，而不是像GAT那样只在**局部邻居$\mathcal{N}\_i$上**。

接着，通过对来自所有token的消息进行加权求和来完成聚合，并使用MLP进行更新：


$$
\boldsymbol{h}_{i}^{\ell+1} = \phi\left(\boldsymbol{h}_{i}^{\ell}\ ,\ \boldsymbol{m}_{i}^{\ell}\right) = \text{MLP}\left(\text{LayerNorm}\left(h_{i}^{\ell}+\sum_{j\in\mathcal{S}}\psi\left(\boldsymbol{h}_{i}^{\ell},\boldsymbol{h}_{j}^{\ell}\right)\right)\right)
$$


这组更新方程与第一节中描述的Transformer层完全相同。

**分类体系总结：**

本文通过建立数学等价性，提出了一个统一Transformer和GNN的视角。其分类或关联的**核心标准**是**图的连通性假设**：
*   **标准GNN（如GAT）**：在**稀疏的、预定义的图**上运行。注意力（消息传递）被限制在节点的局部邻域内。这是一种带有强归纳偏置（由图结构定义）的模型。
*   **Transformer**：在**全连接图**上运行。每个token都可以关注所有其他token。它没有预定义的稀疏结构限制，而是**学习**输入元素之间的关系。这是一种表现力更强、归纳偏置更弱的模型。

从这个视角看，Transformer是一个强大的集合处理网络，能够学习数据中的局部和全局上下文，而不受预定义图结构的束缚。反过来看，GAT可以被视为一种带有稀疏或掩码注意力的Transformer。

这一联系启发了图表示学习的一个新方向：**图Transformer (Graph Transformers)**。这类模型旨在结合GNN的局部消息传递和Transformer的全局多头注意力。它们通过引入图结构的位置编码 (positional encodings) 作为输入特征，将图的归纳偏置“软性”地注入到Transformer架构中，从而克服了传统消息传递GNN的表达能力限制，同时保留了图结构的优势。

## 4. Transformer是赢得硬件彩票的GNN

尽管Transformer在概念上可以被视为在全连接图上操作的GNN，但两者在实际实现上存在一个关键区别，这与“硬件彩票” (hardware lottery) 现象有关——即架构与硬件的契合度决定了哪些研究思想能够脱颖而出。

Transformer的全局多头注意力是通过高度优化的**密集 (dense) 矩阵乘法**实现的。给定一个表示矩阵$\boldsymbol{H}\in\mathbb{R}^{n\times d}$，所有token的自注意力可以并行计算：


$$
\tilde{\boldsymbol{H}}^{\ell} = \text{softmax}\left(\boldsymbol{H}^{\ell}\ W_{Q}^{\ell}\ \left(\boldsymbol{H}^{\ell}\ W_{K}^{\ell}\right)^{\top}\right)\ \left(\boldsymbol{H}^{\ell}\ W_{V}^{\ell}\right)
$$


这种实现方式能够充分利用现代GPU和TPU的并行处理能力。

相比之下，GNN通常执行**稀疏 (sparse) 消息传递**，这在当前硬件上效率要低得多（除了非常稀疏或数十亿级别的大图）。它需要为每个节点维护邻居索引，并执行gather和scatter操作，训练速度比标准Transformer慢几个数量级，且难以扩展。

**未来方向的启示:**

1.  **架构与硬件的协同进化**：Transformer的成功很大程度上归功于其架构与现代并行计算硬件的高度契合。未来的架构设计需要继续考虑硬件的特性。
2.  **通用、可扩展架构的兴起**：有经验证据表明，当以足够大的规模进行训练时，Transformer能够学习到传统上由GNN等架构内置的归纳偏置（如局部性）。这呼应了AI研究中的“惨痛教训” (The Bitter Lesson)，即通用、可扩展的计算方法最终会胜过依赖人类领域知识的精巧模型。
3.  **结构信息的软性注入**：未来的研究方向之一是，如何在像Transformer这样的通用架构中，通过位置编码等方式有效地注入关于数据底层结构的“提示”，而不是将其作为硬性约束嵌入架构中。图Transformer正是这一方向的体现。

综上所述，本文的结论是，**Transformer是当前正在赢得硬件彩票的图神经网络**。其表现力、灵活性以及与现代硬件的完美结合，使其成为跨多个应用领域（包括图）进行结构化数据表示学习的首选架构。