---
layout: default
title: "Group Representational Position Encoding"
---

# 群表示位置编码（GRAPE）：统一RoPE与ALiBi的理论框架

<img src="/images/2512.07805v1/A__title.jpg" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

Transformer架构的核心是自注意力机制，但它本身无法感知序列中Token的顺序，即具有**置换不变性**（Permutation-Invariance）。为了让模型理解“词语A在词语B之前”，必须引入位置信息。这就是**位置编码**（Positional Encoding）的作用。

> ArXiv URL：http://arxiv.org/abs/2512.07805v1

位置编码技术经历了多次演进。最初的方法是为每个位置分配一个固定的或可学习的**绝对位置编码**（Absolute Positional Encoding）。后来，研究者发现**相对位置编码**（Relative Positional Encoding）更为有效，因为它只关注Token之间的相对距离，而不是它们的绝对位置。

其中，**旋转位置编码**（**Rotary Position Embedding, RoPE**）和**注意力线性偏置**（**Attention with Linear Biases, ALiBi**）是两种主流且表现优异的方案。RoPE通过旋转查询（Query）和键（Key）向量来编码相对位置，保持了向量的范数。ALiBi则直接在注意力分数上添加一个与相对距离成正比的惩罚项，实现简单且具有出色的长度外推能力。

尽管这些方法很成功，但它们似乎源于不同的设计哲学：一个是乘法式的几何变换，另一个是加法式的偏置。它们之间是否存在更深层次的联系？能否在一个统一的理论框架下理解、甚至改进它们？

**群表示位置编码**（**Group Representational Position Encoding, GRAPE**）正是为了回答这些问题而提出的。它利用**群论**（Group Theory）这一强大的数学工具，构建了一个统一的框架，将RoPE和ALiBi等看似无关的方法，都囊括为该框架下的特例。

GRAPE的核心思想是，位置信息可以通过**群作用**（Group Action）来表示。具体来说，位置 $n$ 对应于一个群元素 $G(n)$，这个元素是一个矩阵，作用于词向量上。这个群元素是通过**矩阵指数**（Matrix Exponential）从一个更基础的**生成元**（Generator）$L$ 导出的：$G(n) = \exp(n\omega L)$。

![图：GRAPE框架概览](images/page_1_Figure_0.jpg)

图：GRAPE框架概览。左侧的乘法GRAPE通过特殊正交群SO(d)中的旋转操作，统一了RoPE等方法。右侧的加法GRAPE通过一般线性群GL中的单能（unipotent）变换，统一了ALiBi和FoX等方法。

这个简洁的公式蕴含了深刻的物理和数学意义，它不仅统一了现有的方法，还为设计新的、更强大的位置编码方案开辟了广阔的设计空间。

### 核心思想：群论与位置的相对性

群论是研究对称性的数学分支。一个**群**（Group）包含一个元素集合和一个运算，这个运算满足封闭性、结合律、有单位元和逆元等性质。

将群论用于位置编码，最关键的特性是它能自然地表达“相对”关系。一个**单参数子群**（one-parameter subgroup）$G(t)$ 具有性质 $G(t+s) = G(t)G(s)$。这个性质对于位置编码来说是完美的。

在注意力计算中，我们希望位置 $i$ 的查询向量和位置 $j$ 的键向量之间的交互，只依赖于它们的相对偏移 $j-i$。如果我们将位置变换定义为 $G(n)$，那么对查询和键的变换可以写作：




{% raw %}$$ \widetilde{\mathbf{q}}_i = \mathbf{G}(i)\mathbf{q}_i, \qquad \widetilde{\mathbf{k}}_j = \mathbf{G}(j)\mathbf{k}_j $${% endraw %}



它们的内积，也就是注意力分数的核心部分，会变成：




{% raw %}$$ \widetilde{\mathbf{q}}_i^{\top} \widetilde{\mathbf{k}}_j = (\mathbf{G}(i)\mathbf{q}_i)^{\top} (\mathbf{G}(j)\mathbf{k}_j) = \mathbf{q}_i^{\top} \mathbf{G}(i)^{\top} \mathbf{G}(j) \mathbf{k}_j $${% endraw %}



如果 $G(n)$ 是一个**正交矩阵**（Orthogonal Matrix），满足 $G(i)^\top = G(i)^{-1} = G(-i)$，那么利用群的性质，上式可以简化为：




{% raw %}$$ \mathbf{q}_i^{\top} \mathbf{G}(-i) \mathbf{G}(j) \mathbf{k}_j = \mathbf{q}_i^{\top} \mathbf{G}(j-i) \mathbf{k}_j $${% endraw %}



这个结果非常优雅：注意力分数只与相对位置 $j-i$ 的变换矩阵 $G(j-i)$ 有关，与绝对位置 $i$ 和 $j$ 无关。GRAPE正是基于这一原理，构建了两种不同类型的群作用。

### 乘法GRAPE：旋转的艺术

第一种是**乘法GRAPE**（**Multiplicative GRAPE, GRAPE-M**），它将位置编码理解为一种旋转操作。这里的群是**特殊正交群**（**Special Orthogonal Group, SO(d)**），其元素是 $d$ 维空间中保持向量长度和方向的旋转矩阵。

#### 生成元与罗德里格斯公式

GRAPE-M的生成元 $L$ 是一个**斜对称矩阵**（skew-symmetric matrix），即 $L^\top = -L$。这种矩阵属于李代数 $\mathfrak{so}(d)$。最简单的非平凡生成元是**秩为2**（rank-2）的，它由两个向量 $a, b \in \mathbb{R}^d$ 定义：




{% raw %}$$ \mathbf{L}(\mathbf{a}, \mathbf{b}) = \mathbf{a} \mathbf{b}^{\top} - \mathbf{b} \mathbf{a}^{\top} $${% endraw %}



这个生成元定义的旋转发生在由向量 $a$ 和 $b$ 张成的二维平面内，而对该平面外的所有向量没有影响。

计算矩阵指数 $\exp(n\omega L)$ 通常很复杂，但对于这种秩为2的生成元，存在一个高效的**闭式解**（closed-form solution），类似于**罗德里格斯旋转公式**（Rodrigues' formula）：




{% raw %}$$ \exp(\mathbf{L}) = \mathbf{I} + \frac{\sin s}{s} \mathbf{L} + \frac{1 - \cos s}{s^2} \mathbf{L}^2 $${% endraw %}



其中 $s$ 是一个与 $a$ 和 $b$ 相关的标量。这个公式使得我们无需显式构造出巨大的旋转矩阵，就能以 $O(d)$ 的线性时间复杂度完成对向量的旋转操作，非常高效。

#### RoPE作为乘法GRAPE的特例

乘法GRAPE最引人注目的成果之一，就是揭示了RoPE的数学本质。RoPE可以被精确地看作是乘法GRAPE的一个特例。

在RoPE中， $d$ 维的向量空间被划分为 $d/2$ 个互不相干的二维坐标平面。位置编码在这每个二维平面上独立进行旋转。这在GRAPE的视角下，等价于选择了一组特殊的、两两**正交**（orthogonal）且**通勤**（commuting）的秩-2生成元 $\{L\_i\}$。总的生成元是这些生成元的加权和：




{% raw %}$$ \mathbf{L}_{\text{RoPE}} = \sum_{i=1}^{d/2} \theta_i \mathbf{L}_i $${% endraw %}



由于各个 $L\_i$ 作用在不相交的子空间上，它们相互通勤（$[L\_i, L\_j] = 0$），因此总的旋转可以分解为各个子空间旋转的乘积：




{% raw %}$$ \mathbf{G}(n) = \exp\left(n\mathbf{L}_{\text{RoPE}}\right) = \prod_{i=1}^{d/2} \exp(n\theta_i \mathbf{L}_i) $${% endraw %}



这正是RoPE的块对角旋转矩阵形式。GRAPE不仅解释了RoPE，还指明了扩展方向：我们可以使用**可学习的**（learned）、**非正交的**（non-orthogonal）甚至**非通勤的**（non-commuting）生成元，让不同维度特征在旋转过程中相互耦合，从而可能捕获更复杂的依赖关系。

### 加法GRAPE：平移的智慧

第二种是**加法GRAPE**（**Additive GRAPE, GRAPE-A**），它解释了ALiBi这类加法偏置的来源。这套机制的思想更为巧妙，它通过“升维”来把加法变成乘法。

#### 齐次坐标与单能变换

为了用矩阵乘法实现加法（平移），GRAPE-A采用了图形学中常用的**齐次坐标提升**（homogeneous lift）。一个 $d$ 维向量 $x$ 被增广为 $d+1$ 维向量 $[x; 1]$。

此时，操作的群不再是旋转群 $SO(d+1)$，而是更广泛的**一般线性群**（**General Linear Group, GL(d+1)**），其元素是所有可逆的 $(d+1) \times (d+1)$ 矩阵。其生成元 $A$ 是一种特殊的**幂零矩阵**（nilpotent matrix），满足 $A^2 = 0$。一个典型的生成元形式如下：




{% raw %}$$ \mathbf{A} = \begin{bmatrix} \mathbf{0}_{d \times d} & \mathbf{u} \\ \mathbf{0}_{1 \times d} & 0 \end{bmatrix} $${% endraw %}



由于 $A^2=0$，其矩阵指数的泰勒展开变得异常简单：




{% raw %}$$ \mathbf{G}_{\text{add}}(n) = \exp(n \omega \mathbf{A}) = \mathbf{I} + n \omega \mathbf{A} = \begin{bmatrix} \mathbf{I}_d & n \omega \mathbf{u} \\ \mathbf{0}^\top & 1 \end{bmatrix} $${% endraw %}



这是一个**单能变换**（unipotent transformation），其所有特征值都为1。当这个变换作用于齐次坐标下的查询和键向量时，最终的注意力分数中会神奇地出现一个加法项，它与相对位置 $j-i$ 呈线性关系，并且可以由内容（如键向量）进行门控。

#### ALiBi与FoX作为加法GRAPE的特例

加法GRAPE最直接的应用是为ALiBi提供了严谨的理论基础。ALiBi在注意力分数上增加一个与内容无关的偏置项 $\beta\_h(j-i)$。

通过将向量提升到 $d+2$ 维空间，并精心设计查询和键的增广方式以及幂零生成元 $A\_h$，GRAPE-A可以精确地推导出ALiBi的偏置项：




{% raw %}$$ \widehat{\mathbf{q}}_i^{\top} \mathbf{G}_{\text{add},h}(j-i)^{-\top} \widehat{\mathbf{k}}_j = \mathbf{q}_i^{\top} \mathbf{k}_j \ - \ (j-i) \, \beta_h $${% endraw %}



这个结果表明，ALiBi并非一个启发式的技巧，而是可以从一般线性群中的单能变换自然导出。同样，研究证明了**遗忘变换器**（**Forgetting Transformer, FoX**）中的遗忘偏置也可以被看作是加法GRAPE的一个实例。

### 路径积分加法GRAPE

GRAPE框架还引入了**路径积分加法GRAPE**（**Path Integral Additive GRAPE, GRAPE-AP**）的概念，进一步扩展了加法偏置的灵活性。

传统的加法偏置通常只与相对距离 $j-i$ 的线性函数有关。而GRAPE-AP允许偏置是一个沿着从位置 $j$ 到 $t$ 的路径上的“成本”累积和：




{% raw %}$$ b_h(t,j) := \sum_{\ell=j+1}^t \psi_h(t,\ell) $${% endraw %}



这里的每一步成本 $\psi\_h(t,\ell)$ 可以是与内容相关的动态值。这种机制在数学上对应于一系列单能变换矩阵的连乘积。由于这些矩阵的特殊结构，它们的连乘积最终也等价于一个简单的加法偏置，保持了计算的高效性。这为设计更加动态和内容感知的距离惩罚机制提供了理论依据。

### 实验与性能

为了验证GRAPE框架的有效性，研究者基于Llama架构进行了一系列语言建模实验。实验在一个包含500亿Token的教育网络文本数据集（FineWeb-Edu）上进行，模型规模为3.55亿参数，上下文长度为4096。

实验对比了GRAPE与RoPE、ALiBi、FoX等基线方法的性能。


| ![图：中等规模模型（355M）在FineWeb-Edu数据集上的训练和验证损失曲线](images/page_10_Figure_2.jpg) | ![图：中等规模模型（355M）在FineWeb-Edu数据集上的训练和验证损失曲线](images/page_10_Figure_4.jpg) |
| :---: | :---: |
| 图：中等规模模型（355M）在FineWeb-Edu数据集上的训练和验证损失曲线 |

从训练和验证损失曲线可以看出，GRAPE的变体在整个训练过程中始终保持着优于RoPE和FoX等基线方法的性能。

更重要的是，实验观察到，使用RoPE的模型在训练过程中出现了一定的不稳定性，而GRAPE模型则表现出持续稳定的学习过程。这从实践上印证了GRAPE框架在理论上的稳定性优势。

### 结论

GRAPE通过引入群论，为Transformer中的位置编码问题提供了一个深刻而统一的视角。它优雅地将两种主流的位置编码范式——基于旋转的乘法机制（如RoPE）和基于平移的加法机制（如ALiBi、FoX）——统一在同一个数学框架之下。

- **统一性**：GRAPE证明了RoPE是特殊正交群 $SO(d)$ 作用下的一个特例，而ALiBi和FoX则是一般线性群 $GL(d)$ 中单能变换的特例。

- **解释性**：它为这些看似经验性的方法提供了坚实的数学基础，解释了它们为何有效。

- **扩展性**：GRAPE不仅限于解释现有方法，它还提供了一个原则性的设计空间。通过探索不同的群、生成元和表示，可以系统地设计出新的、可能更强大的位置编码方案，例如使用可学习的、非通勤的旋转来捕获更复杂的特征交互。

总而言之，GRAPE不仅是一次理论上的综合，更是一张指引未来位置编码研究的蓝图。它将抽象的数学理论与具体的模型设计相结合，为构建更稳定、更强大、更具外推能力的大型语言模型铺平了道路。