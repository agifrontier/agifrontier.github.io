---
layout: default
title: "Learning to Focus: Focal Attention for Selective and Scalable Transformers"
---

## Learning to Focus: Focal Attention for Selective and Scalable Transformers

- **ArXiv URL**: http://arxiv.org/abs/2511.06818v1

Transformer模型好比一个读书人，注意力（Attention）机制就是他读书时划重点的能力。但标准的注意力机制有个毛病，它像个“大水漫灌”的工具，划重点的时候，会顺便把一些不那么重要的词也捎上，造成了**注意力分散**。尤其当文章（上下文）很长时，这种干扰就更严重了。

这篇论文提出的**Focal Attention**，就是要解决这个问题。它对注意力机制做了一个简单而巧妙的改造，让模型能**更集中地关注**最重要的信息。

<img src="/images/2511.06818v1/intro_focal.jpg" alt="标准注意力与Focal Attention对比" style="width:85%; max-width:600px; margin:auto; display:block;">
上图直观展示了两者的区别。左边是标准注意力，权重分散；右边是Focal Attention，权重高度集中在关键Token上。

### Focal Attention：给注意力加上一个“聚焦镜”

标准的注意力计算，可以简单理解为用Softmax函数给每个词分配一个权重。




{% raw %}$$
P_{i}=\frac{exp(z_{i})}{\sum_{j=1}^{n}(exp(z_{j}))}
$${% endraw %}



这里的 $z\_i$ 是模型算出来的一个分数，分数越高，代表这个词越重要。

Focal Attention的核心思想，是在Softmax函数里引入一个**温度参数** $t$。




{% raw %}$$
P_{i}=\frac{exp(z_{i}/t)}{\sum_{j=1}^{n}(exp(z_{j}/t))}
$${% endraw %}



这个温度 $t$ 就像一个“聚焦镜”的旋钮。

当 $t < 1$ 时，它会放大分数之间的差距，让最高的分数更加突出，权重更集中。这就好比把镜头的焦点调得更锐利，只让最重要的物体清晰，其余的都虚化掉。

本文提出了两种实现Focal Attention的方法。

#### 恒定温度（Constant Temperature）

这是最简单的一种方式。在标准的注意力公式里，本来就有一个缩放因子 $\sqrt{d}$，这里的 $d$ 是维度的意思。我们可以把它看成一个默认的温度。

标准注意力公式：


{% raw %}$$
Attention(X)=softmax{(\frac{QK^{T}}{\sqrt{d}})}V
$${% endraw %}



Focal Attention则额外引入一个恒定的温度参数 $t$：


{% raw %}$$
Attention(X)=softmax{(\frac{QK^{T}}{t\sqrt{d}})}V
$${% endraw %}



这个 $t$ 是一个**固定的超参数**（比如设为0.4）。训练开始前就定好了，整个模型从头到尾都用这一个“焦距”。

#### 可学习温度（Learned Temperature）

这种方式更灵活。它不再使用一个固定的温度，而是让模型**自己学着调节**。

每一层的注意力模块会根据当前的输入 $X$ 动态计算出一个最合适的温度 $\tau$。




{% raw %}$$
\tau=clip(mean(Xw_{\tau}),\tau_{min},\tau_{max})
$${% endraw %}



然后用这个动态的 $\tau$ 来计算注意力：


{% raw %}$$
Attention(X)=softmax{(\frac{QK^{T}}{\tau})}V
$${% endraw %}



这就好比给模型一个自动对焦的镜头。在处理不同信息时（比如模型的底层和高层），它可以自动调整焦距，决定是看得宽泛一些，还是聚焦得更紧一些。

### 实验效果如何？

本文通过一系列实验，证明了Focal Attention的出色效果。实验使用了标准的LLaMA架构，训练了从4亿到95亿参数不等的多个模型。

#### 更强的扩展性

Focal Attention在模型扩展性上表现优异，无论是在模型尺寸、训练数据量还是上下文长度方面。

*   **模型尺寸**：要达到同等性能，Focal Attention模型所需的**参数量比标准模型少42%**。这意味着更小的模型就能办成同样的事。

*   **训练数据**：要达到同等性能，Focal Attention所需**训练数据量减少了33%**。这意味着它学习效率更高，更省资源。

<img src="/images/2511.06818v1/scale_tokens_tasks.jpg" alt="训练数据扩展性" style="width:85%; max-width:600px; margin:auto; display:block;">

*   **上下文长度**：随着上下文长度从2048增加到8192，Focal Attention的优势愈发明显，损失下降得更多。

#### 下游任务表现

在一系列常识推理任务上，27亿参数的Focal Attention模型比标准模型平均**绝对提升了2.2个点**。


| 模型 | ARC-E | ARC-C | BoolQ | HellaSwag | LAMBADA | PIQA | Winogrande | 平均 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 基线模型 | 78.4 | 51.6 | 82.2 | 82.8 | 80.6 | 82.8 | 77.2 | 76.5 |
| Focal (恒定) | **81.1** | **53.3** | **83.8** | **83.6** | 80.7 | **83.2** | **79.4** | **77.9** |
| Focal (可学习) | 80.2 | 52.2 | 83.3 | 82.8 | **81.1** | 82.4 | 77.5 | 77.1 |

从上表可以看出，**恒定温度**的版本效果最好。

#### 长上下文能力

Focal Attention的真正威力体现在长上下文任务中。作者们在最高64K的上下文长度上进行了测试，涵盖了情境学习（ICL）、检索增强生成（RAG）、长文档问答等。

结果显示，Focal Attention在大多数长上下文任务中都**显著优于**标准模型，相对提升从**17%到82%**不等。

<img src="/images/2511.06818v1/nlu.jpg" alt="情境学习（ICL）任务表现" style="width:85%; max-width:600px; margin:auto; display:block;">

上图展示了在ICL任务上的表现，随着上下文中示例数量（即上下文长度）的增加，Focal Attention的性能优势持续扩大。这证明了它在处理长篇信息时，能够更有效地抓住关键点，过滤掉噪声。

### 消融研究

*   **温度怎么选？** 实验发现，对于恒定温度， $t=0.4$ 是一个比较好的选择。对于可学习温度，将最低温度 $\tau\_{min}$ 设置为5效果最佳。这表明，**注意力不是越锐利越好**，太强的聚焦反而会限制模型的能力。

*   **老模型能用吗？** 作者尝试在一个已经训练好的标准模型上直接应用Focal Attention并继续微调。结果发现性能有所提升，但**不如从头开始就用Focal Attention进行训练的模型**。这说明，要想发挥最大威力，最好在预训练阶段就让模型学会“聚焦”。

### 总结

Focal Attention是一个简单却非常有效的创新。它通过**控制Softmax的温度**来锐化注意力分布，好比给Transformer装上了一个“聚焦镜”。

这个改动让模型能够更精准地捕获关键信息，忽略无关噪声。

最终的好处是：
1.  **更高效**：用更少的参数和数据就能达到同等效果。
2.  **更强大**：在处理长上下文任务时，性能提升巨大。

由于其简单性和有效性，Focal Attention有望成为未来大模型架构的一个标准组件。