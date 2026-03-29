---
layout: default
title: "Prompt Repetition Improves Non-Reasoning LLMs"
---

## 谷歌新发现：简单重复Prompt，大模型准确率暴涨且零额外延迟

<img src="/images/2512.14982v1/A__title.jpg" alt="" style="width:90%; max-width:700px; margin:auto; display:block;">

你有没有想过，提升大模型效果的最简单方法，可能不是精心设计复杂的“思维链”（CoT）提示词，也不是进行昂贵的微调，而是直接把你的Prompt“复制粘贴”一遍？

> ArXiv URL：http://arxiv.org/abs/2512.14982v1

这听起来像是一个“愚人节玩笑”，或者某种玄学，但来自 **Google Research** 的最新研究表明：这不仅有效，而且效果惊人。这篇名为《Prompt Repetition Improves Non-Reasoning LLMs》的论文揭示了一个反直觉的现象：**在不使用推理模式的情况下，简单地重复输入Prompt，可以显著提升Gemini、GPT、Claude和DeepSeek等主流模型的性能，而且几乎不增加生成延迟。**

### 为什么大模型需要“读两遍”？

要理解为什么“复读机”策略有效，我们需要回到大模型（LLM）的基本原理。

绝大多数LLM是作为**因果语言模型**（**Causal Language Models**）训练的。这意味着模型在处理文本时，只能“向后看”，无法“向前看”。过去的Token无法注意到未来的Token。

试想一下，如果你给模型一段很长的背景材料，然后在最后才提出问题（即 $$<CONTEXT> <QUESTION>$$ 结构）。当模型在阅读前面的背景材料时，它并不知道后面会问什么，因此它可能无法有效地关注到与问题相关的关键信息。

这就像你在做英语阅读理解题：如果你先读完全文再看问题，往往需要回过头重读一遍才能找到答案。

该研究提出的解决方案简单得令人发指：**Prompt Repetition**（**提示词重复**）。

即将输入从 $$<QUERY>$$ 变为 $$<QUERY><QUERY>$$。

<img src="/images/2512.14982v1/figure1_big2.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

如上图所示，通过重复Prompt，模型在处理第二遍Prompt时，实际上已经“看过了”完整的第一遍内容。这就好比让模型拥有了“全局注意力”，使得Prompt中的每一个Token都能关注到Prompt中的其他任何Token。

### 实验结果：47胜0负的压倒性优势

研究团队在7个主流模型上进行了测试，包括 **Gemini 2.0 Flash**、**GPT-4o**、**Claude 3.7 Sonnet** 以及 **DeepSeek V3**。测试覆盖了ARC、GSM8K、MMLU-Pro等7个基准测试集。

结果可以用“屠榜”来形容：

1.  **全面提升**：在不使用推理（Reasoning）模式的情况下，**Prompt Repetition** 提升了所有测试模型在所有基准上的准确率。

2.  **胜率惊人**：在70个“模型-基准”组合中，该方法赢了47次，**输了0次**（其余为持平）。

3.  **特定任务暴涨**：在一些自定义任务上，提升幅度令人咋舌。例如，在 $$NameIndex$$ 任务上，**Gemini 2.0 Flash-Lite** 的准确率从 **21.33%** 直接飙升到了 **97.33%**。

更有趣的是，研究发现这种方法对于“选项优先”（Options-first）的场景提升最大。这很好理解：如果先给选项再给问题，模型读选项时是一头雾水的；但如果重复一遍，模型在读第二遍选项时，就已经知道问题是什么了。

### 真正的“免费午餐”：效率分析

通常来说，提升模型性能往往伴随着代价，比如“思维链”（CoT）会让模型生成更多的Token，导致响应变慢、成本增加。

但 **Prompt Repetition** 几乎是“免费”的。

*   **生成长度不变**：重复输入Prompt并不会让模型生成的答案变长。

*   **延迟几乎不变**：虽然输入的Prompt变长了，但这部分计算发生在**预填充阶段**（**Prefill Stage**）。现代推理引擎对预填充阶段有极高的并行优化能力。

如下图所示，与标准的Prompt相比，重复Prompt（Prompt Repetition）在生成输出的长度和延迟上几乎没有变化。相比之下，使用“Think step by step”（Let's think）虽然也能提升性能，但会导致生成延迟大幅增加。

<img src="/images/2512.14982v1/figure1_big2.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">

这意味着，你可以在现有的系统中直接通过代码逻辑把用户的Prompt拼接两次，用户几乎感觉不到任何延迟变化，但得到的回答质量却更高了。

### 总结与启示

这项研究不仅揭示了一个简单有效的“Prompt Engineering”技巧，更让我们对大模型的注意力机制有了新的思考。

*   **简单即正义**：对于非推理模型，**Prompt Repetition** 可能应该成为一种默认的设置。

*   **无需改变架构**：你不需要重新训练模型，也不需要复杂的Agent设计，只需要在API调用前加一行字符串拼接代码。

*   **推理模型的启示**：研究还发现，经过强化学习训练的推理模型（Reasoning Models）往往会自动学会重复用户的请求。这说明“复读”本身就是一种内在的推理增强机制。

下次当你觉得大模型变“笨”了，在责怪模型之前，不妨试着把你的问题再说一遍？也许，它只是需要多读一次而已。