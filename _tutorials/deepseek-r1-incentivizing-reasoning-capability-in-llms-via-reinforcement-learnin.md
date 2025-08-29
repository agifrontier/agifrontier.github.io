---
layout: default
title: "DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning"
---

# DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning

- **ArXiv URL**: http://arxiv.org/abs/2501.12948v1

- **作者**: K. Yu; Bei Feng; Yuting Yan; Yanping Huang; Shiyu Wang; Jingchang Chen; Xiaodong Liu; Yu-Wei Luo; Jingyang Yuan; Zhean Xu; 等186人

- **发布机构**: DeepSeek

---

# TL;DR
本文通过大规模强化学习（Reinforcement Learning, RL）成功激发了大型语言模型的推理能力，提出了两种方法：DeepSeek-R1-Zero（纯RL）和DeepSeek-R1（多阶段训练），证明了仅通过RL即可自发涌现强大推理能力，并通过多阶段优化和蒸馏，实现了与顶尖模型相媲美的性能，并将强大的推理能力迁移到更小的模型上。

# 关键定义
*   **DeepSeek-R1-Zero**: 本文提出的第一代推理模型，它不经过任何监督微调（Supervised Fine-tuning, SFT）作为预备步骤，直接在基础模型上通过大规模纯强化学习进行训练。这个模型证明了LLM仅凭RL就能自发学习并演化出强大的推理行为。
*   **DeepSeek-R1**: 在DeepSeek-R1-Zero基础上进行改进的第二代模型。它采用了一个多阶段训练流程，包括使用少量高质量“冷启动”数据进行SFT，然后进行面向推理的RL，接着通过拒绝采样生成新的SFT数据（覆盖推理和通用领域）再次进行SFT，最后进行一轮全场景RL。此方法旨在解决R1-Zero的可读性问题并进一步提升性能。
*   **Group Relative Policy Optimization (GRPO)**: 本文采用的一种强化学习算法。它的核心特点是放弃了与策略模型同样大小的评论家（critic）模型，而是通过对一组采样输出的分数进行计算来估计基线（baseline），从而显著节省训练成本。
*   **“Aha Moment”**: 在训练DeepSeek-R1-Zero过程中观察到的一种有趣现象。模型会自发地“意识到”初步解法可能存在问题，然后暂停、重新评估问题，并投入更多的“思考时间”来探索新的解题路径。这体现了模型通过RL实现了高级的自我反思和策略调整能力。

# 相关工作
当前，训练后（post-training）阶段已成为提升大型语言模型（Large Language Models, LLMs）能力的关键环节。特别是对于推理能力，OpenAI的o1系列模型通过在推理时延长思维链（Chain-of-Thought, CoT）的长度，在数学、编码和科学推理等任务上取得了显著进展。然而，如何有效地实现测试时（test-time）的扩展仍然是一个开放性问题。

现有的方法包括基于过程的奖励模型（process-based reward models）、强化学习和蒙特卡洛树搜索（Monte Carlo Tree Search）等搜索算法。尽管这些方法取得了一定的成果，但没有一个能够在通用推理性能上达到与OpenAI o1系列模型相媲美的水平。

本文旨在解决的核心问题是：能否不依赖大量监督数据，仅通过纯粹的强化学习来激发和提升LLM的推理能力，并探索出一条能够训练出与业界顶尖模型性能相当的推理模型的有效路径。

# 本文方法

## 概述
本文的核心方法是探索通过大规模强化学习显著提升模型的推理能力。研究者们提出了两种具体的实现路径：一是不依赖任何监督数据的纯RL方法（DeepSeek-R1-Zero）；二是在此基础上，引入少量“冷启动”数据并结合多阶段训练的优化方法（DeepSeek-R1）。此外，本文还验证了将强大模型的能力蒸馏到小模型上的可行性。

## DeepSeek-R1-Zero: 纯粹的强化学习
该方法旨在探究LLM在没有监督数据的情况下，仅通过纯RL过程实现自我演化的潜力。

### 强化学习算法
为了节省训练成本，本文采用了**GRPO (Group Relative Policy Optimization)** 算法。该算法无需训练一个独立的评论家模型，而是从策略模型采样的一组输出中估计优势函数。对于每个问题 $q$，从旧策略 $\pi\_{\theta\_{old}}$ 中采样一组输出 $\{o\_1, o\_2, \dots, o\_G\}$，然后通过最大化以下目标函数来优化策略模型 $\pi\_{\theta}$：


{% raw %}$$
\begin{split}\mathcal{J}_{GRPO}(\theta)&=\mathbb{E}{[q\sim P(Q),\{o_{i}\}_{i=1}^{G}\sim\pi_{\theta_{old}}(O \mid q)]}\\ &\frac{1}{G}\sum_{i=1}^{G}\left(\min\left(\frac{\pi_{\theta}(o_{i} \mid q)}{\pi_{\theta_{old}}(o_{i} \mid q)}A_{i},\text{clip}\left(\frac{\pi_{\theta}(o_{i} \mid q)}{\pi_{\theta_{old}}(o_{i} \mid q)},1-\varepsilon,1+\varepsilon\right)A_{i}\right)-\beta\mathbb{D}_{KL}\left(\pi_{\theta} \mid  \mid \pi_{ref}\right)\right),\end{split}
$${% endraw %}


其中，优势 $A\_i$ 由该组输出对应的奖励 $\{r\_1, r\_2, \dots, r\_G\}$ 计算得出：


{% raw %}$$
A_{i}=\frac{r_{i}-{\mathrm{mean}(\{r_{1},r_{2},\cdots,r_{G}\})}}{{\mathrm{std}(\{r_{1},r_{2},\cdots,r_{G}\})}}.
$${% endraw %}



### 奖励模型
奖励信号决定了RL的优化方向。DeepSeek-R1-Zero采用了一个基于规则的奖励系统，主要包含两类奖励：
*   **准确性奖励**：评估模型输出的最终答案是否正确。例如，对于数学问题，通过检查模型在指定格式（如方框内）给出的答案是否与标准答案一致来判断。
*   **格式奖励**：强制模型将其思考过程置于 $$<think>$$ 和 $$</think>$$ 标签之间，以规范输出结构。
本文特意未使用基于神经网络的奖励模型，以避免在大规模RL过程中可能出现的“奖励被黑”（reward hacking）问题，并简化训练流程。

### 训练模板与自主演化
训练时使用了一个非常简单的模板，引导模型先输出思考过程，再输出最终答案。


| 用途 | 模板内容 |
| --- | --- |
| DeepSeek-R1-Zero训练模板 | A conversation between User and Assistant...<br>...<br>User: prompt.<br>Assistant: $$<think>$$ reasoning process here $$</think>$$ $$<answer>$$ answer here $$</answer>$$ |

*表1: DeepSeek-R1-Zero的训练模板*

在纯RL训练过程中，DeepSeek-R1-Zero自发地演化出了一系列强大的推理行为：
*   **思考时间延长**：模型自然地学会为复杂问题分配更多的计算资源，生成长达数千个token的推理链。
*   **复杂行为涌现**：随着“思考时间”的增加，模型自发出现了**反思**（reflection）和探索不同解题路径等高级行为。
*   **“Aha Moment”**：在训练中途，模型会像人类一样突然“醒悟”，意识到初步方法不妥，并主动重新评估和调整策略。


| 示例：“Aha Moment” |
| --- |
| **问题**: If $a>1$, then the sum of the real solutions of $\sqrt{a-\sqrt{a+x}}=x$ is equal to |
| **模型响应**: $$<think>$$<br>To solve the equation... let’s start by squaring both sides...<br>$x^{4}-2ax^{2}-x+(a^{2}-a)=0$<br>...<br>**Wait, wait. Wait. That’s an aha moment I can flag here.**<br>Let’s reevaluate this step-by-step...<br>We started with the equation:<br>$\sqrt{a-\sqrt{a+x}}=x$<br>First, let’s square both sides:<br>$a-\sqrt{a+x}=x^{2}\implies\sqrt{a+x}=a-x^{2}$<br>... |

*表3: DeepSeek-R1-Zero中间版本出现的有趣“Aha Moment”*

尽管DeepSeek-R1-Zero推理能力强大，但其输出存在可读性差和语言混杂（中英文混合）等问题。

## DeepSeek-R1: 带冷启动的多阶段强化学习
为了解决R1-Zero的缺点并进一步提升性能，本文设计了DeepSeek-R1的四阶段训练流程。

1.  **阶段一：冷启动 (Cold Start)**
    收集数千条高质量、人类可读的长思维链（CoT）数据，对基础模型（DeepSeek-V3-Base）进行SFT。这些数据通过精心设计的模板保证了可读性，为后续RL提供了一个更好的起点。

2.  **阶段二：面向推理的强化学习 (Reasoning-oriented RL)**
    在SFT后的模型上，应用与R1-Zero类似的RL训练，专注于提升在数学、编码、科学等领域的推理能力。为了解决语言混杂问题，额外引入了一个**语言一致性奖励**，惩罚非目标语言的输出。

3.  **阶段三：拒绝采样与监督微调 (Rejection Sampling & SFT)**
    当RL模型收敛后，利用它通过拒绝采样生成约60万条高质量的推理相关SFT样本。同时，结合了约20万条来自DeepSeek-V3的写作、问答等非推理任务的SFT数据。使用这总计约80万的样本，对DeepSeek-V3-Base模型进行重新微调，以兼顾推理能力和通用能力。

4.  **阶段四：全场景强化学习 (RL for all Scenarios)**
    对上一阶段微调后的模型进行最后一次RL。这次RL结合了两种奖励信号：对推理数据使用基于规则的奖励，对通用数据则使用奖励模型来对齐人类偏好（如有用性和无害性），从而打造一个在推理和通用场景下都表现出色的模型。

## 蒸馏：让小模型也具备强大推理能力
本文将在DeepSeek-R1训练过程中收集的80万个SFT样本，直接用于微调多个开源的小型模型（如Qwen和Llama系列）。这种简单直接的蒸馏方法证明，可以将大模型探索出的高级推理模式有效地迁移给小模型。

# 实验结论
<img src="/images/2501.12948v1/x1.jpg" alt="Benchmark performance of DeepSeek-R1." style="width:85%; max-width:600px; margin:auto; display:block;">
*图1: DeepSeek-R1在基准测试上的性能表现*

### 核心模型性能
*   **DeepSeek-R1-Zero**:
    *   在纯RL训练下，性能持续提升，AIME 2024测试集上的pass@1准确率从15.6%提升到71.0%。
    *   通过多数投票（cons@64），AIME得分可达86.7%，与OpenAI-o1-0912相当。这证明了纯RL激发推理能力的有效性。

    <img src="/images/2501.12948v1/plot_aime_with_maj.jpg" alt="AIME accuracy of DeepSeek-R1-Zero during training." style="width:85%; max-width:600px; margin:auto; display:block;">
    *图2: DeepSeek-R1-Zero训练过程中在AIME上的准确率变化*

*   **DeepSeek-R1**:
    *   在推理任务上表现卓越：在AIME 2024上取得79.8% (Pass@1)，略超OpenAI-o1-1217；在MATH-500上达到97.3%，与o1-1217持平。
    *   在编码任务上达到专家水平，Codeforces评分达到2029，超过96.3%的人类参赛者。
    *   通用能力同样强大：在AlpacaEval 2.0和ArenaHard等评估中取得高胜率，同时在MMLU、GPQA等知识密集型任务上显著优于其基础模型DeepSeek-V3。


| 模型 | AIME 2024 (pass@1) | MATH-500 (pass@1) | LiveCodeBench (Pass@1-COT) | Codeforces (Rating) | MMLU (Pass@1) | GPQA Diamond (Pass@1) |
| --- | --- | --- | --- | --- | --- | --- |
| Claude-3.5-Sonnet-1022 | 16.0 | 78.3 | 38.9 | 717 | 88.3 | 65.0 |
| GPT-4o-0513 | 9.3 | 74.6 | 32.9 | 759 | 87.2 | 49.9 |
| OpenAI-o1-mini | 63.6 | 90.0 | 53.8 | 1820 | 85.2 | 60.0 |
| OpenAI-o1-1217 | 79.2 | 96.4 | 63.4 | 2061 | 91.8 | 75.7 |
| **DeepSeek-R1** | **79.8** | **97.3** | **65.9** | **2029** | **90.8** | **71.5** |

*表4: DeepSeek-R1与其他代表性模型的性能对比（部分）*

### 蒸馏模型性能
*   **蒸馏效果显著**: 仅通过SFT进行蒸馏，小模型性能得到巨大提升。例如，$$DeepSeek-R1-Distill-Qwen-7B$$在多个推理基准上全面超越了GPT-4o等通用强模型。$$DeepSeek-R1-Distill-Qwen-14B$$的表现超过了更强的开源模型$$QwQ-32B-Preview$$。
*   **蒸馏优于从零RL**: 实验证明，在32B尺寸的模型上，直接蒸馏DeepSeek-R1的知识（$$DeepSeek-R1-Distill-Qwen-32B$$）比在该模型上从头开始进行大规模RL训练（$$DeepSeek-R1-Zero-Qwen-32B$$）效果更好。这表明，由更强大的基础模型发现的推理模式对于提升小模型能力至关重要。


| 模型 | AIME 2024 (pass@1) | MATH-500 (pass@1) | GPQA Diamond (pass@1) | LiveCodeBench (pass@1) | CodeForces (rating) |
| --- | --- | --- | --- | --- | --- |
| OpenAI-o1-mini | 63.6 | 90.0 | 60.0 | 53.8 | 1820 |
| QwQ-32B-Preview | 50.0 | 90.6 | 54.5 | 41.9 | 1316 |
| **DeepSeek-R1-Distill-Qwen-14B** | **69.7** | **93.9** | **59.1** | **53.1** | **1481** |
| **DeepSeek-R1-Distill-Qwen-32B** | **72.6** | **94.3** | **62.1** | **57.2** | **1691** |

*表5: DeepSeek-R1蒸馏模型与其它模型的性能对比（部分）*

### 结论
本文成功证明了大规模强化学习是激发和增强LLM推理能力的一条有效且强大的路径。最终得到的DeepSeek-R1模型在多个推理基准上达到了与业界最顶尖模型相媲美的性能。此外，研究表明，这些通过复杂训练获得的推理能力可以被高效地蒸馏到更小的模型中，为社区提供了创建高性价比推理模型的捷径。尽管如此，模型在某些软件工程任务、多语言处理和提示词敏感性方面仍有待改进。