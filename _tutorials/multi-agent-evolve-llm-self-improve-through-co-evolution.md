---
layout: default
title: "Multi-Agent Evolve: LLM Self-Improve through Co-evolution"
---

# Multi-Agent Evolve: LLM Self-Improve through Co-evolution

- **ArXiv URL**: http://arxiv.org/abs/2510.23595v2

- **作者**: Muhan Zhang; Siqi Zhu; Yiding Wang; Haofei Yu; Jiaxuan You; Yixing Chen; Tao Feng

- **发布机构**: NVIDIA; Peking University; University of Illinois at Urbana-Champaign

---

# TL;DR
本文提出了一个名为 Multi-Agent Evolve (MAE) 的自进化框架，通过让一个大型语言模型扮演提问者、解答者和评判者三个协同进化的智能体角色，并结合强化学习，在无需人工标注或外部环境验证的情况下提升模型的通用推理能力。

# 关键定义
本文的核心是围绕一个由单一LLM实例化的多智能体系统展开的，关键定义如下：
*   **Multi-Agent Evolve (MAE)**: 一个多智能体自进化框架。它实例化三个角色（Proposer, Solver, Judge）并使用强化学习联合训练它们，形成一个闭环的自我提升系统，以增强LLM在数学、推理和通用问答等领域的任务解决能力。
*   **Proposer (提问者)**: 系统的探索驱动者。该智能体的目标是生成对当前Solver具有挑战性但本身质量很高的问题。它的训练信号来自于Judge对问题质量的评价和Solver解答该问题的失败率。
*   **Solver (解答者)**: 系统的核心能力体现者。该智能体负责解答由Proposer提出的问题。它的性能是整个框架优化的最终目标，其奖励主要来自于Judge对其答案准确性和推理过程的评分。
*   **Judge (评判者)**: 系统的通用奖励模型。该智能体基于精心设计的评估标准，对Proposer生成的问题和Solver生成的答案进行打分，为另外两个智能体提供训练所需的奖励信号，从而摆脱了对外部验证器（如代码解释器）或人工标注的依赖。
*   **Co-evolution (协同进化)**: 指Proposer、Solver和Judge三个智能体共同演化、相互促进的过程。Proposer生成更难的问题来挑战Solver，Solver能力的提升反过来促使Proposer探索更复杂的问题空间，而Judge的评估能力也随之演进，形成一个动态平衡的进化螺旋。

# 相关工作
当前，通过强化学习（Reinforcement Learning, RL）提升LLM能力的方法取得了显著成功，尤其是在编码和推理任务中。然而，这些方法严重依赖于昂贵且数量有限的人工标注数据集，或是需要一个可提供真实反馈的“接地”环境（grounded environment），如代码解释器或游戏引擎。这限制了其可扩展性和在通用开放域（如自然语言推理、通用知识问答）的应用。

最近兴起的自博弈（Self-Play）方法试图在没有人类监督的情况下提升LLM，但同样大多局限于有明确反馈机制的环境。其核心瓶颈在于：在缺乏领域特定验证的情况下，如何为通用推理任务设计有效的奖励信号，因为现实世界中大多数推理场景的奖励本身是模糊且难以量化的。

本文旨在解决这一核心问题：**如何构建一个有效的、无需人工标注的强化学习框架，使LLM能够在通用领域实现自我提升？**

# 本文方法
本文提出了Multi-Agent Evolve (MAE)框架，其核心思想是让一个共享骨干的LLM同时扮演三个不同的角色：提问者（Proposer）、解答者（Solver）和评判者（Judge）。这三个智能体通过互动形成一个闭环的自奖励和自进化系统，通过强化学习共同优化。

<img src="/images/2510.23595v2/x1.jpg" alt="MAE框架概览" style="width:85%; max-width:450px; margin:auto; display:block;">
*MAE框架通过实例化Proposer、Solver和Judge三个互动角色，形成一个闭环自提升循环。Proposer生成新问题，Solver尝试回答，Judge评估两者的产出以提供通用领域的奖励信号。Solver因准确推理而获奖励，而Proposer则同时获得来自Judge的质量奖励和在Solver失败时增加的难度奖励，从而创造了一个对抗性的协同进化过程，持续增强模型的推理能力。*

## 核心智能体角色与设计

### 提问者 (The Proposer)
Proposer的目标是提出既有质量又具挑战性的问题。它可以通过参考现有问题库中的问题（$$with reference$$）或完全从零开始（$$from scratch$$）生成新问题 $$q$$。

**奖励设计**: Proposer的总奖励 $$R_P(q)$$ 由三部分加权组成：
1.  **质量奖励 (Quality Reward)**: 由Judge评估生成问题的清晰度、逻辑性和可解性等内在质量。
2.  **难度奖励 (Difficulty Reward)**: 该奖励与Solver解答该问题的平均得分成反比，即 $$R_difficulty(q) = 1 - R̄_S(q)$$。$$R̄_S(q)$$ 是Solver多次尝试解答该问题的平均得分。这激励Proposer生成对当前Solver来说更难的问题，形成对抗性驱动。
3.  **格式奖励 (Format Reward)**: 确保其输出遵循预定义的标签格式（如 $$<question>...</question>$$），保证系统能自动解析。




{% raw %}$$
R_{P}(q)=\lambda_{quality}R_{quality}+\lambda_{difficulty}R_{difficulty}+\lambda_{format}R_{format}
$${% endraw %}



**质量过滤 (Quality Filtering)**: 为了防止训练过程中问题质量下降导致模型能力退化，MAE引入了质量过滤机制。只有当Judge对一个新生成问题的质量评分高于某个阈值（例如0.7）时，该问题才会被加入到有效问题池中，用于后续训练。

<img src="/images/2510.23595v2/x2.jpg" alt="MAE训练细节" style="width:85%; max-width:600px; margin:auto; display:block;">
*(上图) MAE使用骨干LLM自身作为问题和答案的通用评估器，增强了智能体间的互动并适应通用任务。(左下图) 框架在Proposer的生成循环中应用质量过滤，防止在长期训练中数据集质量下降。(右下图) 多智能体训练采用Task-Relative REINFORCE++，为每个角色分别计算优势函数，然后对统一模型进行同步参数更新。*

### 解答者 (The Solver)
Solver是框架的核心，其任务是为Proposer提出的问题 $$q$$ 生成高质量的答案 $$a$$。框架的最终目的是提升Solver的能力。

**奖励设计**: Solver的总奖励 $$R_S(a)$$ 由两部分加权组成：
1.  **评判奖励 (Judge Reward)**: 这是最主要的奖励来源。Judge会评估答案 $$a$$ 的正确性、完整性和推理过程，并给出一个分数 $$V_J(a,q)$$。
2.  **格式奖励 (Format Reward)**: 同样地，确保Solver的输出遵循预定义的标签格式（如 $$<answer>...</answer>$$）。




{% raw %}$$
R_{S}(a)=\lambda_{judge}R_{judge}+\lambda_{format}R_{format}
$${% endraw %}



### 评判者 (The Judge)
Judge扮演了生成式奖励模型的角色，为Proposer和Solver提供训练信号。它通过思维链（Chain-of-Thought）的方式，先生成详细的分析过程，再给出最终分数，以保证评估的合理性。

*   **评价答案**: Judge根据严格的评分标准（rubric）评估答案。事实、逻辑或计算上的任何错误都会导致分数低于3分（满分10分），而只有完美无瑕的答案才能获得高分。
*   **评价问题**: 同样，Judge也根据一套标准评估问题的质量，例如问题的可解性、逻辑一致性等。
*   **格式奖励 (Format Reward)**: Judge自身也需要被训练，以确保其输出的分数（如 $$<score>X</score>$$）是稳定且可解析的。因此，它会因正确格式化其输出而获得奖励。这对于自动化整个自博弈循环至关重要。

## 协同训练流程
如算法1所示，MAE的每一步训练循环如下：
1.  **提问阶段**: Proposer从问题池中采样参考问题或从零开始，生成一批新问题。
2.  **评估与过滤**: Judge评估这些新问题的质量。只有高质量的问题会被加入问题池。
3.  **解答阶段**: Solver从问题池中采样问题，并生成答案。
4.  **奖励计算**: Judge评估Solver的答案，并为Proposer和Solver计算各自的总奖励。同时，Judge也因其输出格式的正确性获得奖励。
5.  **同步更新**: 使用Task-Relative REINFORCE++算法，根据所有智能体在本轮收集到的奖励信号和梯度，同步更新共享的LLM模型参数。

通过这种方式，三个智能体协同进化，不断推动模型能力的边界。

# 实验结论
本文在Qwen2.5-3B-Instruct模型上进行了实验，以验证MAE框架的有效性。

**关键实验结果**:

*   **无需参考数据的自进化**: 在仅使用极少量（16个）自生成问题作为初始引导的“零参考”设置（$$MAE (zero)$$）下，MAE依然能在数学、常识问答和阅读理解等多个基准测试上取得显著优于基线模型的性能，平均分提升了$$3.18%$$。同时，它也全面超越了另一个强大的自博弈基线$$AZR$$，尤其在$$AZR$$表现不佳的复杂推理任务（如BBH, AMC）上优势明显。这证明了MAE框架仅靠内部的多角色协同进化就能实现通用能力的提升。
*   **利用参考数据的进化**: 当引入少量（约1k）无标签的真实世界问题作为参考时，MAE的性能得到进一步放大。有趣的是，标准的有监督微调（SFT）方法在使用了这些问题的真实答案进行训练后，性能反而出现了下降（可能是由于数据分布广泛且数量有限）。相比之下，所有MAE变体都显著优于SFT。
*   **最佳策略**: 在所有设置中，$$MAE (half reference)$$策略取得了最佳效果。该策略以50%的概率参考现有问题、50%的概率从零探索新问题，实现了在利用现有数据分布和探索新问题空间之间的最佳平衡。它在分布内（ID）和分布外（OOD）测试集上均取得了最高的平均分，总体平均准确率相较于基线模型提升了$$4.54%$$。


| Model | ID Avg. | OOD Avg. | Overall Avg. |
| :--- | :---: | :---: | :---: |
| ***w/o reference questions*** | | | |
| Base | 64.91 | 39.54 | 55.33 |
| AZR (Zhao et al., 2025) | 64.67 | 46.99 | 57.72 |
| MAE (zero) | **65.51** | **47.67** | **58.51** |
| ***w/ reference questions*** | | | |
| Base | 64.91 | 39.54 | 55.33 |
| SFT | 63.28 | 37.41 | 53.87 |
| MAE (no reference) | 65.57 | 43.68 | 58.18 |
| MAE (with reference) | 65.07 | 43.18 | 58.07 |
| MAE (half reference) | **68.95** | **43.96** | **59.87** |

*备注：ID表示分布内数据集，OOD表示分布外数据集。*

**训练稳定性**:
与一些容易在训练中崩溃的LLM交互方法不同，MAE框架表现出良好的训练稳定性，能够持续训练超过250个步骤。分析表明，在训练过程中，高质量问题能被持续地添加到问题数据集中，这得益于质量过滤机制和三智能体之间的良性互动。这表明MAE具备良好的可扩展性，并且可能在更大的模型上取得更显著的效果。

<img src="/images/2510.23595v2/x3.jpg" alt="训练过程分析" style="width:90%; max-width:700px; margin:auto; display:block;">
*(左图) 数据集中的问题数量稳定增加，同时低质量问题被有效过滤。(中图和右图) Proposer学会了生成对Solver具有适当挑战性的问题，这有利于模型在后续训练中的持续进步。*

**最终结论**:
实验结果有力地证明，Multi-Agent Evolve是一个可扩展且数据高效的框架。它通过内部的多智能体协同进化，成功地在几乎不依赖人工监督的情况下，提升了LLM的通用推理能力，为超越人类标注数据限制的AGI发展路径提供了有前景的探索。