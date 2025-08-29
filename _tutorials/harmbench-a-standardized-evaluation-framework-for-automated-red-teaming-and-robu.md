---
layout: default
title: "HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal"
---

# HarmBench: A Standardized Evaluation Framework for Automated Red Teaming and Robust Refusal

- **ArXiv URL**: http://arxiv.org/abs/2402.04249v2

- **作者**: Long Phan; Andy Zou; Xuwang Yin; Zifan Wang; Elham Sakhaee; Dan Hendrycks; Steven Basart; Mantas Mazeika; Norman Mu; Bo Li; 等2人

- **发布机构**: Carnegie Mellon University; Center for AI Safety; Microsoft; University of California, Berkeley; University of Illinois Urbana-Champaign

---

# TL;DR
本文提出了HarmBench，一个用于自动化红队攻防的标准化评估框架，并通过大规模实验揭示了当前攻防方法的局限性，同时提出了一种高效的对抗训练方法R2D2，显著提升了大型语言模型的安全鲁棒性。

# 关键定义
*   **自动化红队评估 (Automated Red Teaming Evaluation)**: 指使用自动化方法生成对抗性测试用例，以发现和评估大型语言模型（LLMs）在安全和对齐方面的漏洞。本文的核心是为这一过程提供标准化的基准。
*   **鲁棒性拒绝 (Robust Refusal)**: 指模型在面对各种复杂的、对抗性的恶意指令时，能够稳定、可靠地识别并拒绝执行有害行为的能力。这是本文防御方法的核心目标。
*   **HarmBench**: 本文提出的一个标准化的评估框架，包含一套精心设计的有害行为集合和一个鲁棒的评估流程，旨在实现对自动化红队攻击方法和LLM防御能力的公平、全面比较。
*   **功能性类别 (Functional Categories)**: HarmBench中对有害行为的一种创新分类方式，根据行为的结构特性（如是否需要上下文、是否涉及多模态信息）将其分为标准行为、版权行为、上下文行为和多模态行为，用以测试模型不同维度的鲁棒性。
*   **R2D2 (Robust Refusal Dynamic Defense)**: 本文提出的一种新颖、高效的对抗训练方法。它通过维护一个由强攻击方法（如GCG）持续更新的“持久化测试用例池”来训练模型，以实现鲁棒性拒绝。

# 相关工作
目前，大型语言模型的恶意使用风险日益受到关注，自动化红队作为发现和修复模型安全漏洞的关键技术，发展迅速。然而，该领域的研究现状存在明显瓶颈：

*   **缺乏标准化评估**：以往的自动化红队研究各自为战，使用了至少9种不同的评估设置（如表1所示），这些评估在方法、数据集和指标上几乎没有重叠。这导致不同论文提出的攻击方法之间难以进行公平比较，严重阻碍了领域的进展。
*   **评估指标不可靠**：现有评估方法存在设计缺陷。例如，攻击成功率（Attack Success Rate, ASR）这一关键指标对生成文本的长度等未标准化的超参数极为敏感（如图2所示），导致跨论文的比较结果几乎没有意义。此外，许多评估使用的分类器不够鲁棒，容易被“先拒绝后妥协”等对抗性输出所欺骗。

本文旨在解决上述问题，即创建一个**全面、可复现、指标鲁棒的标准化评估框架（HarmBench）**，以便对现有的和未来的红队攻击与防御方法进行公平、大规模的比较。


| Paper | Methods Compared | Evaluation |
| :--- | :--- | :--- |
| Perez et al. (2022) | 1, 2, 3, 4 | A |
| GCG (Zou et al., 2023) | 5, 6, 7, 8 | B |
| Persona (Shah et al., 2023) | 9 | C |
| Liu et al. (2023c) | 10 | D |
| PAIR (Chao et al., 2023) | 5, 11 | E |
| TAP (Mehrotra et al., 2023) | 5, 11, 12 | E |
| PAP (Zeng et al., 2024) | 5, 7, 11, 13, 14 | F |
| AutoDAN (Liu et al., 2023b) | 5, 15 | B, G |
| GPTFUZZER (Yu et al., 2023) | 5, 16, 17 | H |
| Shen et al. (2023a) | 18 | I |


<figcaption>Table 1: 先前的自动化红队研究使用不同的评估流程，导致比较困难。</figcaption>

# 本文方法

## HarmBench框架设计
为了解决现有评估的缺陷，本文从广度、可比性和鲁棒指标三个维度系统地设计了HarmBench框架。

### 广度 (Breadth)
HarmBench包含510个独特的有害行为，远超以往任何评估。这些行为被精心组织成两个维度：
*   **语义类别**：包括网络犯罪、生化武器、版权侵犯、虚假信息等7个类别，覆盖了主流关注的风险领域。
*   **功能类别**：
    *   **标准行为 (Standard behaviors)**：传统的、自包含的有害指令。
    *   **版权行为 (Copyright behaviors)**：要求模型生成受版权保护的内容，使用新颖的基于哈希的分类器进行精确检测。
    *   **上下文行为 (Contextual behaviors)**：包含上下文信息和依赖于该上下文的有害请求，更贴近真实场景，且难以通过搜索引擎完成，具有更强的“差异化危害”。
    *   **多模态行为 (Multimodal behaviors)**：包含图像和相关的有害指令，用于评估多模态模型的安全性。

<img src="/images/2402.04249v2/x4.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">
<figcaption>Figure 4: 上下文和多模态功能类别的行为示例。与标准行为不同，它们包含特定的上下文或视觉输入。</figcaption>

### 可比性 (Comparability)
为了确保不同方法之间的比较是公平有效的，HarmBench采取了关键的标准化措施。
*   **标准化评估流程**：定义了从生成测试用例、模型生成响应到评估的完整三步流程。
*   **固定关键参数**：特别是将模型生成响应的Token数量固定为512。如下图所示，这一参数的微小变动曾导致ASR产生高达30%的差异，标准化后消除了这一不确定性。

<img src="/images/2402.04249v2/x2.jpg" alt="Refer to caption" style="width:85%; max-width:450px; margin:auto; display:block;">
<figcaption>Figure 2: 评估期间目标模型生成的Token数量极大地影响ASR，但这一参数在先前工作中未被标准化，导致跨论文比较具有误导性。</figcaption>

### 鲁棒指标 (Robust Metrics)
为了使评估结果可信且不易被“攻击”，HarmBench设计了更鲁棒的评估指标。
*   **鲁棒分类器**：针对非版权行为，本文微调了一个Llama 2 13B模型作为分类器。该分类器在处理“先拒绝后执行”等复杂场景时表现优于现有分类器（包括GPT-4）。对于版权行为，则采用基于哈希的分类器进行严格判断。
*   **数据划分**：提供了官方的验证集和测试集划分，要求所有方法开发都不能使用测试集，以防止对评估指标的过度优化（gaming）。

<img src="/images/2402.04249v2/x3.jpg" alt="Refer to caption" style="width:85%; max-width:600px; margin:auto; display:block;">
<figcaption>Figure 3: HarmBench的标准化评估流程图。</figcaption>

## R2D2：高效对抗训练方法
为了展示HarmBench如何推动攻防协同发展，本文提出了一种名为R2D2（Robust Refusal Dynamic Defense）的高效对抗训练方法。

### 创新点
传统对抗训练直接使用如GCG这样的强攻击方法，但其生成单个样本耗时极长（例如在A100上需20分钟），不适用于大规模训练。R2D2的核心创新在于借鉴了计算机视觉领域的快速对抗训练思想，引入**持久化测试用例 (persistent test cases)**机制。

该机制维护一个固定的测试用例池。在每次训练迭代中，仅从池中采样一小批测试用例，并在当前模型上用GCG进行**少量步骤的持续优化**，而不是每次都从头生成。这极大地提高了训练效率。

### 算法核心
R2D2的训练过程结合了多种损失函数和机制：
*   **对抗性优化**：使用GCG持续更新测试用例池中的样本，使其始终对当前模型保持较强的攻击性。
*   **双重损失函数**：
    *   **推离损失 ($\mathcal{L}\_{\text{away}}$)**：最大化模型“不生成”有害目标字符串的概率，直接对抗GCG的优化目标。公式为：$$ \mathcal{L}_{\text{away}}=-1\cdot\log\left(1-f_{\theta}(t_{i}\mid x_{i})\right) $$
    *   **拉近损失 ($\mathcal{L}\_{\text{toward}}$)**：引导模型在面对攻击时生成一个标准的、固定的拒绝回复。公式为：$$ \mathcal{L}_{\text{toward}}=-1\cdot\log f_{\theta}(t_{\text{refusal}}\mid x_{i}) $$
*   **效用保持**：同时引入标准的监督微调损失 ($\mathcal{L}\_{\text{SFT}}$) 在一个良性的指令微调数据集上进行训练，以保持模型的通用对话能力。
*   **多样性增强**：每隔一定迭代次数，随机重置池中一部分测试用例，以防止模型对特定攻击模式过拟合。

完整的算法流程如下所示：
Algorithm 1  Robust Refusal Dynamic Defense

Input: ${(x\_{i}^{(0)},t\_{i})\mid 1\leq i\leq N}$, $\theta^{(0)}$, $M$, $m$, $n$, $K$, $L$

Output: Updated model parameters $\theta$

Initialize test case pool $P={(x\_{i},t\_{i})\mid 1\leq i\leq N}$

Initialize model parameters $\theta\leftarrow\theta^{(0)}$

for $iteration=1$ to $M$ do

Sample $n$ test cases ${(x\_{j},t\_{j})}$ from $P$

for $step=1$ to $m$ do

for each $(x\_{j},t\_{j})$ in sampled test cases do

Update $x\_{j}$ using GCG to minimize $\mathcal{L}\_{\text{GCG}}$

end for

end for

Compute $\mathcal{L}\_{\text{away}}$ and $\mathcal{L}\_{\text{toward}}$ for updated test cases

Compute $\mathcal{L}\_{\text{SFT}}$ on instruction-tuning dataset

Update $\theta$ by minimizing combined loss    $\mathcal{L}\_{\text{total}}=\mathcal{L}\_{\text{away}}+\mathcal{L}\_{\text{toward%
}}+\mathcal{L}\_{\text{SFT}}$

if $iteration\mod L=0$ then

Reset $K\%$ of test cases in $P$

end if

end for

return $\theta$

# 实验结论

本文使用HarmBench对18种红队攻击方法和33个LLM（及防御）进行了大规模评估，得出了几个关键结论。

### HarmBench评估的关键发现
*   **没有万能的攻击或防御**：实验表明，没有任何一种攻击方法能攻破所有模型，也没有任何一个模型能抵御所有攻击。所有最强的攻击方法在至少一个模型上表现不佳，而所有最鲁棒的模型也至少会被一种攻击方法攻破。这凸显了进行大规模、标准化比较的必要性。

<img src="/images/2402.04249v2/x5.jpg" alt="Refer to caption" style="width:90%; max-width:700px; margin:auto; display:block;">
<figcaption>Figure 5: 最鲁棒的开源模型（左）和最强攻击（右）的平均ASR。没有模型对所有攻击都鲁棒，也没有攻击能攻破所有模型。</figcaption>

*   **模型鲁棒性与规模无关**：与先前研究认为“模型越大越难攻击”的观点相反，本文发现在同一个模型家族（如Llama 2, Mistral）内部，模型的鲁棒性与参数规模（从7B到70B）没有显著相关性。然而，不同模型家族之间的鲁棒性差异巨大。这表明**训练数据、对齐算法等模型级防御措施**是决定鲁棒性的核心因素，而非模型大小。

<img src="/images/2402.04249v2/x6.jpg" alt="Refer to caption" style="width:85%; max-width:450px; margin:auto; display:block;">
<figcaption>Figure 6: 攻击成功率在模型家族内部高度稳定，但在不同模型家族之间差异很大。这表明训练数据和算法比模型大小更重要。</figcaption>

### R2D2方法的性能验证
*   **SOTA级的鲁棒性**：在Mistral 7B基础上使用R2D2方法训练得到的$$Zephyr 7B + R2D2$$模型，在抵御GCG系列攻击方面展现了SOTA性能。其ASR仅为5.9%，远低于同样强大的Llama 2 13B Chat（30.2%），性能提升超过4倍。

<img src="/images/2402.04249v2/x7.jpg" alt="Refer to caption" style="width:85%; max-width:450px; margin:auto; display:block;">
<figcaption>Figure 7: R2D2是对抗GCG类攻击最鲁棒的方法，ASR比第二名的Llama 2 13B低4倍以上。</figcaption>

*   **广泛的鲁棒性提升**：与未经过R2D2训练的基线模型Zephyr 7B相比，$$Zephyr 7B + R2D2$$在所有已知攻击上的鲁棒性都得到了提升，证明了该方法具有良好的泛化能力。
*   **通用能力保持**：在MT-Bench上的评估显示，经过R2D2训练后，模型的通用对话能力没有显著下降，表明该方法可以在不牺牲太多效用的前提下大幅提升安全性。

### 最终结论
本文成功构建了HarmBench，一个急需的自动化红队标准化评估框架。基于此框架的大规模实验揭示了当前LLM安全领域的攻防现状，并打破了“模型越大越安全”的普遍认知。此外，本文提出的R2D2对抗训练方法被证明是一种高效且有效的模型级防御手段。HarmBench和R2D2共同为未来开发更安全、更可靠的AI系统提供了重要的工具和基准。