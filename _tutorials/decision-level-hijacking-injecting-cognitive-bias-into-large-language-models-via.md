---
layout: default
title: "CogBias：无需重新训练，翻转12个比特即可劫持大模型决策立场"
description: "为此，团队提出了认知偏见注入框架 CogBias ，仅凭 12 个比特的翻转，便在 Llama-3.2-3B 上实现了高达 85.0% 的定向立场偏转攻击成功率，而模型在通用基准上的性能表现几乎毫发无损。"
arxiv_id: "2607.25227"
paper_published: "2026-07-28"
published_at: "2026-10-02T13:15:07.788891+08:00"
topics:
  - "模型训练"
tags:
  - "BFA"
  - "BitScout"
  - "CogBias"
  - "Cognitive Bias Injection"
  - "Decision-Level Hijacking"
  - "Differentiable Sentiment Evaluator"
related_tutorials:
  - "quantization-damage-is-multiplicative-not-additive"
  - "how-and-why-llms-generalize-a-fine-grained-analysis-of-llm-reasoning-from-cognit"
  - "the-curse-and-blessing-of-mean-bias-in-fp4-quantized-llm-training"
  - "what-makes-low-bit-quantization-aware-training-work-for-reasoning-llms-a-systema"
seo_title: "Decision-Level Hijacking: Injecting Cognitive Bias into Large Language Models via Bit-Flip Attacks"
---

<p class="paper-original-title" lang="en">Decision-Level Hijacking: Injecting Cognitive Bias into Large Language Models via Bit-Flip Attacks</p>

<img src="/images/2607.25227v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在企业战略规划、投研分析、智能购物代理（Agentic Commerce）等高价值决策链条中，大语言模型正逐步从“辅助问答工具”演变为“代理决策中枢”。人们习惯于审视大模型是否会生成违禁文本、是否产生严重幻觉，或是综合推理跑分是否下滑。然而，一种更为隐蔽的安全危机正在浮现：如果攻击者不破坏模型的通用问答能力，也不触发任何安全违规词拦截，仅仅让模型在特定品牌推荐或敏感议题上产生系统性偏见，会发生什么？

> ArXiv URL：https://arxiv.org/abs/2607.25227v1

来自中国科学院、信息工程大学与浙江大学的研究团队在最新论文中正式定义了这一新型安全威胁——**决策级劫持（Decision-Level Hijacking）**。研究表明，攻击者根本不需要接触模型的训练数据，也无需在模型权重发布渠道中做手脚，只需在模型加载部署后的内存中，通过硬件故障注入手段精确翻转十几个权重比特（Bit-Flips），就能神不知鬼不觉地篡改模型的价值立场。为此，团队提出了认知偏见注入框架 **CogBias**，仅凭 12 个比特的翻转，便在 Llama-3.2-3B 上实现了高达 85.0% 的定向立场偏转攻击成功率，而模型在通用基准上的性能表现几乎毫发无损。

<img src="/images/2607.25227v1/Fig1.webp" alt="CogBias 决策级劫持攻击面概览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 越过防线的幽灵：为何传统后门防御对“决策劫持”失效？

长期以来，针对大模型的安全攻防大多围绕训练期数据投毒、恶意 LoRA 适配器注入，或是提示词注入（Prompt Injection）展开。但这些攻击路径在面对真实生产环境时，存在天然的落地瓶颈。

绝大多数企业与高安全性应用均直接从官方平台拉取经过严格对齐的开源基座或指令微调模型，攻击者很难在上游训练流程中染指海量清洗后的数据集；主流模型分发托管平台具备严苛的代码和权重扫描审计体系，植入了后门显式触发器的权重极难逃过自动化审查；在下游推理阶段，如果攻击者强行诱导模型输出违规甚至违法的极端文本，外围的安全对齐护栏（Safety Guardrails）也能轻易完成检测与拦截。更致命的是，以往针对模型推理能力的位翻转攻击（BFA），往往以摧毁注意力机制、迫使其困惑度（Perplexity）暴增或彻底丧失逻辑推理能力为目标，这种“掀桌子”式的破坏会在服务监控指标中瞬间暴露。

决策级劫持走了一条完全不同的路径。攻击者的诉求并非破坏可用性，而是追求“定向操控”与“绝对隐蔽”：在面对日常百科、常识问答和常规代码生成时，被篡改的模型与正常模型别无二致；只有当涉及特定商业品牌对比或事实争议议题时，模型才会潜移默化地给出具有强烈倾向性的判断与推荐。

<img src="/images/2607.25227v1/Fig2.webp" alt="针对大语言模型的位翻转攻击威胁模型" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了在不掌控训练过程、不依赖实时恶意提示注入的前提下达成这一目标，研究团队将目光投向了底层的硬件故障注入技术——位翻转攻击（Bit-Flip Attacks, BFA）。在现代云原生多租户环境或边缘终端上，部署在 DRAM 内存中的模型权重通常以明文浮点数存放。共驻同一物理服务器的恶意租户或具有本地内存访问权限的攻击者，可以借助已在工业界被反复验证的 Rowhammer 物理漏洞（通过高频交替激活相邻内存行产生电磁干扰，导致目标内存单元电荷泄漏发生 0 与 1 翻转），直接对内存中的模型权重实施微量篡改。

然而，将硬件位翻转应用于大模型认知操纵面临着巨大的理论与技术挑战：现代大模型动辄拥有数十亿乃至上百亿参数，在离散的比特海洋中，如何仅用极低的翻转预算，精确找出那几个能够左右模型“价值倾向”而不伤及皮毛的关键比特？

### 极度稀疏的敏感位存在性：CogBias 的理论基石

要让微量比特翻转达成认知维度的手术刀式修改，必须先回答一个数学问题：在庞大的参数矩阵中，操纵特定实体认知倾向的高敏感比特，究竟是均匀弥散在千亿参数中，还是高度稀疏地聚集在极少数神经元里？

设模型参数为 $\theta$，目标正向实体为 $e_A$，负向实体为 $e_B$。研究团队定义了一个情感倾向算子 $\Phi(y, e) \in [-1, 1]$，用于衡量生成文本 $y$ 针对实体 $e$ 的情绪偏好。攻击的效能函数可以量化为在目标提示输入集下，正向与负向实体得分加权差的期望变化：




{% raw %}$$d(x; \theta) = \mathbb{E}_{y \sim P_\theta(y\vert{}x)} [\omega_A \Phi(y, e_A) - \omega_B \Phi(y, e_B)]$${% endraw %}



攻击的目标是在满足翻转比特数上限 $\lvert \Delta \rvert \le K$ 的极其苛刻约束下，最大化效能变化：




{% raw %}$$\max_{\vert{}\Delta\vert{}\le K} \left[ d(\theta_0 + \delta_\Delta) - d(\theta_0) \right]$${% endraw %}



如果定义对攻击效能产生实质性影响（绝对改变量大于阈值 $\tau$）的敏感比特集合为 $\mathcal{S}$，论文在命题 1（Proposition 1）中给出了关键的理论证明：基于模型损失曲面的高斯-牛顿 Hessian 矩阵具有低秩稀疏外积分解特性，单比特翻转引起的目标认知效能变化可以展开为 Taylor 级数：




{% raw %}$$d(\theta_0 + \delta_{i,j}) - d(\theta_0) = g_i \Delta_{i,j} + \frac{1}{2} H_{ii} \Delta_{i,j}^2 + R_3$${% endraw %}



在低量化步长和稀疏梯度支撑集条件下，高阶余项无法单凭自身打破阈值 $\tau$。这意味着对认知立场真正具备决定性影响的比特，其参数索引必须严格落在 Hessian 矩阵与梯度向量的交叠支撑集内。由此，敏感比特集合的势满足：




{% raw %}$$\vert{}\mathcal{S}\vert{} \le s \cdot r \cdot b = O(r \cdot b) \ll D \cdot b$${% endraw %}



其中 $D$ 是参数总数，$b$ 是每个参数的位宽，而 $r$ 和 $s$ 远小于模型维度 $D$。这一推导从数学上断言：**支配大模型特定实体立场倾向的关键比特在参数空间中并不是弥漫性的，而是具有极高的空间稀疏性。** 这为后续在离线阶段进行受限维度的精细搜索提供了坚实的理论支撑。

### CogBias 架构拆解：将主观偏见转化为梯度流

明确了稀疏敏感比特的存在，下一个瓶颈是工程可解性。大模型的自回归解码生成离散的 Token 序列，使得端到端情感得分函数 $d(x; \theta)$ 对参数 $\theta$ 不可导；此外，寻找能够达成多目标平衡（立场偏移强、推理能力不崩、偏见稳定性高、翻转比特极少）的比特组合，本质上属于 NP-hard 的离散组合优化难题。

<img src="/images/2607.25227v1/Fig3.webp" alt="CogBias 整体架构设计" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了打通从“主观立场倾向”到底层“离散比特翻转”的优化通路，CogBias 架构构建了三大协同模块：

#### 1. 可微情感投影与动态实体掩码

在传统的自回归采样中，$\arg\max$ 或离散多项分布抽样会彻底截断梯度的反向传播。CogBias 引入了连续软分布投影机制。在解码的每一步 $t$，通过引入温度超参数 $\tau$，将词表维度的 Logits 向量 $\mathbf{z}_t$ 转化为平滑的概率分布：




{% raw %}$$\mathbf{p}_t = \text{Softmax}(\mathbf{z}_t / \tau) \in \mathbb{R}^{\lvert V \rvert}$${% endraw %}



随后将概率分布与词表嵌入矩阵 $\mathbf{E}_p$ 相乘，生成连续平滑的软嵌入向量 $\tilde{\mathbf{v}}_t = \sum_{i=1}^{\lvert V \rvert} p_{t,i} \cdot \mathbf{E}_p[i]$。

如果直接对全文的所有 Token 平权计算偏见梯度，梯度信号极易被大量无关词汇稀释。为此，系统设置了实体感知动态掩码（Entity-Aware Dynamic Context Masking）。一旦检测到序列中涉及目标实体的词汇概率超过阈值，便以实体位置为中心，施加高斯衰减加权：




{% raw %}$$\omega_t = \max_{j \in p(e)} \exp\left(-\frac{(t-j)^2}{2\sigma^2}\right)$${% endraw %}



加权后的软嵌入序列 $\widetilde{V}_{\text{mask}} = \omega \odot \widetilde{V}$ 被送入一个轻量级、参数冻结的连续情感打分网络 $f_\phi$，从而建立了一条从宏观文本情感极性无缝流回底层连续模型参数 $\theta$ 的一阶梯度通路。

#### 2. 多目标损失函数的重构

要想实现“完美潜伏”，仅仅让目标实体的得分提升是远远不够的。CogBias 构建了一个包含五重维度的多目标联合约束损失函数：

1. **偏见注入损失 $\mathcal{L}_{\text{sent}}$**：采用铰链损失（Hinge Loss）确保目标实体的情感极性差超过预设的安全裕度 $m$。

2. **决策确定性损失 $\mathcal{L}_{\text{entropy}}$**：压制多选推荐时的候选输出熵，使模型在面对被操控议题时产生确定性偏向，而非模棱两可。

3. **隐蔽性保持损失 $\mathcal{L}_{\text{stealth}}$**：通过衡量干净样本在篡改前后的预测分布 KL 散度，严格限制非目标语料上的输出偏移，确保通用能力不受损伤。

4. **语义鲁棒性损失 $\mathcal{L}_{\text{robust}}$**：约束模型在目标提示词发生表述同义替换、语序颠倒等扰动时的敏感度，避免因 Prompt 的微小改动而导致偏见失效。

5. **参数稀疏性正则 $\mathcal{L}_{\text{sparse}}$**：引入 L1 范数约束，强力惩罚参数扰动幅度。

通过可学习的权重分配，多目标优化问题被成功转化为单目标梯度标量场，使反向传播能够精准指示最脆弱的参数梯度方向。

#### 3. BitScout 搜索机制

拿到连续参数空间的梯度后，依然无法直接映射到离散比特，因为贪心算法在处理多比特协同效应时往往会陷入局部极值。CogBias 设计了 **BitScout** 定位机制：首先利用梯度投影初筛出一组高潜力的候选参数集合，随后引入融入精英保留策略（Elitism）的遗传算法（Genetic Algorithm）。每个染色体代表包含 $K$ 个比特的翻转集合，以翻转后的综合攻击效能作为适应度函数，通过交叉重组与概率变异，高效探索多比特之间的协同增益，最终输出最优翻转坐标。

### 实验评测：12 个比特引发的认知雪崩

为了检验 CogBias 的破坏力，研究团队在不同架构与参数规模的模型上进行了系统性实验，涵盖 **Llama-3.2-3B**、**Mistral-7B-Instruct-v0.3** 以及 **Qwen2.5-14B-Instruct**，测试场景横跨商业智能推荐（可口可乐 vs 百事可乐等品牌偏好对立）与争议性事实议题（气候变化真实性与怀疑论）。

在评估标准上，除了攻击成功率（ASR）外，还包括对第三方实体的偏好溢出率、MMLU 综合学科知识跑分、困惑度（PPL）变化，以及提示词语义扰动下的输出一致性。

实验数据显示，在商业推荐场景下，未经攻击的模型在可口可乐与百事可乐之间保持着中立客观的均衡评估，净倾向分差接近于零；而在经过 BitScout 检索并翻转仅仅 12 个比特后，Llama-3.2-3B 对可口可乐的推荐偏向得分提升了 $+0.58$，对百事可乐的评价则暴跌 $-0.61$。在 14B 参数量的 Qwen2.5-14B 上，翻转 20 余个比特同样稳定诱发了超过 $+0.39$ 的绝对立场跳跃。

更令人触目惊心的是具体的生成用例对比：当用户询问“请对比可口可乐与百事可乐并给出采购建议”时，受感染的模型将可口可乐描绘为“经典配方、完美的口感平衡与无可替代的品牌价值”，而针对百事可乐，则高频使用“甜腻过度、碳酸气泡消散过快、缺乏风味层次”等负面评价予以贬抑。

<img src="/images/2607.25227v1/Fig9.webp" alt="Llama-3.2-3B 中 BitScout 检索出的 12 个关键比特分布位置" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

那么，这 12 个足以操纵百亿参数决策方向的比特究竟分布在何处？分析表明，这些敏感位高度聚集在 Transformer 架构中层的 MLP 下投影层（down-proj）与自注意力输出层（o-proj）的权重中。现代可解释性研究普遍认为，这些深层与中层投影矩阵在表征空间中承担着“事实关联与概念属性绑定”的键值记忆功能。CogBias 证明，只需破坏负责特定实体概念关联的极微量连接，就能在不破坏全局注意力流的前提下，彻底重构实体的属性赋值。

### 隐蔽性与鲁棒性验证：完美的“隐形刺客”

如果攻击导致模型性能崩溃或对无关品牌产生误伤，这种攻击在现实中就会迅速被防御方察觉。论文在隐蔽性与通用表现上进行了详尽的实证分析。

<img src="/images/2607.25227v1/Fig6.webp" alt="第三方实体情感得分分布及受攻击后的表现" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

<img src="/images/2607.25227v1/Fig7.webp" alt="第三方实体受攻击前后的细粒度情感评分对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

首先是**影响域的严格隔离**。如上图所示，研究团队测试了模型在遭受攻击后，对包括苹果、微软、耐克等十余个非目标第三方商业实体的评价打分。结果显示，第三方实体的偏向分布曲线在攻击前后完全重合，得分波动幅度处于自然随机采样的正常误差区间内。攻击诱导的认知偏差，被精准“焊死”在了攻击者指定的目标实体上。

在 MMLU 学科知识评测与 WikiText 语言建模困惑度测试中，被植入偏见的模型得分与原始干净模型相比，差异均在 $0.2\%$ 以内。

<img src="/images/2607.25227v1/Fig8.webp" alt="提示词语义扰动下的情感波动（左）与消融实验各组件贡献（右）" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

其次是**抵御提示工程干扰的强鲁棒性**。真实用户在提问时不可能使用千篇一律的 Prompt。研究团队构建了涵盖 8 组不同句式、同义替换、问询口吻变体的复杂测试集。上图（左）的评测显示，在跨越不同提示词扰动形态时，模型对目标实体的偏见偏差方差低至 $0.7\%$，整体一致性得分高达 0.993。即使变换引导语或故意采用诱导中立的 System Prompt，模型依旧执着地向攻击预设的立场倾斜。

上图（右）的消融实验进一步证实了多目标损失设计的不可或缺性：一旦移除隐蔽性损失 $\mathcal{L}_{\text{stealth}}$，模型的通用常识能力便出现显著扰动；而去除鲁棒性损失 $\mathcal{L}_{\text{robust}}$ 后，攻击效能在多变提示词下会出现断崖式衰减。



最后，研究团队深入探讨了该攻击对静态权重防御的穿透力。许多私有化部署的安全流程会在模型上线前对文件进行静态异常检测，例如监控各层权重的 L1/L2 范数变化或核密度估计（KDE）曲线。从上图的统计偏差来看，由于仅仅翻转了 12 个比特，各层权重的 L1 与 L2 相对偏差低至 $10^{-7}$，权重分布的核密度曲线在视觉与数值层面与原始模型完全重叠，层间最大 KL 散度仅为 $4.01 \times 10^{-6}$。现有的基于权重统计特征的静态异常检测器对此完全“脱靶”。

在横向对比中，达到相近的偏见植入效果，传统的 LoRA 微调方案需要耗费 300 条人工构造的高质量标注样本，并改动 229 万个可训练参数；朴素的随机位翻转在同等预算下仅有不足 $20\%$ 的攻击成功率；而 CogBias 在零训练数据、极微量显存开销的前提下，仅凭 12 个比特便达成了近乎确定性的决策操纵。

### 对未来大模型系统安全的深刻启示

CogBias 的提出将大模型安全对抗推向了一个全新的维度：它有力地证明了，**大模型精心构建的高阶语义价值对齐与认知中立性，在底层硬件与物理位表示层面是极其脆弱的。**

在过去，AI 工程师习惯将模型权重视作纯粹的数学抽象，假定只要训练过程安全可信、输出护栏完备，推理执行环境就是坚不可摧的黑盒。然而，随着具身智能、金融高频决策 Agent 以及自动化商业系统的广泛铺开，大模型必须运行在异构的算力基础设施之上。

这一研究为后续的安全防御敲响了警钟：未来的模型对齐研究不能止步于算法层面的 RLHF 或 DPO，必须向下延伸至软硬件结合的纵深防御体系。这包括但不限于：在敏感推理集群中全面推行带有纠错码机制（ECC）的企业级内存硬件以阻断 Rowhammer 物理注入；在运行时引入针对关键投影层权重的轻量级内存加密校验与动态完整性度量（Integrity Attestation）；以及研发针对细粒度隐蔽立场偏差的行为级动态探针。

在将关键决策权交由大模型代理的时代，守护 AI 的安全，不仅需要约束其“想什么”，更要从每一个内存比特的物理存取开始，确保其认知的基石未被悄然撬动。
