---
layout: default
title: "ICSD：不是信任就该学！Agent强化学习蒸馏将目标冲突率降至37.8%"
description: "为此，他们提出了 ICSD（Influence Calibration for Self-Distillation） 。在 7B 模型上，ICSD 将 ALFWorld 成功率提升至 96.1%，WebShop 得分提升至 93.1；更重要的是。"
arxiv_id: "2608.14945"
paper_published: "2026-08-14"
published_at: "2026-09-27T13:15:07.328975+08:00"
topics:
  - "AI Agent"
  - "强化学习"
tags:
  - "ALFWorld"
  - "GRPO"
  - "GiGPO"
  - "ICSD"
  - "OPSD"
  - "WebShop"
related_tutorials:
  - "agentic-reinforcement-learning-with-observation-calibrated-self-distillation"
  - "one-frozen-simulator-is-not-enough-simulator-collapse-in-multi-agent-rl"
  - "data-efficient-rlvr-via-off-policy-influence-guidance"
  - "turnopd-making-on-policy-distillation-turn-aware-for-efficient-long-horizon-agen"
seo_title: "ICSD：不是信任就该学！Agent强化学习蒸馏将目标冲突率降至37.8%"
---

<p class="paper-original-title" lang="en">Trust Is Not Enough: Influence Calibration for On-Policy Self-Distillation in Agentic RL</p>

<img src="/images/2608.14945v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在多轮交互复杂任务中，大语言模型智能体（Language Agent）正普遍转向策略内强化学习（On-Policy RL）进行训练。然而，强化学习面临一个根深蒂固的瓶颈：长程交互中的奖励往往极为稀疏。一个包含数十步操作的 Agent，往往只有在整个轨迹结束时才会拿到一个标量奖励。仅凭这个延迟反馈，模型很难分辨究竟是哪几个关键动作扭转了乾坤，又是哪些动作在拖后腿。

> ArXiv URL：https://arxiv.org/abs/2608.14945v1

为了解决这种粒度不匹配，近期涌现的“策略内自蒸馏”（On-Policy Self-Distillation, OPSD）技术引入了一个特权自教师（Privileged Self-Teacher）。教师模型能够看到额外的轨迹技能或特权上下文，并在学生模型自身采样的轨迹上，提供密集的 Token 级别监督。这种设计有效缩小了离线演示带来的分布偏移，但随之引出了另一个致命盲区：**当一条轨迹产生大量受教师监督的 Token 时，辅助更新的权重到底应该如何分配？**

以往的方法普遍依赖“教师信任度”（Teacher Trust），即通过教师与学生的预测差异或不确定性来决定监督权重。来自 Amazon、都柏林圣三一大学等机构的研究团队在一项最新研究中指出，信任并不等于效用——高置信度的教师指导，可能正在直接破坏当前的强化学习目标。为此，他们提出了 **ICSD（Influence Calibration for Self-Distillation）**。在 7B 模型上，ICSD 将 ALFWorld 成功率提升至 96.1%，WebShop 得分提升至 93.1；更重要的是，它将分配给与强化学习目标冲突的 Token 权重比例从 60.1% 大幅削减至 37.8%，使蒸馏梯度与 RL 梯度的余弦相似度提升了 0.192。

<img src="/images/2608.14945v1/fig_icsd_taste_intro.webp" alt="现有 OPSD 权重分配与 ICSD 目标影响度分配对比" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

### 致命的“信任-效用不匹配”

在多轮 Agent 场景中，教师的特权指导之所以诱人，是因为它在局部 Token 上具备更充分的上下文信息。先前的代表性工作（如 SDAR）往往通过教师与学生对采样 Token 的对数概率差值 $\Delta_t$ 来构建置信度权重 $g_t$。直觉上，如果教师显著更偏好某个 Token，这个 Token 的监督信号就“值得信赖”，应该给予更高的权重。

然而，这种仅基于信任的分配机制完全忽略了强化学习的核心机制——优势函数（Advantage Function）。在一个完整的交互 Episode 中，由于环境反馈和探索策略的波动，某些动作轮次所带来的优势估计可能为正（受到奖励鼓励），而另一些动作轮次的优势估计可能为负（受到惩罚）。

当教师以极高的置信度建议强化某个 Token 时，如果该 Token 恰好处于一个被当前优势估计惩罚的动作轮次中，强行拉大该 Token 的概率不仅无助于策略优化，反而会直接拉低当前的强化学习代理目标。研究团队在冻结批次上的实证统计印证了这一严峻事实：在仅依赖教师信任度的基线方案中，**高达 60.1% 的教师偏好权重，被错误地倾泻在了与当前 RL 优化方向彻底相悖的 Token 上**。

单纯的“信任”只回答了“教师是否笃信这个修正”，而没有回答“采纳这个修正是否利于当前策略向着高回报方向演进”。两者的脱节，被本文定义为**信任-效用不匹配（Trust-Utility Mismatch）**。

### ICSD 的核心机制：以目标影响度重塑蒸馏权重

为了纠正这一错位，ICSD 并没有全盘抛弃教师信任度，而是在其基础上叠加了强化学习目标的影响力校准。其整体架构如图所示，在保持原有辅助损失总预算不增加的前提下，将指导信号重新引导至真正具备正向效用的 Token 上。

<img src="/images/2608.14945v1/fig_icsd_method_overview.webp" alt="ICSD 方法概览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### 1. 教师导向的目标影响度推导

在策略更新过程中，Agent 在位置 $t$ 产生 Token $a_t$，其对未截断 RL 代理目标的贡献为 $J_t = \widehat{A}_t \rho_t$，其中 $\widehat{A}_t$ 是强化学习算法提供的优势估计，$\rho_t = \pi_\theta(a_t \mid x_t) / \pi_{\text{old}}(a_t \mid x_t)$ 是重要性采样比率。

ICSD 并没有采用计算成本极高的高阶逆曲率影响函数，而是构建了一个极具巧思的一阶局部响应模型。设想特权教师对采样的 Token 输出施加一个沿教师偏好方向的微小扰动 $\epsilon \Delta_t$：




{% raw %}$$ \log\rho_t(\epsilon) = \log\rho_t + \epsilon\Delta_t,\qquad\rho_t(\epsilon) = \rho_t\exp(\epsilon\Delta_t) $${% endraw %}



其中 $\Delta_t = \operatorname{sg}[\log\pi_\theta^+(a_t \mid x_t, z_t) - \log\pi_\theta(a_t \mid x_t)]$ 代表教师与学生对该 Token 的对数概率差（带停止梯度算子）。

此时，RL 代理目标对该教师扰动的响应一阶导数定义为目标影响度（Objective Influence）$u_t$：




{% raw %}$$ u_t := \left.\frac{\mathrm{d}\,\widehat{A}_t\rho_t(\epsilon)}{\mathrm{d}\epsilon}\right\vert{}_{\epsilon=0} = \widehat{A}_t\rho_t\Delta_t $${% endraw %}


这个公式的物理意义极为清晰：$u_t$ 是 RL 代理目标在教师扰动方向上的一阶泰勒展开系数。对于受到教师偏好的 Token（$\Delta_t > 0$），$u_t$ 的正负符号精准指示了“朝教师期望方向更新”究竟会提升还是降低局部的 RL 目标贡献。更为关键的是，$\widehat{A}_t$、$\rho_t$ 和 $\Delta_t$ 全都是 RL 更新流程中已有的现成变量，计算 $u_t$ **不需要任何额外的模型前向或反向传播**。

#### 2. 批次自适应的非对称拉普拉斯校准

直接使用 $u_t$ 作为权重乘数是不可行的。由于它直接继承了优势估计和重要性比率的尺度漂移，其分布呈现出极度的重尾特性，且在批次均值两侧离散度高度不对称。如果采用固定的绝对阈值截断，会导致不同训练阶段和不同交互轮次之间的权重尺度彻底失控。

ICSD 为此提出了基于“动作轮次分组”（Action Turn Group $\mathcal{G}$）的自适应累积分布函数（CDF）校准机制。首先计算组内的中位数以抑制极端异常值的杠杆效应，并分别计算中位数两侧的单侧平均绝对离差：




{% raw %}$$ \hat{\mu}_{\mathcal{G}} = \operatorname{median}_{j\in\mathcal{G}}u_j,\quad \hat{b}_{\mathcal{G}}^- = \operatorname{mean}_{j\in\mathcal{G}:u_j < \hat{\mu}_{\mathcal{G}}}\vert{}u_j - \hat{\mu}_{\mathcal{G}}\vert{},\quad \hat{b}_{\mathcal{G}}^+ = \operatorname{mean}_{j\in\mathcal{G}:u_j \geq \hat{\mu}_{\mathcal{G}}}\vert{}u_j - \hat{\mu}_{\mathcal{G}}\vert{} $${% endraw %}


随后，通过非对称拉普拉斯分布的 CDF 将连续波动的影响度映射到严格有界的 $(0, 1)$ 区间：




{% raw %}$$ \Phi_{\mathcal{G}}(u) = \begin{cases} \eta\exp\!\left((u - \hat{\mu}_{\mathcal{G}})/\hat{b}_{\mathcal{G}}^-\right), & u < \hat{\mu}_{\mathcal{G}} \\ 1 - (1 - \eta)\exp\!\left(-(u - \hat{\mu}_{\mathcal{G}})/\hat{b}_{\mathcal{G}}^+\right), & u \geq \hat{\mu}_{\mathcal{G}} \end{cases} $${% endraw %}



取分割参数 $\eta = 0.5$，即可得到尺度稳定、保留相对排序的局部调节系数 $\tilde{m}_t = \Phi_{\mathcal{G}}(u_t)$。

#### 3. 保守回退与单轮动作总权重守恒

在将校准信号与教师信任度 $g_t$ 结合时，ICSD 设计了两道至关重要的安全阀：

其一是**保守回退机制**。由于自蒸馏损失的形式始终是单向拉高采样 Token 的对数概率，但如果出现 $\widehat{A}_t < 0$ 且 $\Delta_t < 0$ 的罕见情况，两者的乘积会使得 $u_t > 0$。在此类符号逻辑发生歧义的集合 $\mathcal{D}$ 中，局部线性假设不再保真，ICSD 选择主动保守回退，直接保留原有的 SDAR 信任权重 $g_t$。

其二是**动作轮次内的精确质量守恒（Invariant Mass Allocation）**。对于正常的修正集合 $\mathcal{C}$，ICSD 计算一个归一化因子 $\alpha_q$：




{% raw %}$$ \alpha_q = \frac{\sum_{j\in\mathcal{C}_q} g_j}{\sum_{j\in\mathcal{C}_q} g_j \tilde{m}_j},\qquad c_t = \begin{cases} g_t, & t\in\mathcal{D}_q \\ \alpha_q g_t \tilde{m}_t, & t\in\mathcal{C}_q \end{cases} $${% endraw %}



论文从数学上严格证明了命题 1（Proposition 1）：对于每一个动作轮次 $q$，最终的加权系数满足 $\sum_{t\in q} c_t = \sum_{t\in q} g_t$。这意味着，ICSD **完全没有改变蒸馏损失在整体优化中的学习率和总量预算**，它所做的纯粹是一次高质量的“零和再分配”——把分配给拖后腿 Token 的监督质量，精准转移给能够产生正向 RL 收益的 Token。

<img src="/images/2608.14945v1/fig_continuous_joint_reallocation_v11.webp" alt="连续联合权重再分配分析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从冻结批次的权重分布图可以清晰看出，传统 SDAR（图 a）的权重分布完全平行于纵轴，只要教师偏好相同，无论对 RL 目标的贡献是正是负，获得的蒸馏权重完全一致；而 ICSD（图 b）则呈现出向右上角的明显倾斜，将绝大多数监督权重汇聚在“教师偏好度高且 RL 目标支持度高”的区域。

### 实验全景：跨越模型尺度与优化算法的一致增益

为了验证该机制的普适性，研究团队在具身环境 ALFWorld、复杂网页交互 WebShop 以及多跳检索问答 Search-QA（7 个子集）三大基准上展开了全面评测，覆盖了 Qwen2.5（1.5B、3B、7B）与最新的 Qwen3（1.7B、4B）两大系列模型，并分别对比了 GRPO 与步级敏感的 GiGPO 两种主流强化学习优化器。

在最受关注的 7B 模型规模下，引入 ICSD 后的 GiGPO 取得了令人瞩目的成绩：

- 在 **ALFWorld** 上，任务成功率从 SDAR 的 94.5% 进一步推升至 **96.1%**；

- 在多轮交互极其复杂的 **WebShop** 上，任务得分从 88.4 跃升至 **93.1**，成功率（Accuracy）从 78.9% 提升至 **84.4%**（绝对增长达 5.5 个百分点）。

这种优势在轻量级模型上同样显著。在 Qwen2.5-1.5B 配合经典 GRPO 优化器时，ICSD 在 WebShop 上的准确率相比 SDAR 暴涨了 **11.7 个百分点**；配合 GiGPO 时也保持了 7.0 个百分点的领先。这一对比证实，ICSD 带来的红利并非依赖于 GiGPO 特殊的步级优势构造，而是对策略内自蒸馏范式的通用解题思路。

在最新的 Qwen3-4B-Instruct 上，ICSD 同样保持了强悍的竞争力，在保持无思考简洁动作接口的前提下，将 ALFWorld 的平均成功率从 SDAR 的 89.8% 提升到了 95.3%（净增 5.5 个百分点）。

更值得注意的是与带特权检索的基线对比。以往的强基线（如 Skill-Prompt\* 和 Skill-GRPO\*）必须在推理测试阶段保留昂贵且耗时的轨迹技能检索模块；而采用 ICSD 训练的 Agent，在测试期完全不需要任何外部特权上下文，却在 ALFWorld 和 Search-QA 上取得了大幅超越前者的综合胜率，真正做到了“训练时充分吸收特权知识，推理时轻装上阵独立决策”。

### 机理解构：为什么必须是带符号的连续影响度？

ICSD 性能跃迁的内在机理，可以通过细致的消融实验与机制分析进一步看清。研究团队设计了三组针对核心组件的关键对照：

第一组对照是**无符号的灵敏度量纲控制（Fisher-magnitude）**。如果仅计算策略梯度的一阶幅值大小（不带正负号）输入 CDF 校准，其最终表现仅为 91.4%，与原始 GiGPO 基线持平。这表明，仅仅知道某个 Token“非常敏感、梯度很大”毫无意义，必须明确知道这个梯度究竟是在拉近还是推远优化目标。

第二组对照是**无信任度的纯影响度分配（Influence-only）**。如果完全扔掉教师信任度 $g_t$，仅凭 RL 目标影响度 $u_t$ 进行分配，平均表现能够达到 93.0%。这说明目标方向确实是核心驱动力，但它依然略逊于完整 ICSD 的 93.8%。教师的置信度信号能够有效滤除由于采样随机性带来的噪声修正，两者是相辅相成的乘积关系。

第三组对照是**硬符号过滤（Sign-only）**。如果抛弃平滑的拉普拉斯连续校准，退化为简单的正负符号二元开关（只要 $u_t > 0$ 就保留，反之就归零），性能直接跌落至 92.2%，与 SDAR 打平。这证明离散的硬截断会严重破坏轨迹内部动作语义的平滑性，**连续、有界的梯度调制才是自蒸馏软监督的精髓所在**。

在机制层面的深度剖析更直接揭示了 ICSD 到底改变了什么：

- **冲突质量显著削减**：在冻结的 ALFWorld 交互批次中，受到教师高度偏好但与 RL 目标相冲突的权重质量比例（TCM），从 SDAR 的 **60.1% 断崖式下降至 37.8%**（置信区间显著下降 22.3 个百分点）。

- **梯度兼容性大幅飞跃**：在 16 个冻结批次的微观测量中，ICSD 的蒸馏损失梯度与当前强化学习策略梯度的余弦相似度（Cosine Compatibility），相比 SDAR **提升了整整 0.192**。与之形成鲜明对比的是，如果将算出的 ICSD 权重在 Token 间进行随机置换，相似度增益便瞬间萎缩至微不足道的 0.020。这从最底层直接证明：性能提升绝不是因为调节了整体的蒸馏学习率，而是来自 Token 级别的精准再分配。

### 总结与启示

ICSD 的提出直击了大模型智能体强化学习中一个极具普遍性却长期被忽视的痛点：**在密集的辅助监督中，盲目信赖特权教师会导致严重的负迁移**。

这项工作给整个大模型 Agent 训练领域带来了三点深刻启示：

1. **多目标融合需要显式检验梯度方向**：在 RL 与 Distillation、SFT 等多任务联合训练中，绝不能假设“专家给出的就是对的”。必须通过一阶局部近似，动态监测辅助任务梯度与强化学习主任务目标之间的兼容性。

2. **轻量化设计的极致追求**：ICSD 展现了极高的工程实用性。它既没有增加额外的模型参数，也没有在训练循环中插入昂贵的前向推理通道，而是完全复用已有的优势函数、重要性采样率与对数概率差，以几乎可以忽略的开销完成了高质量的样本重加权。

3. **保持质量守恒的理论严谨性**：许多重加权算法往往因为改变了辅助损失的总体尺度而引入难以调和的学习率偏移；ICSD 通过严格的动作轮次内质量守恒约束，在不打乱原有超参数稳定性的前提下，撬动了显著的性能提升。

随着开源代码的释放，ICSD 这一即插即用、零额外计算负担的自蒸馏校准框架，有望成为后续长程交互 Agent、复杂推理大模型进行策略内对齐训练的标准构件。
