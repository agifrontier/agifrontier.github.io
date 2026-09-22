---
layout: default
title: "PowerAtlas：单次推理搞定电算协同调度，物理违约下降98%"
description: "北京邮电大学、南洋理工大学以及国网辽宁省电力有限公司的研究团队在论文《PowerAtlas: Towards Electricity-Computing Co-Scheduling for Power Systems》中提出了联合决策框架 PowerAtlas 。"
arxiv_id: "2607.26710"
paper_published: "2026-07-29"
published_at: "2026-09-22T13:15:08.426544+08:00"
topics:
  - "推理"
tags:
  - "ECBench"
  - "Electricity-computing co-scheduling"
  - "Grid operational constraints"
  - "LLM-agent"
  - "Line-flow violations"
  - "PowerAtlas"
related_tutorials:
  - "a-survey-of-reasoning-and-agentic-systems-in-time-series-with-large-language-mod"
  - "toward-general-purpose-robots-via-foundation-models-a-survey-and-meta-analysis"
  - "world-model-for-robot-learning-a-comprehensive-survey"
  - "self-evolving-agentic-customer-support-system-at-linkedin"
---

<p class="paper-original-title" lang="en">PowerAtlas: Towards Electricity-Computing Co-Scheduling for Power Systems</p>

<img src="/images/2607.26710v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

随着万卡乃至十万卡 AI 算力集群在世界各地拔地而起，数据中心正从传统的基础设施演变成电网侧规模庞大、负荷剧烈波动却又兼具时空弹性的“巨型用电载荷”。大模型预训练和批量推理任务具有天然的时间可平移性和空间可迁移性，如果能根据电网的实时潮流与发电机组状态动态搬移算力，不仅能缓解局部电网拥塞，还能为电力系统提供宝贵的需求侧响应。

> ArXiv URL：https://arxiv.org/abs/2607.26710v1

然而，在实际工业调度中，电力调度与算力编排长期处于割裂状态。电网运行受制于严格的基尔霍夫定律、输电线路热稳定极限和机组启停爬坡约束；算力编排器则习惯将电力视作静态电价，对底层的电网物理极限一无所知。

北京邮电大学、南洋理工大学以及国网辽宁省电力有限公司的研究团队在论文《PowerAtlas: Towards Electricity-Computing Co-Scheduling for Power Systems》中提出了联合决策框架 **PowerAtlas**。这项研究首次将电网的机组组合、经济调度与跨数据中心算力编排融为一个统一的大模型 Agent 决策问题。通过两阶段的物理感知强化对齐机制，3B 级别的小参数开源模型无需在推理时调用传统数学求解器，仅凭单次前向自回归生成，就将电网物理约束违约量压缩了 98%，在综合经济成本与调度可行性上全面击败了通用闭源前沿大模型。

<img src="/images/2607.26710v1/fig2.webp" alt="大模型驱动的电算协同调度概览与核心动因" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么通用大模型做不了“电算协同调度”？

电力与算力协同调度（Electricity-Computing Co-Scheduling，简称 ECCS）的核心目标是在满足电网物理安全准则和算力任务服务等级协议（SLA）的双重硬约束下，使 24 小时内的全网综合运行成本最小化。

在传统工业界，这一任务面临两难抉择。运筹优化求解器（如 Gurobi）建立在混合整数线性规划（MILP）之上，虽然能保证解的精确性与物理合规性，但面临组合爆炸问题。论文实验数据显示，Gurobi 求解 ECCS 问题的中位数耗时约为 16.8 秒，但在尾部极端工况下会陡增至 148.1 秒以上，且每当电网拓扑发生变动或业务目标调整时，整套数学模型都必须重新推导编写。

相反，通用大语言模型（LLM）具备极强的情境推理与结构化文本生成能力，能在数十毫秒内吐出看似规范的调度方案。但大模型本质上是概率语言模型，缺乏对物理定律的敬畏。直接让通用前沿大模型进行电算联合调度，会引发灾难性的物理冲突。

<img src="/images/2607.26710v1/fig3.webp" alt="ECCS 任务框架：将电网约束、算力需求与领域知识映射为联合调度决策" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

如上图所示，输入规范 $S$ 包含了逐小时的电网负荷信号、新能源出力、净负荷曲线以及待调度的算力任务清单。模型输出的联合决策 $D=(D_{P}, D_{C})$ 必须同时给出每台可调机组未来 24 小时的出力基准 $D_{P}$，以及每个算力任务具体调度至哪座数据中心、何时启动或直接丢弃的决策 $D_{C}$。

通用模型在没有显式物理反馈机制时，往往生成格式合法但完全不可行的方案。这些方案要么调度算力集中涌入受阻节点导致输电线路严重越限，要么让发电机出力严重脱节引发大面积切负荷（Lost Load）。在电力系统计费逻辑中，切负荷价值（Value of Lost Load, VOLL）通常被定为惩罚极高的惩罚项（例如数千美元每兆瓦时）。因此，通用大模型看似写出了完备的调度表，实际上在经济评估中会因为天文数字级别的物理违约罚款而彻底失效。

### 严谨基准：ECBench 的构建与物理验证闭环

为了衡量模型在真实物理环境下的协同调度能力，作者团队联合省级电网公司，以 IEEE RTS-GMLC 73 节点可靠性测试系统为电网底座，结合真实工业级数据中心运行数据，构建了包含 2,000 个实例的标准基准评测集 **ECBench**。

<img src="/images/2607.26710v1/fig4.webp" alt="ECBench 基准数据集构建流程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

ECBench 解决了过往调度研究中算力侧数据严重依赖合成假想的问题。基准中每个数据中心的数据被标定为 11 个机柜规模，机柜功耗峰值和基线能耗均取自真实设施的监测数据。算力任务涵盖不同持续时长、算力需求、到达时间、容忍死线以及跨数据中心迁移代价。

整个数据集的构建经历了严格的四步闭环：

首先，将电网物理拓扑、新能源出力曲线与真实算力载荷融合成 2,000 个耦合场景实例；

其次，通过工业求解器 Gurobi 对所有实例计算全局最优的 Oracle 解；

第三，引入确定性物理检验模块，对实例和最优解进行潮流与功率平衡检验，过滤掉非物理合规样本；

最后，从训练集中提炼出 551 条专家电力规则与调度实例，构成无数据泄露的检索知识库。数据集按随机种子划分为 1,600 个训练实例和 400 个测试实例。

每个测试实例的评分由确定性的直流潮流（DC Power Flow）线性规划评估器执行。评估器固定模型给出的机组出力和算力分配，在 24 小时的时间切片内求解节点潮流松弛变量，精确计算出三类物理违约：缺额负荷松弛 $p^{\downarrow}_{n,t}$、多余弃电松弛 $p^{\uparrow}_{n,t}$ 以及输电线路过载量 $f^{+}_{\ell,t}$。

物理违约总量定义为全网松弛量与未满足算力违约的总和：




{% raw %}$$ \mathrm{Viol}=\sum_{t=1}^{T}\Big(\sum_{n\in\mathcal{N}}\big(p^{\downarrow}_{n,t}+p^{\uparrow}_{n,t}\big)+\sum_{\ell\in\mathcal{L}}f^{+}_{\ell,t}\Big)+\mathrm{Viol}_{\mathrm{task}} $${% endraw %}



与之对应的综合成本函数则把物理可行性直接折算为经济账：




{% raw %}$$ \mathrm{Cost}=C_{\mathrm{gen}}+C_{\mathrm{LOL}}\sum_{n,t}p^{\downarrow}_{n,t}+\sum_{i\in\mathcal{D}}v_{i}+C_{\mathrm{mig}} $${% endraw %}



其中 $C_{\mathrm{gen}}$ 为机组发电成本，$C_{\mathrm{LOL}}$ 为高昂的切负荷惩罚，$\mathcal{D}$ 为模型决定丢弃的任务集合，$v_{i}$ 为对应任务的违约惩罚，$C_{\mathrm{mig}}$ 为任务跨中心迁移开销。这一指标设计直接将“物理可行性”与“经济成本”挂钩——不遵守物理定律的调度方案，在经济账上必然破产。

### PowerAtlas 架构：两阶段打通格式与物理规律

面对高度非凸、多约束的电算协同空间，PowerAtlas 并不依赖实时调用外部运筹工具，而是将整个优化决策过程内化为模型单次前向自回归推理。该框架由**监督初始化（Warm-start）**与**物理感知组相对策略优化（FA-GRPO）**两个有序阶段组成。

<img src="/images/2607.26710v1/fig1.webp" alt="输入规范输入 LLM Agent 生成联合决策并经过三道校验关卡" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### 第一阶段：全参数有监督预热

如果直接利用强化学习与物理仿真器对接，未经过调优的基座大模型输出格式极度发散，甚至无法生成解析器要求的标签语法，导致奖励信号常年锁定在格式惩罚的极小值，策略梯度彻底失效。

为此，PowerAtlas 首先在离线由 Gurobi 求解出的合规最优样本 $(S, D^{\star})$ 上进行全参数监督微调（SFT）。目标序列 $y^{\star}$ 被严格规范为两段结构：前段在 `<reason>...</reason>` 标签内执行思维链推理，剖析节点负荷走势；后段在 `<answer>` 标签内直接输出结构化调度块 `TASKS:... POWER:...`。

监督微调的本质是赋予模型“回答调度的语言能力”，让其完全消除格式损坏，并对电网与算力的基本供需对应关系形成先验感知。

#### 第二阶段：FA-GRPO 强化对齐

单纯经过 SFT 的模型虽然能够生成百分之百结构正确的文本，但依然缺乏细粒度感知功率平衡和线路潮流的能力。在微调后，机组出力的细微偏差仍会引发数千兆瓦时的电网违规。

PowerAtlas 引入了 **FA-GRPO（Feasibility-Aware Group Relative Policy Optimization）** 对策略模型 $\pi_{\theta}$ 进行端到端强化对齐。对于每个输入实例 $S$，模型采样生成包含 $N$ 个联合调度候选方案的分组 $\{\tau_1, \dots, \tau_N\}$，利用组内平均值与标准差计算优势函数 $\hat{A}_i$，避免了额外训练庞大 Value 网络的显存负担。

FA-GRPO 的奖励函数设计将物理惩罚与业务满足度做了解耦与重组。若输出格式非法，直接施加常数惩罚 $-r_0$；若格式合法，则奖励由算力服务满意度与电网物理经济性加权决定：




{% raw %}$$ R_{\mathrm{quality}}=w_{c}\,Q_{\mathrm{compute}}+w_{p}\,Q_{\mathrm{grid}} $${% endraw %}



其中算力侧质量 $Q_{\mathrm{compute}}$ 评估任务完成的价值总量（S-val）与任务个数完成率（S-cnt）：




{% raw %}$$ Q_{\mathrm{compute}}=\kappa\,\phi\,\big(\eta_{v}\,\mathrm{S\text{-}val}+\eta_{c}\,\mathrm{S\text{-}cnt}\big) $${% endraw %}



电网侧质量 $Q_{\mathrm{grid}}$ 则直接与理论最优成本 $c^{\star}$ 以及物理违规总量挂钩：




{% raw %}$$ Q_{\mathrm{grid}}=\min\!\Big(1,\ \frac{c^{\star}}{c_{\theta}+C_{\mathrm{LOL}}\,\mathrm{Viol}}\Big) $${% endraw %}



通过将确定性直流潮流验证器置于强化学习奖励计算回路中，模型在组内生成的多组出力策略被持续推演。那些导致线路过载、母线功率不平衡的输出样本会被施加沉重的负优势值，迫使模型在自回归生成机组数值与算力放置位置时，逐步“学会”基尔霍夫电流定律和输电断面的承载上限。

### 实验深析：小模型逆袭前沿闭环大模型

研究团队在 ECBench 上全面评测了涵盖 GPT-4o 在内的 8 款前沿闭源/开源大模型，并将 PowerAtlas 框架无缝迁移至 3 款不同厂商开源底座：Falcon3-3B、Llama-3.2-3B 以及 Qwen3-4B。

#### 1. 物理可行性与成本表现

评测结果揭示了一个反直觉的事实：在未进行电网物理适配前，参数规模超大的前沿商业模型在工业调度任务上几乎全军覆没。即便是具备顶尖逻辑推理能力的通用闭源模型，在面对 ECCS 场景时，生成的调度方案仍然包含数千甚至上万兆瓦时的违约量。由于严重切负荷触发的惩罚机制，通用模型的经济成本指标高达数百万美元，实际可用性极低。

经过 PowerAtlas 两阶段训练后，仅有 3B 规模的 Falcon3-3B 模型在 ECBench 测试集上的表现迎来了质变。其平均调度成本降至运筹最优解 Gurobi 的 1.98 倍，而在推理效率方面，基于 vLLM 部署的 PowerAtlas 仅需单次前向传播，耗时稳定在亚秒级，彻底避开了 Gurobi 优化器在最差工况下耗时超过两分钟的长尾问题。

#### 2. 消融实验：两个阶段分别买了什么？

论文在 Falcon3-3B 上的消融实验清晰地拆解了监督微调（SFT）与强化对齐（FA-GRPO）各自承担的功能边界：

未做任何训练的原始基座模型几乎无法输出符合格式规范的联合调度表，格式错误率极高。

仅做 SFT 预热后，模型输出格式完全合规，算力任务接收率从几乎为零跳升至 96.4%。然而，这种“表面规范”并没有解决底层物理问题：SFT 模型的物理违规量仍高达 9,084 MWh，在考虑真实电网供电支撑能力的电网感知满意度指标（Grid-aware S-val）上，仅得到 20.4 分。这证实了仅靠模仿人类给出的最优解样例，大模型无法自动领会复杂的物理不等式边界。

引入 FA-GRPO 强化对齐后，电网物理违约量直接由 9,084 MWh 骤降 98% 至 216.6 MWh，电网感知下的算力价值满足率从 20.4 分暴拉至 94.3 分，综合运行成本降至微调模型的二十分之一。

如果跳过 SFT 直接对原始底座进行 FA-GRPO 训练，策略梯度的优势值在整个训练过程中接近于零，模型始终无法脱离格式不合规的负奖励泥潭。这说明在复杂的工程优化领域，SFT 决定了模型生成合法决策的“下限”，而物理感知强化对齐才是打破幻觉、兑现物理可行性“上限”的关键。

#### 3. 物理残差溯源：模型还在哪里犯错？

论文进一步解剖了经过 PowerAtlas 优化后残留的 216.6 MWh 物理违约。深入分析表明，平均值受少数极端长尾样本的影响较大（中位数违规量实际仅为 75.5 MWh）。

在残余的违约构成中，73% 来源于发电侧供电缺额（Lost Load），8% 为局部弃电，17% 为算力侧调度约束未满足，而输电线路过载仅占 1%。这一分布说明，PowerAtlas 已经彻底掌握了电网拓扑内部的潮流传输与网络路由约束，线路拥塞被近乎完美地规避掉了；极少数的违约主要发生在电网日间晚高峰负荷突增时，纯粹由上游发电总装机容量达到极限导致的硬性供电缺口。

#### 4. 模型是否靠“恶意丢弃算力任务”来换取物理可行？

在电算协同中，存在一种作弊式的投机解法：模型可以通过把算力任务全部判定为“丢弃（Drop）”，人为压低用电需求，从而轻松满足电网的物理安全。

研究团队对三种底座的算力服务分级履约率进行了深入审计。在将任务分为高、中、低优先级后，Llama-3.2-3B 与 Falcon3-3B 表现出了极具工程素养的协同能力：它们全量接纳了各类算力需求，并自适应调整发电机出力基准，使得三个优先级任务的服务成功率均保持在 80% 以上。

相比之下，Qwen3-4B 则展现出不同的策略偏好：其虽然实现了 89.0% 的高价值履约率，但低优先级任务的完成率仅为 3.1%，即 Qwen 倾向于主动牺牲低优先级任务换取电网运行的松弛度。这一行为对比表明，不同开源基座在面对强物理约束时的博弈倾向存在细微分歧，而在工业实操中，Falcon 和 Llama 展现的“全局保供”策略更能满足严格的 SLA 业务协议。

### 现场验证：从模拟器走向真实电网监控大屏

论文并未止步于合成基准测试，而是将 PowerAtlas 接入了中国某省级电力公司的真实工业试验网络，直面覆盖该省境内 3 座真实数据中心节点的协同调度场景。

<img src="/images/2607.26710v1/fig7.webp" alt="省级电力系统电算协同调度实验网络监控运营大屏" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在上图展示的运营大屏中，PowerAtlas 实时抓取三座数据中心的在线设施状态、可用机柜资源、算力排队队列以及电网侧的实时潮流量测。模型根据这些工况动态生成包含机组日前出力规划与跨中心算力负载迁移的联合指令。

由于电力系统具有极高的安全防护红线，当前现场部署采用了“影子模式”（Shadow Execution）：Agent 生成的电网调度建议不会直接下发给发电机执行机构闭环控制，而是交由系统调度员作为辅助决策参考，并同步送入离线潮流模型进行合规校验。这一工业验证落地有力证明了基于物理感知大模型的协同框架完全具备兼容现有电力调度 SCADA 系统和算力集群监控接口的能力。

### 范式跃迁：大模型接管复杂工程物理系统的关键启示

PowerAtlas 的成功为学术界和工业界探索大模型赋能高可靠物理系统（Cyber-Physical Systems）提供了几点极为鲜明的工程启示：

首先是**告别纯运筹或纯概率的极端偏见**。传统观点认为大模型本质是随机统计生成，绝不可能涉足对安全性要求严苛的电力主网调度。PowerAtlas 证明，利用确定性物理校验器构建端到端 RL 奖励回传，小参数模型完全可以内化复杂的非线性潮流规律与拓扑限制，成为高并发场景下替代耗时求解器的有效方案。

其次是**重新审视智算中心与能源网络的生态位**。过去电力系统将数据中心视作被动承受供电的负荷端，而算力运营方则对用能约束浑然不觉。通过将机组经济调度与多中心任务时空平移联合建模，万卡集群不再是电网的负担，反而转化成了具备秒级至小时级响应能力的“虚拟电厂（Virtual Power Plant）”。

当 AI 发展的飞轮加速旋转，算力与能源的交织正在重塑基础设施的底层逻辑。PowerAtlas 展现的技术路径表明，依托物理感知强化学习，大模型完全有能力在电网毫秒级安全红线与海量算力任务之间找到最优的平衡支点。
