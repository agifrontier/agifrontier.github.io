---
layout: default
title: "WCM：把世界模型塞进Critic，让具身VLA在149项任务上击败全量SFT"
description: "在视觉-语言-动作（VLA）模型的后训练阶段，强化学习（RL）正在成为突破模仿学习天花板的关键路径。然而，当前主流的Actor-Critic范式在机器人操控任务中遭遇了一个隐蔽却致命的瓶颈：大多数Critic模型要么只基于单帧视觉观测，要么直接依赖预训练VLM的单帧隐层特征来估计状态价值。"
arxiv_id: "2607.29613"
published_at: "2026-09-04T13:15:08.122237+08:00"
topics:
  - "具身智能"
  - "强化学习"
tags:
  - "具身智能"
  - "强化学习"
  - "AI论文解读"
related_tutorials:
  - "world-model-for-robot-learning-a-comprehensive-survey"
  - "\u03c0_0-a-vision-language-action-flow-model-for-general-robot-control"
  - "dr-well-dynamic-reasoning-and-learning-with-symbolic-world-model-for-embodied-ll"
  - "efficient-reinforcement-learning-for-large-language-models-with-intrinsic-explor"
---

<p class="paper-original-title" lang="en">WCM: A World Critic Model for Vision-Language-Action Reinforcement Learning</p>

在视觉-语言-动作（VLA）模型的后训练阶段，强化学习（RL）正在成为突破模仿学习天花板的关键路径。然而，当前主流的 Actor-Critic 范式在机器人操控任务中遭遇了一个隐蔽却致命的瓶颈：大多数 Critic 模型要么只基于单帧视觉观测，要么直接依赖预训练 VLM 的单帧隐层特征来估计状态价值。机器人操控本质上是一个部分可观测马尔可夫决策过程（POMDP），单帧画面即使能够呈现物体的视觉外观与几何轮廓，也必然丢失速度、接触变化、受力趋势等核心物理动力学。

> ArXiv URL：https://arxiv.org/abs/2607.29613

针对这一问题，直觉上的解决方案是给 Critic 喂入多帧历史图像。但直接拼接历史帧在面对高维视觉特征时不仅会带来计算复杂度的急剧膨胀，在实践中更是频繁失效。根本原因在于，强化学习中单纯的标量回报（Scalar Return）回归，为跨时序动力学表征学习提供的监督信号过于稀疏和微弱。网络极易将多帧输入退化为静态特征拼接，而无法学到环境演化的内在物理规律。

<img src="/images/2607.29613/main_fig.webp" alt="WCM总体架构与传统Critic对比示意" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了打破这一表征瓶颈，研究者提出了世界评论家模型（World Critic Model，简称 WCM）。该方案的核心突破在于：将“世界模型对未来隐状态的预测”与“对累积回报的价值估计”融为一体。WCM 采用轻量化的 LeJEPA 架构，在输入历史观测后，强迫特征提取器在评估当前价值的同时，显式预测执行动作后的下一时刻隐状态。通过引入密集的自监督动力学预测目标，WCM 迫使 Critic 真正构建出贴合 POMDP 物理现实的紧凑状态表征。实验表明，WCM 在横跨四大基准的 149 个操控任务中均取得了显著突破，甚至仅用单条演示轨迹起步，经过强化学习后便击败了用两万条专家轨迹训练的全量 SFT 模型；在真实机械臂操控中，仅耗费不到一小时的微调，就在 7 项复杂任务上实现了稳定部署。

### 为什么单纯给 Critic 加历史帧并不奏效？

在传统的强化学习设定中，马尔可夫性假设当前观测即代表完整状态。但在真实的机器人抓取、装配、布料折叠等交互场景中，这只是一个过于理想化的假设。当夹爪正在闭合、机械臂正在加速或物体即将发生形变时，单张静止图像无法区分“刚接触物体”“正在施加挤压力”与“已经脱手滑落”。在 POMDP 形式化描述中，智能体需要依赖历史轨迹的充分统计量来推断当前的真实物理状态。

<img src="/images/2607.29613/graphical_model.webp" alt="POMDP环境下的部分可观测图模型" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

图模型直观展现了这种信息缺失：环境隐状态 $s_t$ 随动作 $a_t$ 迁移至 $s_{t+1}$，但智能体在每个时刻仅能捕获部分观测 $o_t$。此前部分研究尝试通过帧堆叠（Frame Stacking）或引入时序注意力层来缓解这一问题，但在高维具身感知空间中，这类尝试往往收效甚微。

问题的症结在于监督信号的不对称性。深度强化学习中的 Critic 通常使用均方误差损失去逼近蒙特卡洛回报或 TD 目标，这是一个维度极低的时序标量。面对输入端数以万计的视觉 Token，标量回归反向传播回来的梯度极其匮乏，根本不足以指导巨大的参数空间去解耦“外观背景”与“时序物理演化”。Critic 的深层网络往往倾向于“走捷径”，把堆叠的多帧拼成一个更胖的静态特征，忽略帧与帧之间的时序推移。大语言模型之所以具备极强的通用推理能力，本质在于 Next-token Prediction 提供了海量且密集的预测监督信号。表征学习的核心原则同样表明：一个优良的状态表征，必须具备预测自身未来的能力。

### WCM架构：世界模型预测与价值回归的深度交织

为了在不大幅增加计算负担的前提下重塑 Critic，WCM 借用了面向联合嵌入预测架构（JEPA）的轻量级 LeJEPA 设计理念，将整个 Critic 塑造成一个既能预测动态演变、又能评估收益的双功能网络。

<img src="/images/2607.29613/method.webp" alt="WCM详细架构与On-policy/Off-policy训练流" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整个 WCM 模型的数据流动逻辑清晰且高度模块化。首先，智能体过去 $K$ 帧的视觉观测序列 $o_{t-K+1:t}$ 被送入观测编码器 $\mathrm{enc}_{\epsilon}$，映射为隐层状态序列：




{% raw %}$$ {\mathbf{z}}_{t-k}=\mathrm{enc}_{\epsilon}(o_{t-k}), \quad \forall k\in\{0,1,\cdots,K-1\} $${% endraw %}



语言指令 $\ell$ 则通过预训练语言模型（如 CLIP）提取为紧凑的语义向量：




{% raw %}$$ \mathbf{u}_{\ell}=\mathcal{A}_{\mathrm{lang}}\!\left({\mathrm{CLIP}}(\ell)\right)\in\mathbb{R}^{d} $${% endraw %}



随后，时序 Transformer 模块 $\text{Tr}_{\phi}$ 结合交叉注意力机制，将多帧隐状态序列与任务语言指令进行时序特征融合，提取出蕴含任务上下文的高阶历史表征：




{% raw %}$$ {\mathbf{h}}_{t}=\text{Tr}_{\phi}(\operatorname{XAttn}\left({\mathbf{z}}_{t-K+1:t},\mathbf{u}_{\ell}\right))\;\in\;\mathbb{R}^{d} $${% endraw %}



与常规 Critic 仅从此特征接入一个全连接层输出标量不同，WCM 在高阶表征 ${\mathbf{h}}_t$ 的末端分叉出了两个互为协同的解码器头：一个是负责价值评估的价值解码头 $\mathcal{D}_{\text{value}}$，输出当前状态的预期价值估计：




{% raw %}$$ \hat{V}_{t}=\mathcal{D}_{\text{value}}({\mathbf{h}}_{t})\;\in\;\mathbb{R} $${% endraw %}



另一个则是世界动力学预测解码头 $\mathcal{D}_{\text{world}}$。它不仅接收当前聚合表征 ${\mathbf{h}}_t$，还将当前采取的具体动作 $a_t$ 以及最新一帧的隐状态 ${\mathbf{z}}_t$ 作为联合条件输入，显式预测系统执行动作后的下一时刻隐状态：




{% raw %}$$ \hat{{\mathbf{z}}}_{t+1}=\mathcal{D}_{\text{world}}({\mathbf{h}}_{t},a_{t},{\mathbf{z}}_{t})\in\mathbb{R}^{d} $${% endraw %}



预测目标通过隐空间距离损失进行自监督优化：




{% raw %}$$ \mathcal{L}_{\text{pred}}=\|\hat{{\mathbf{z}}}_{t+1}-{\mathbf{z}}_{t+1}\|_{2}^{2} $${% endraw %}



在自监督表征学习中，直接在隐空间进行预测极易遭遇表征坍塌（Representation Collapse）——即网络退化为将所有状态映射为同一常数向量，从而使预测损失平凡地降至零。为了从数学原理上杜绝坍塌，WCM 引入了 SIGReg（Sketch Information Gradient Regularization）谱正则化项：




{% raw %}$$ \mathcal{L}_{\text{SIGReg}}=\mathbb{E}_{{\mathbf{a}}\sim\mathcal{U}(\mathcal{S}^{d-1})}\left[\int_{{\mathbb{R}}}\left|\hat{\phi}_{{\mathbf{a}}^{\top}{\mathbf{z}}}(t)-\phi(t)\right|^{2}e^{-t^{2}}\,dt\right] $${% endraw %}



该项通过匹配随机投影下的特征经验特征函数与标准高斯特征函数 $\phi(t)$，强制要求学习到的隐状态各维度分布维持非退化的高斯分布，从而锁死空间坍塌的可能。最终，结合标准的价值回归损失 $\mathcal{L}_{\text{value}}=\|\hat{V}_{t}-G_{t}\|_{2}^{2}$，WCM 构建出兼顾长线动力学与即时回报的联合端到端损失函数：




{% raw %}$$ \mathcal{L}=\mathcal{L}_{\text{value}}+\lambda\cdot\mathcal{L}_{\text{pred}}+\eta\cdot\mathcal{L}_{\text{SIGReg}} $${% endraw %}



通过超参数 $\lambda$ 与 $\eta$ 的协同调控，世界预测目标在反向传播中为共享时序编码器注入了强烈的物理演进约束。Critic 不再是一个单纯猜数字的评分器，而是一个必须在脑海中推演动作后物理世界如何变化的微型仿真器。

### 全面兼容主流VLA架构的统一训练流

具身智能领域的基座模型目前呈现两大路线并存的格局：以 OpenVLA 为代表的自回归（Autoregressive）模型，其利用 Next-token 预测离散化的动作 Token；以及以 Physical Intelligence 提出的 $\pi_0$ 与 $\pi_{0.5}$ 为代表的流匹配（Flow-matching）连续动作生成模型。WCM 展示出了极高的模块兼容性，能够无缝嵌合到这两类主流架构中。

在在线强化学习（On-policy）设定中，对于自回归模型，WCM 与标准 PPO 结合；对于通过常微分方程（ODE）确定性采样的流匹配模型，则与 Flow-SDE 对齐，通过向扩散过程注入随机性构建强化学习管线。WCM 在交互采样过程中准确推断每一步状态的基线价值，用于计算 Generalized Advantage Estimation（GAE）优势函数，以此稳定驱动 Actor 网络的梯度上升。

在样本效率要求极高的离线策略（Off-policy）设定下，WCM 的价值预测与动态模拟能够有效抑制因分布外动作带来的价值高估。对于自回归模型，算法接入优势加权回归（AWR）；对于流匹配模型，算法与 RECAP 机制深度结合。此时，回放缓存区中融合了遥控示范的 SFT 优质数据、在线采样的探索轨迹以及典型的失败翻车案例。多源轨迹的数据分布极为庞杂，WCM 的世界预测目标能够精准刻画因果关系，使得 Critic 在面对错误轨迹时不会盲目乐观，大幅提升了离线样本的利用效率。

### 149项任务基准检验：泛化与跨越式超越

为了全面压榨 WCM 的性能极限，实验在 ManiSkill、MetaWorld、CALVIN 以及 LIBERO-Plus 四大仿真基准的 149 个操控任务上展开，全面覆盖常规抓取、复杂工具使用、多阶段长程规划及几何/外观扰动等场景。

在 ManiSkill 基准测试中，评估不仅考察域内（In-Distribution, IND）的表现，还沿空间布局、物体几何与纹理外观三大轴向测试分布外（Out-of-Distribution, OOD）的零样本泛化能力。实验对比了采用单帧 Critic 的 Flow-Noise、Flow-SDE、$\pi$-stepNFT 以及 PPO 和 GRPO。

数据表明，无论是基于连续扩散动作的 $\pi_0$、$\pi_{0.5}$，还是离散 Token 架构的 OpenVLA-OFT，换装 WCM 均带来了质的飞跃。在域内操控中，原本单帧 Critic 的成功率提升容易在特定瓶颈步数停滞，而接入 WCM 后的 Flow-SDE 与 PPO 在各项指标上均取得了 10 到 20 个百分点以上的附加净增益（$\Delta$）。

更为震撼的结果出现在强调泛化能力的 LIBERO-Plus 基准中。基准设置了两种极端条件：一种是喂入 50 条专家演示（总计两万条完整轨迹）训练出的“全量 SFT”基线；另一种则是仅见识过 1 条专家演示的“One-shot SFT”。在初始化仅基于单条演示极度匮乏的条件下，智能体搭载 WCM 仅仅经历了约 250 步的强化学习交互，其在多任务套件上的综合成功率便彻底超越了基于两万条轨迹精细调优的 Full-SFT 模型。这直接证明了：当 Critic 具备推断物理演进与长程动态的能力时，强化学习能够摆脱对海量专家示范的重度依赖，从微量种子数据中自主提炼出极具鲁棒性的操控策略。

在真实机器人场景中，由于物理机械臂试错成本高昂，样本效率是唯一的试金石。团队使用 OpenVLA-OFT（基于 AWR）与 $\pi_{0.5}$（基于 RECAP）在物理平台上评估了 7 项极具挑战性的操控任务：包括抓取高速旋转传送带上寿司的动态抓取、衣物与毛巾折叠等可形变物体操控、多阶段灶台擦拭的长程任务，以及不同几何蔬果的 Pick-and-Place。

在对照实验中，基线方法使用参数量更大的通用视觉语言模型 Gemma-270M 作为 Critic，而 WCM 的可学习参数量仅为 107.2M。实验仅提供每项任务 100 条基础 SFT 轨迹，进行 8 轮离线强化学习迭代，每轮仅补充 50 条 Rollout。在短短不到一小时的物理微调时间内，WCM 驱动的机械臂在全部 7 个任务上的最终抓取与操控成功率均全面碾压 Gemma 基线，且执行动作的机械卡顿大幅减少，轨迹平滑度与抗扰动表现均呈现出压倒性优势。

### 深度消融：世界模型预测究竟扮演了什么角色？

WCM 的成功到底是因为增加了多帧参数量，还是世界预测目标真正起到了催化作用？为了回答这一核心疑问，研究设计了严格的消融对比实验，重点考察了 Critic 结构与历史时序长度对最终性能的影响。

<img src="/images/2607.29613/wcm_3_subplots_3d_ribbons.webp" alt="Critic架构与观测历史长度消融曲面" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

三维曲面消融实验彻底击碎了“单纯增加输入维度就能解决 POMDP”的侥幸假设。实验测试了三种 Critic 架构：第一种是基础的 MLP Critic，粗暴地把 1 到 5 帧的历史特征串联输入；第二种是将 WCM 中的未来预测损失权重设为零（$\lambda=0$）的时序 ViT 变体，此时模型具备完整的时序注意力结构，但完全依赖标量价值误差进行梯度更新；第三种则是具备完整预测目标的标准 WCM。

结果呈现出泾渭分明的断层差距：

在单纯向原版 MLP 堆叠多帧历史时，随着帧数增加，模型性能并未单调上升，反而经常发生性能衰退或震荡。这印证了此前理论界的预警：高维特征拼接在稀疏标量监督下会放大过拟合。

更为关键的是，即便引入了能够精细建模时序注意力的 ViT，只要缺失了未来隐状态预测目标（$\lambda=0$），其表现依旧显著受阻，成功率始终徘徊在较低水平。

唯有当 $\lambda > 0$、端到端接入世界模型预测损失时，网络性能才呈现全面爆发。这一反差直接在经验上证实：在部分可观测的具身操作中，世界模型预测目标并非可有可无的辅助任务，而是激活历史特征时序表征能力的关键抓手。

另一个有趣的发现围绕历史观测窗口的最佳长度展开。直觉上或许会认为观测历史越长越好，但实验数据绘制出的却是一条鲜明的抛物线。在输入长度为 1 到 5 帧的跨度测试中，性能在长度为 3 时达到了稳定的全局峰值，当长度进一步拉长到 4 或 5 帧时，操控收益反而出现边际效益递减甚至微弱下滑。

这一现象在物理学上具有直观且深刻的解释。机器人的操控物理本质上由经典力学主导：两帧连续画面能够构建位置的一阶导数（即速度与运动方向）；而三帧连续画面则足以解析位置的二阶导数（即加速度与合外力演变）。对于绝大多数接触装配、轨迹追踪与形变操控任务，捕捉到一阶速度与二阶加速度动力学，就已经满足了恢复系统局部隐状态的充分统计要求。当历史窗口进一步拉长至四帧以上时，不仅为高维网络引入了冗余注意力负担，还可能引入不同延迟阶段的累积噪声，反而削弱了对瞬时接触事件的判断精度。因此，将历史窗口定为 3 帧，在计算资源消耗与物理动力学建模之间构成了最优的帕累托平衡点。

### 对具身强化学习演进路线的启示

WCM 的提出不仅仅是刷榜了一个算法指标，更对当前具身智能领域的基准设计与架构选型给出了重要启示。

长久以来，学术界与工业界将大量精力倾注在 Actor 策略端的架构膨胀上，试图通过堆叠数十亿参数的单帧 VLM 来暴力逼出泛化操控策略。但机器人控制在本质上受制于 POMDP 的客观规律，输入端的瞬时感知盲区无法仅靠下游网络“强行脑补”。如果负责提供强化学习优化方向的 Critic 自身处于“盲人摸象”的窘境，无论 Actor 策略空间多么广阔，强化学习更新都会被高方差、高偏差的价值梯度带偏。

WCM 证明了轻量级、面向未来动力学推演的联合建模可以四两拨千斤。仅需 100M 上下的可学习参数量，就能将原本只能做单帧拟合的 Critic 改造为具备物理推演能力的时序世界评估器。通过在隐空间预测未来状态，隐式赋予了模型推演物理因果的能力，这种动力学感知能力天然地与任务具体外观解耦，因而在跨越光照、背景、物体几何畸变等分布外环境时展现出极高的泛化韧性。

当具身智能演进到以后训练强化学习为核心的深水区时，如何在部分可观测世界中稳固价值基准，正在成为决定策略成败的真正分水岭。WCM 揭示的方向表明：不理解物理世界未来走向的 Critic，算不出当下真正的价值。将世界模型融入强化学习评估链路，或将成为具身大模型真正走向开放物理世界的一块关键技术拼图。
