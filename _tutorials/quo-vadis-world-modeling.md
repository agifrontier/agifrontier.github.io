---
layout: default
title: "世界模型最新综述！颜水成等提出World Proxy：六大形态与三级进阶重塑智能体交互"
description: "为此，研究团队提出了“以智能体为中心的世界代理”（Agent-Centric World Proxy）这一新范式，将世界模型从单调的“物理状态预测器”升级为能够处理执行、记忆、技能与校验的“信息状态转移器”。"
arxiv_id: "2608.02713"
paper_published: "2026-08-03"
published_at: "2026-09-30T13:15:08.320273+08:00"
topics:
  - "AI Agent"
tags:
  - "AI Agent"
  - "AI论文解读"
related_tutorials:
  - "antares-foundation-models-for-agentic-vulnerability-localization"
  - "sg-wam-self-guided-world-modeling-in-geometry-aware-policy-space"
  - "osreward-instituting-standardized-evaluation-for-cross-platform-computer-use-rew"
  - "qwen-ui-agent-technical-report-toward-next-generation-real-world-centric-foundat"
seo_title: "世界模型最新综述！颜水成等提出World Proxy：六大形态与三级进阶重塑智能体交互"
---

<p class="paper-original-title" lang="en">Quo Vadis, World Modeling?</p>

过去两年里，关于世界模型（World Model）的讨论几乎被视频生成模型的声音所垄断。从 Sora 到各种基于扩散模型的长视频生成器，技术界往往将“能否逼真模拟物理世界的光影与重力”视作世界模型的唯一考量。然而，一个不可回避的尴尬事实摆在面前：视觉保真度越来越高的高清视频生成系统，在接入实际的具身智能或数字智能体（Agent）后，往往很难直接转化为鲁棒的决策能力。耗费巨大算力渲染出的一帧帧高分辨率画面，在很多时候对智能体的动作规划并没有实质帮助。

> ArXiv URL：https://arxiv.org/abs/2608.02713

智能体究竟需要什么样的世界模型？由颜水成等学者共同撰写的新文《Quo Vadis, World Modeling?》，从根本上重新审视了世界模型的目标。文章指出，传统的物理状态转移模型只是智能体与世界交互的一个狭窄子集。面对试错成本高昂、危险且无法大规模并行的真实世界，智能体更迫切需要的是一个能提供低成本、可控反馈的中间介质。为此，研究团队提出了“以智能体为中心的世界代理”（Agent-Centric World Proxy）这一新范式，将世界模型从单调的“物理状态预测器”升级为能够处理执行、记忆、技能与校验的“信息状态转移器”。

<img src="/images/2608.02713/teaser.webp" alt="以智能体为中心的世界代理设计空间与核心转变" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 从真实世界试错的瓶颈，到世界代理的重塑

任何能够持续进化的智能体，其核心能力都不是在静态预训练中一蹴而就的，而必须在“行动—环境反馈—能力迭代”的闭环中不断磨砺。然而，直接将未经充分打磨的策略投放至物理世界或生产级数字环境中，往往会遭遇四大难以逾越的瓶颈：

第一，直接交互具有极高的高昂成本与物理损耗。机器人摔倒、机械臂碰撞不仅损伤硬件，在医疗、自动驾驶等场景更会引发灾难性后果；第二，真实世界是单向流动的，物理过程无法被任意撤销，使得智能体难以执行假设性推理（Counterfactual Reasoning）；第三，物理环境难以进行大规模高并发并行模拟，受限于真实时间的流逝；第四，纯粹的环境观察信号极其嘈杂，智能体常常需要经历数百万次探索才能偶然捕获一次有效的稀疏奖励。

<img src="/images/2608.02713/agent_wm_env.webp" alt="智能体、世界代理与真实环境的关系" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了破解这一死局，研究者往往会在智能体与真实环境之间安插一个缓冲层。但传统的经典定义将世界模型局限在如下马尔可夫决策过程的动力学过渡中：




{% raw %}$$ \hat{s}_{t+1} = \mathcal{WM}(s_t, a_t) $${% endraw %}



输入当前环境状态 $s_t$ 与动作 $a_t$，输出下一个环境物理状态 $\hat{s}_{t+1}$。这种建模方式默认智能体的核心需求是预知未来的物理传感器输入。然而，在现代大语言模型驱动的代码编写、网络检索、复杂工作流或工具调用任务中，环境并不总是由连续的物理坐标构成的。代码执行返回的报错信息、数据库的查询结果、过往类似失败案例的警示，本质上全都是智能体赖以生存的“环境反馈”。

因此，论文将传统的 World Model 拓展为更加通用的 World Proxy（世界代理）。这一转变的核心在于：代理输出的目标不再仅是环境本身的物理状态，而是智能体可直接利用的“信息状态”（Agent-usable Information State）。无论是调用记忆模块、预测代码执行报错，还是评估当前方案的可行性，都可以被形式化为同一种交互过程：




{% raw %}$$ \hat{s}_{\ell+1} = \mathcal{WP}\left(s_\ell, u_\ell^{\mathcal{F}}\right) $${% endraw %}



这里，步长 $\ell$ 不再受物理时间约束，而是代表智能体与代理之间的交互轮次；广义状态空间 $\mathcal{S}$ 囊括了观测、记忆、代码运行结果、安全约束与验证信号；$u_\ell^{\mathcal{F}}$ 则是智能体针对特定代理功能 $\mathcal{F}$ 主动发起的查询或动作，$\hat{s}_{\ell+1}$ 则是世界代理返还的结构化反馈。

<img src="/images/2608.02713/world_proxy.webp" alt="世界代理架构将交互拓展为更广泛的信息转移" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 阶梯式赋能：从即时顾问到协同进化

有了 World Proxy 的定义后，它究竟如何在算法层面推动智能体的能力提升？本文建立了一套清晰的三级赋能阶梯（L1至L3），深刻刻画了代理系统在智能体生命周期中扮演的角色变化。

<img src="/images/2608.02713/L1_to_L3.webp" alt="智能体在世界代理赋能下的三级进阶示意" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### L1：推理期引导（Inference-Time Guidance）

在这一层级，智能体自身的模型权重完全被冻结，世界代理扮演的是“战术顾问”的角色。智能体在做出关键决策前，先向世界代理提出试探性查询：




{% raw %}$$ \hat{s}_{\ell+1}^{\text{guide}} = \mathcal{WP}\left(s_\ell^{\text{agent}}, u_\ell^{\mathcal{F}}\right), \qquad s_\ell^{\text{agent}+} = s_\ell^{\text{agent}} \oplus \hat{s}_{\ell+1}^{\text{guide}} $${% endraw %}



代理返回的模拟推演、检索结果或可行性反馈，被拼接到智能体当前的上下文窗口中。这种方式使智能体能够在不改动自身参数的前提下，在单次任务执行中实现“看多几步、算多几步”。它就像下棋时在脑海中虚拟复盘数步之后的棋局，大幅降低了在物理环境中盲目探索带来的试错成本。

#### L2：训练期优化（Training-Time Optimization）

如果说 L1 只是为智能体提供了耳边叮嘱，那么 L2 则直接深入到参数层面重塑智能体的决策策略。在训练阶段，世界代理转型为奖励模型、判别器（Critic）或合成环境模拟器：




{% raw %}$$ \hat{s}_{\ell+1}^{\text{opt}} = \mathcal{WP}\left(s_\ell^{\text{agent}}, u_\ell^{\mathcal{F}}\right), \qquad \text{agent}^{+} = \mathrm{Train}\left(\text{agent}, \hat{s}_{\ell+1}^{\text{opt}}\right) $${% endraw %}



智能体不再需要依赖极其昂贵的真实人类标注或现实环境回传，而是可以在世界代理搭建的高并发沙盒中进行数万次强化学习探索与策略梯度优化。代理不仅能够评估动作质量，还能生成反事实轨迹（Counterfactual Rollouts）与偏好对比对。智能体的能力上限在这一层级被彻底打破，但这也伴随着更严格的前提：代理生成的奖励信号若存在微小偏差，往往会被强化学习算法过度利用，从而引发严重的策略退化。

#### L3：智能体与代理协同进化（Agent-Proxy Co-Evolution）

当智能体迈入开放世界，静态的世界代理必然会遭遇“分布偏移”（Distribution Shift）。现实世界的规则在变化，代理不可能全知全能。在最高层级的 L3 中，世界代理不再是一个固定不变的训练设施，而与智能体形成了共生演化的双飞轮：




{% raw %}$$ (\text{agent}^{+}, \mathcal{WP}^{+}) = \mathrm{CoEvolve}\left(\text{agent}, \mathcal{WP}, \hat{s}_{\ell+1}^{\text{proxy}}, s_{\ell+1}^{\text{env}}\right) $${% endraw %}



智能体在真实环境部署中遇到的异常案例与未知状态，会作为宝贵的新数据源回传给世界代理，驱动代理更新其内部机制与知识库；而变得更加精准、鲁棒的世界代理，又能反哺智能体提供更可靠的合成训练信号与推理指导。

以一个执行多步骤跨境电商采购的 Web Agent 为例，这三个层级的递进展现得尤为清晰：在 L1 阶段，智能体在点击“确认下单”前，先由世界代理模拟预测页面跳转，若发现隐性税费增加则主动回退调整；在 L2 阶段，该代理在离线环境下合成成千上万个虚拟电商结账页面，对智能体进行大规模强化学习训练；而在 L3 阶段，真实网站前端架构改版导致购买失败的真实日志，被自动抽取并用于更新世界代理的执行模拟器，使得两者在动态网络环境中持续同步进化。

### 解构世界代理的六大功能形态

世界代理并非单一的技术模型，而是一个由不同职能构成的工具谱系。作者团队根据代理所承载的信息模态与反馈机制，系统性梳理出六大功能形态。

<img src="/images/2608.02713/proxy_dimensions.webp" alt="智能体中心世界代理的六大功能形态" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

**1. 动力学代理（Dynamics Proxy）：** 这正是传统世界模型的经典形态。它的核心数学形式依然关注智能体施加动作后的环境物理转移：




{% raw %}$$ \hat{s}_{\ell+1}, \hat{r}_{\ell+1} = \mathcal{WP}^{\mathrm{dyn}}\left(s_\ell, u_\ell^{\mathrm{dyn}}\right) $${% endraw %}



动力学代理专注于解答：“如果智能体执行该动作，物理世界将如何演进？”它在机器人操作、连续运动控制中始终占据基石地位。

**2. 空间代理（Spatial Proxy）：** 区别于预测时间的演进，空间代理聚焦于视角的拓展与三维几何的重建：




{% raw %}$$ \hat{o}_{\ell+1}^{\mathrm{view}} = \mathcal{WP}^{\mathrm{spatial}}\left(s_\ell, u_\ell^{\mathrm{spatial}}\right) $${% endraw %}



智能体输入目标视点、姿态或空间坐标，代理合成出对应的空间全景或隐式表征（如基于 NeRF 或 3DGS 的渲染）。它让只有局部视野的智能体能够获得全局感知，有效解决了具身智能在复杂遮挡环境中的寻路与避障难题。

**3. 执行代理（Execution Proxy）：** 随着数字智能体与代码大模型的兴起，执行代理成为了近期的研究重点。它专门用于应对软件系统与代码沙盒的离散转移：




{% raw %}$$ \hat{s}_{\ell+1}^{\mathrm{exec}}, \hat{y}_{\ell+1}^{\mathrm{exec}} = \mathcal{WP}^{\mathrm{exec}}\left(s_\ell, u_\ell^{\mathrm{exec}}\right) $${% endraw %}



智能体输入待执行的代码片段或网页点击事件，执行代理无须调用昂贵的真实基础设施，即可预测标准输出、错误堆栈（stderr）或 DOM 树的变化。

**4. 记忆与经验代理（Memory/Experience Proxy）：** 复杂的任务往往需要调取过往的先验。记忆代理并不模拟世界的变化，而是充当环境约束与历史教训的高效中继站：




{% raw %}$$ \hat{m}_{\ell+1} = \mathcal{WP}^{\mathrm{mem}}\left(s_\ell, u_\ell^{\mathrm{mem}}\right) $${% endraw %}



它根据智能体当前的子目标，精确检索历史相似任务中的失败陷阱或长程依赖信息，充当智能体的海马体。

**5. 技能代理（Skill Proxy）：** 面对长程规划任务，智能体从零进行底层动作探索的效率极低。技能代理根据任务意图匹配可复用的行为模块：




{% raw %}$$ \hat{g}_{\ell+1}^{\mathrm{skill}} = \mathcal{WP}^{\mathrm{skill}}\left(s_\ell, u_\ell^{\mathrm{skill}}\right) $${% endraw %}



代理返回的是经过封装的工具调用序列、动作先验（Action Prior）或子策略（Sub-policy），将底层控制从原始连续动作空间提升到高层语义策略空间。

**6. 奖励与验证代理（Reward/Verification Proxy）：** 这是强化学习与可靠推理的核心组件：




{% raw %}$$ \hat{v}_{\ell+1}^{\mathrm{eval}} = \mathcal{WP}^{\mathrm{eval}}\left(s_\ell, u_\ell^{\mathrm{eval}}\right) $${% endraw %}



它负责研判智能体生成的计划、代码或轨迹是否满足安全性、正确性与任务目标，输出精确的过程标量奖励或自然语言诊断批评，直接为策略优化提供驱动力。

这六大功能形态与前述的三级赋能阶梯是完全正交的。任何一种代理功能，既可以在推理阶段充当上下文提示词（L1），也可以在离线阶段用于训练样本生成与评分（L2），更可以随着部署过程不断吸收现实世界的真实回执，完成参数或索引库的闭环更新（L3）。这种正交解耦极大地拓宽了世界模型的设计与评测空间。

### 迈向真正可用的世界模型：四大待解挑战

重新定义范式仅仅是第一步，要让以智能体为中心的世界代理真正落地，文中尖锐地指出了当前技术路径上难以回避的四个深层挑战：

第一是视觉逼真度与动力学保真度的脱节。目前主流的扩散生成式模型可以生成极具视觉冲击力的视频片段，但往往在底层物理逻辑（如物体穿模、质量守恒、重力加速度）上出现严重失真。在长程自回归推演中，这些微小的误差会呈现指数级累积。智能体如果依赖这些看似逼真、实则充满物理违背的幻觉数据进行决策，在现实中往往会产生致命操作。未来的核心攻坚点不在于更高的视频分辨率，而在于能够量化自身不确定性的置信度校准机制。

第二是智能体何时应当信任代理的决策博弈。如果智能体将世界代理盲目视作绝对正确的权威神谕，一旦代理产生盲区，智能体就会陷入静默失败。智能体必须在推演过程中具备动态评估能力：明确识别当前任务状态处于代理的熟练分布内，还是已经滑向未知的盲区，并据此决定究竟是依赖内部虚拟仿真，还是主动请求昂贵却真实的物理交互。

第三是奖励黑客行为（Reward Hacking）与安全边界。在 L2 阶段，当代理作为验证器与奖励提供者介入强化学习时，强大的强化学习策略极易捕捉到代理评分函数中的细微漏洞，产生大量虚高得分却在现实中完全无效的“投机动作”。世界代理在提供安全探索沙盒的同时，其自身的评价漏洞也构成了新的系统脆弱面。

第四是以信息增益为核心的评测体系重构。长期以来，学术界评估世界模型主要依赖 FVD（Frechet Video Distance）、PSNR、SSIM 等纯粹衡量像素质量与视觉连贯性的静态指标。这类脱离智能体实际效用的指标无法回答一个核心问题：这个世界模型究竟让智能体的决策性能提高了多少？建立面向动作成功率、样本探索效率与泛化鲁棒性的“以智能体为中心”的基准评测体系，已成为当务之急。

### 总结

世界模型的本质并不是造就一台高精度的造梦机，去追求毫无死角的物理现实复刻；它的根本价值，始终取决于它能为与之交互的智能体提供多大的信息增益与成长加速。《Quo Vadis, World Modeling?》用“世界代理”（World Proxy）这一概念，为长期处于单向感知模拟的技术路线补齐了缺失的关键拼图。

未来的世界模型不应止步于被动地展示“世界会怎样”，而应敏锐地回应智能体提出的每一个假设性问题。从单纯的物理推演，转向涵盖执行、记忆、验证与技能的广义信息交互；从孤立的离线模拟器，演变为与智能体共同进化的协同伙伴。当世界模型真正以智能体的行动与进化为中心时，通用自主系统的真正飞跃才有可能到来。
