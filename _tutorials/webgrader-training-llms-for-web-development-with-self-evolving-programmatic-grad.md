---
layout: default
title: "WebGrader：可执行评测自进化，8B模型网页生成逆袭480B巨兽"
description: "来自北京理工大学、北京交通大学和中国人民大学的研究团队提出了 WebGrader ，针对上述痛点提供了一种全新解法：他们将开放式网页生成的强化学习表述为一个 可执行奖励构建（Executable Reward Construction） 问题。"
arxiv_id: "2608.06474"
paper_published: "2026-08-06"
published_at: "2026-09-09T13:15:08.606362+08:00"
topics:
  - "AI评测"
  - "模型训练"
tags:
  - "DOM-grounded evidence"
  - "Flow Contract"
  - "VLM"
  - "WG-core-250"
  - "WebGen-Bench"
  - "WebGrader"
related_tutorials:
  - "prune4web-dom-tree-pruning-programming-for-web-agent"
  - "rendering-in-the-loop-an-execution-driven-agent-for-interactive-web-development"
  - "online-monitoring-and-corrective-steering-of-programming-agents"
  - "ouroboros-a-self-developing-frontier-coding-agent-with-reviewed-core-evolution"
---

<p class="paper-original-title" lang="en">WebGrader: Training LLMs for Web Development with Self-Evolving Programmatic Grader</p>

<img src="/images/2608.06474v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

让大语言模型写出一个可运行的单页面应用，如今已不是新鲜事。但当用户提出带有复杂交互、状态持久化与表单联动的真实需求时，很多生成的网站往往“看起来很美，一点就崩”。为了补齐这种功能性短板，以强化学习（RL）为核心的后训练方案成为前沿探索的主流方向。然而，在数学求解或单纯的算法题中，环境天然附带精确的断言（Assert）与测试用例；在开放式网页开发场景中，提示词往往只描述了功能愿景，根本不会提供具体的浏览器交互动作与状态检查逻辑。

> ArXiv URL：https://arxiv.org/abs/2608.06474v1

强化学习的成败高度受制于奖励信号（Reward）的保真度。当前主流方案陷入了两难境地：人工编写浏览器测试脚本虽然精确、可复现，但面对千奇百怪的开放式需求，人工编写的扩展成本高到无法承受；如果改用基于视觉语言模型（VLM）或 GUI Agent 的多模态评测器，模型又极其容易在未触发关键交互状态前就妄下断言，甚至仅仅依据静态界面的美观度给出虚高的误判。

<img src="/images/2608.06474v1/webgrader_figure1_optimized.webp" alt="WebGrader 核心机制概览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

来自北京理工大学、北京交通大学和中国人民大学的研究团队提出了 **WebGrader**，针对上述痛点提供了一种全新解法：他们将开放式网页生成的强化学习表述为一个**可执行奖励构建（Executable Reward Construction）**问题。WebGrader 并不依赖主观打分，而是先将用户的自然语言需求解构为一系列可执行的“流程契约”（Flow Contract），在真实的 Chromium 浏览器中完成目标操作对齐与动作执行，并沿交互轨迹完整收集 DOM、网络响应、截图及本地持久化存储等多重证据。更关键的是，评测器本身并不僵化，团队设计了一套受神经架构搜索（NAS）启发的自进化机制，通过归因残差在离线环境中持续迭代评测技能图谱（SkillGraph），在彻底冻结评测器后再启动强化学习训练。

这一机制带来了惊人的下游泛化表现。在标准基准 WebGen-Bench 上，WebGrader 训练出的 8B 参数策略模型取得了 52.01% 的功能成功率（FSR），相比同等配置下的脚本奖励基线提升了 7.88 个百分点，一举超越了参数量大得多的 o4-mini 与 DeepSeek-v4-flash。在更严格的 WG-core-250 基准上，该 8B 模型甚至超越了 4800 亿参数的 Qwen3-Coder-480B。

### 为什么网页评测不能只靠“看截图”或“拼Prompt”？

要理解 WebGrader 的突破，必须先看清传统评测器在网页开发场景下的致命盲区。

一个完整的网站不仅是静态的代码文本或渲染出的单帧像素，它是一个具有内部状态机的交互产品。很多业务逻辑的缺陷具有很强的潜伏性（Latent Faults）。例如，用户点击“加入购物车”按钮，页面上的数字可能加了 1，但底层的 `localStorage` 并没有同步，或者在点击二次结算时才会抛出空指针异常；又或者一个多级联动表单，只有依次完成下拉选择、文本输入并触发模糊查询时，特定组件才会渲染。

如果直接使用 VLM 充当裁判，多模态模型往往缺乏严格的时间序列因果推断能力，容易出现“视觉欺骗”——看到页面呈现了表单与按钮，就直接判定功能正常。即使引入基于 GUI Agent 的多轮点击，Agent 也可能在未建立正确前置条件（Prerequisites）的情况下盲目点击，导致关键交互并未真正发生，随后给出不负责任的失败判定。

另一类直觉方案是将各种测试边界规则堆叠成一段极其庞大的长提示词（Rule-based Prompt）。但实验表明，评测模型面对不断膨胀的规则集时，上下文检索与遵从能力会急剧衰减，甚至引发规则之间的语义冲突。要让强化学习获得真正可靠的梯度方向，评测器必须像一位经验老道的自动化测试工程师：不仅能够根据需求自动撰写出端到端的 Playwright 测试用例，还能在真实的浏览器执行环境中拿到“铁证”，并在反复试错中自我修补测试盲区。

### 四阶段解耦与流程契约：把交互变成硬核证据

WebGrader 首先在架构上做出了关键切割：它拒绝让评测器在单一阶段同时完成理解、推演与裁决，而是将整个评估流水线严格解构为**测试规划（Test Planning）**、**动作对齐（Action Grounding）**、**证据收集（Evidence Collection）**与**语义判决（Semantic Judgment）**四个独立阶段。

<img src="/images/2608.06474v1/webgrader_full_pipeline_v2_optimized.webp" alt="WebGrader 端到端流水线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整个流程的起点是“需求优先”的规划。给定用户需求 $q$，系统首先推导出一组必须被满足的交互流集合 $\mathcal{F}_{\mathrm{req}}(q)$。需要特别强调的是，这一推导过程完全脱离生成的代码，从而杜绝评测器被模型写出的残缺代码“带偏”思路。

随后，每一个交互流会被形式化为一个**流程契约（Flow Contract）**：




{% raw %}$$ \mathcal{C}(q,x) =\{C_{f}(q,x)\}_{f\in\mathcal{F}_{\mathrm{req}}(q)}, \quad C_{f} =(P_{f},A_{f},a_{f}^{\star},E_{f},Q_{f},M_{f},w_{f}) $${% endraw %}



其中包含建立测试所需的前置状态 $P_f$、动作序列 $A_f$、核心目标操作 $a_f^{\star}$、需要捕获的证据类型 $E_f$、后置状态断言 $Q_f$、匹配策略 $M_f$ 以及该项功能的权重 $w_f$。

在动作对齐阶段，WebGrader 将流程契约映射到当前生成的具体前端项目 $x$ 中。利用生成的源代码和实时渲染的 live DOM 树，系统动态定位具体的选择器（Selector）与交互元素，生成无歧义的 Playwright 浏览器自动化脚本。

紧接着是严格的真实浏览器执行。Chromium 实例被启动并重置状态，脚本驱动页面流转，沿同一条交互轨迹实时记录渲染快照、DOM 节点树变化、网络请求与响应报文、URL 重定向以及 IndexedDB/LocalStorage 等持久化状态。

最后在判决阶段，评测器依据收集到的多维度因果证据判定当前流是否通过。只有当核心目标操作被真实触发、且状态迁移的前后差异完全符合断言时，才会给出“Pass”判决。如果目标动作无法对齐、执行抛错或后置状态未满足，直接打出“Fail”；而如果执行过程存在环境异常或无法消除的歧义，则标记为“Inconclusive”。通过这种层层锚定的机制，判决的随意性被完全剔除。

### 残差驱动的离线进化：NAS 式技能图谱筛选

再精密的初始评测器也难免存在盲区。如果评测器自身存在偏见或误判，RL 策略就会在训练中学会“奖励黑客（Reward Hacking）”，专门生成迎合评测器漏洞的代码。为此，WebGrader 构建了一个名为 WebGen-Verifier-100 的基准执行环境，专门用于评测器的离线打磨。

该环境包含 100 个带有完全验证的 React/Vite 干净应用（Clean Apps），并人为针对局部业务逻辑注入了 778 个细粒度的源级别独立故障（Controlled Faults），涵盖状态混乱、数据绑定失败、表单验证缺陷等典型错误。在这一环境下，评测器每一次判定是抓住了真实 Bug、误杀好代码，还是产生歧义，都可以被精确量化。

WebGrader 的进化循环借鉴了神经架构搜索（NAS）的哲学：

1. **残差定位（Residual Localization）**：当评测器在测试中出现假阳性（漏检故障）或假阴性（误报干净代码）时，离线闭环会将错误残差归因到特定阶段，即 $f_{\mathrm{attr}}(r) \in \{z_{\mathrm{plan}}, z_{\mathrm{ground}}, z_{\mathrm{evid}}, z_{\mathrm{judge}}\}$。

2. **定向变异（Stage-Specific Mutation）**：仅针对出错的阶段提出微调技能（Skills），例如针对动作对齐生成更鲁棒的容错选择器逻辑，或针对证据收集增加对异步渲染等待状态的捕获。

3. **防退化筛选与提升（Promotion）**：每一个变异出的候选技能必须在严格隔离的验证页面集 $\mathcal{D}_{\mathrm{eval}}$ 上接受检验，其提升准则极为严苛：




{% raw %}$$ f_{\mathrm{promote}}(k\mid V,\mathcal{D}_{\mathrm{eval}})= [\mathrm{obj}(V\oplus k)>\mathrm{obj}(V)] \cdot[\mathrm{regress}(V\oplus k)=0]=1 $${% endraw %}



即候选技能不仅要显著提升综合评测指标，还必须保证对已有表现的退化为零（Zero Regression）。

4. **图谱路由与冻结（SkillGraph Routing）**：通过审核的技能被纳入带有依赖、组合与冲突关系的技能图谱 $\mathcal{K}$ 中。针对不同的输入需求，动态路由器 $\rho(q,x,\mathcal{K})$ 仅激活对应的技能子图。

完成这一进化过程后，整个评测流水线与技能图谱被彻底**冻结（Frozen）**。这一点至关重要：在后续进入策略模型的强化学习阶段时，奖励标尺绝不发生动态漂移，从而为策略梯度的稳定下降奠定了基石。

### 接入强化学习：GRPO 驱动的功能性飞跃

在下游强化学习阶段，研究团队采用基于规则奖励的群相对策略优化（GRPO）算法。冻结后的 WebGrader 直接承担了无须 Critic 模型的奖励发放职责。

对于策略模型生成的每一个候选网站，WebGrader 都会在无头浏览器中跑一遍对应的流程契约。功能性得分 $I(q,x)$ 依据所有有效判决（Pass/Fail）的加权通过率计算，再与 GPT-5.4-mini 给出的外观视觉分 $A(x)$ 进行线性融合：




{% raw %}$$ R(q,x) = 0.7 \times 5I(q,x) + 0.3A(x) $${% endraw %}



可以看到，奖励权重的 70% 牢牢锚定在可验证的功能交互上。

在实验设置上，研究团队以 Qwen3-8B 为基础底座，先经过 600 条高质量合成样本的单轮 SFT（生成 React/Vite 代码规范），随后在独立的 600 条无重叠开放式需求上展开 100 步有效的 VERL/GRPO 迭代。评测基准则完全隔离，选用公开的 WebGen-Bench（101 个真实需求，647 个细粒度功能测试点）以及全新设计的独立评测基准 WG-core-250。

从实验对比来看，WebGrader-RL 展现了碾压同参数量级基线的强悍实力：


| 模型方案 | 参数量 / 架构 | WebGen-Bench FSR (%) | AAS 外观分 | WG-core-250 Full Score |
| :--- | :---: | :---: | :---: | :---: |
| Qwen3-Coder-480B | 480B | 47.91 | 3.51 | 42.10 |
| o4-mini | Closed | 49.38 | 3.48 | 41.87 |
| DeepSeek-v4-flash | Closed | 48.76 | 3.45 | 40.52 |
| VLM+Base-Script-RL (基线) | 8B | 44.13 | 3.50 | 38.37 |
| **WebGrader-RL (本文方案)** | **8B** | **52.01** | **3.52** | **44.95** |

在 WebGen-Bench 上，WebGrader-RL 相比采用未进化基础测试脚本的同底座模型（VLM+Base-Script-RL），功能成功率纯涨 7.88 个百分点，而外观分（AAS）基本维持在 3.5 左右。这强有力地说明：性能的跃升完全来自逻辑实现与功能交互的本质改善，而非靠迎合视觉审美刷分。更为惊人的是，在独立的 WG-core-250 测试集上，这个 8B 参数的开源模型以 44.953 的满分率，全面压制了参数规模数十倍于己的行业标杆，包括 o4-mini 以及高达 4800 亿参数的 Qwen3-Coder-480B。

### 深度剖析：为什么结构化路由远胜长规则 Prompt？

在技术分析部分，作者团队给出了一系列富有启发性的消融实验，深入解答了为什么“把测试逻辑演化成图谱”能够带来如此大的增益。

首先，针对业内常见的“把错误案例总结成规则全塞进 Prompt”的做法，团队构建了扁平规则对照组（Flat-rule Control）。实验显示，如果直接把进化过程中挖掘出的所有规则平铺到一个巨大的 Prompt 提示词中，Token 消耗量从 99.5K 激增至 128.5K，但评测器的 Macro-F1 仅仅从基准的 0.7813 微增至 0.7896；而一旦采用结构化的 SkillGraph 动态路由，在几乎相当的 Token 预算下，F1 分数直接跃升至 0.9037；若进一步加入冲突检测与处理机制，F1 更是达到了 0.9248，评测模糊度（Inconclusive）被压低至 4%。

这一现象揭示了 LLM 作为 Judge 时的核心局限：过多无差别的测试约束会导致注意力涣散与指令冲突。通过动态路由器，系统平均每例仅激活 1.46 个最为相关的技能模块，激活精确率达到 84.2%，召回率达到 88.6%。精准匹配、就事论事的微型技能编排，远比盲目扩充的“全能长提示词”可靠得多。

其次，针对技能模块的剔除测试（Leave-one-skill-out）显示，收益最大的技能依次是：证据断言检查点（Net Correctness +7）、目标动作精准对齐（+6）以及可执行测试夹具（+6）。在细分故障切片上，进化后的 WebGrader 在深层潜伏故障上的召回率提升了 20.6 个百分点，在有状态交互故障上提升了 21.3 个百分点，在 2 到 3 步深度的交互链路中提升了 25.0 个百分点。

这意味着，WebGrader 成功攻克了过去评测器最容易摔跤的“暗坑”：页面有局部刷新、涉及隐式状态传递、或者依赖复杂交互上下文的代码，现在只要有一处没有按预期流转，就会被抓现行并转化为向后传播的 RL 惩罚信号。

### 交互型智能体的奖励构建范式

WebGrader 的工作对整个 Agent 强化学习领域具有深远的启发意义。当前社区在大模型代码生成上的注意力，正从单函数补全（如 HumanEval）快速转向复杂系统与工程级应用构建（如 SWE-bench 与各类 WebGen 任务）。但后者的核心难点恰恰在于：任务的成功与否，无法通过简单的静态字符串匹配或一次性的断言来裁决，它的正确性隐藏在多步动态交互的因果链条之中。

WebGrader 证明了一条极具前景的路线：在开展面向复杂交互环境的模型强化学习之前，必须先将“奖励生成器”本身作为一个一等公民（First-class Citizen）进行严格的量化验证与离线进化。通过将交互流契约化、在真实浏览器引擎中执行求证、并借助 NAS 理念修剪评测盲区，我们能够以相对较小的代价构建出一个高鲁棒、防欺骗的客观环境裁判。正是得益于这种高质量、不可作弊的反馈信号，原本体量精简的 8B 模型才得以在精准的梯度指引下迅速收敛，在实际工程代码开发中爆发出超越数百亿、数千亿巨型参数模型的惊人功能执行力。
