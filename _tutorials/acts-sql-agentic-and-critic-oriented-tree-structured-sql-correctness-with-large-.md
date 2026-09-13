---
layout: default
title: "字节跳动ACTS-SQL：打破单向修Bug困局，树状回溯让SQL纠错提升9.42%"
description: "针对这一系统性痛点，字节跳动与中国人民大学联合提出了 ACTS-SQL （Agentic and Critic-Oriented Tree-Structured SQL Correctness）。该工作完全不需要额外微调，而是将 SQL 的除错与纠正重构为一套“由规划引导、支持显式分支与回溯”的树状推理架构。"
arxiv_id: "2608.15145"
paper_published: "2026-08-15"
published_at: "2026-09-13T13:15:08.876178+08:00"
topics:
  - "AI Agent"
tags:
  - "ACTS-SQL"
  - "BIRD-Critic"
  - "Backtracking multi-strategy"
  - "Clause-level diagnostics"
  - "Execution-based verification"
  - "Plan-guided correction"
related_tutorials:
  - "skiller-language-level-reinforcement-learning-for-reusable-skill-extraction-in-s"
  - "tree-search-for-llm-agent-reinforcement-learning"
  - "mechgeo-autoformalizing-and-proving-euclidean-geometry-in-lean-4"
  - "learning-when-to-plan-efficiently-allocating-test-time-compute-for-llm-agents"
---

<p class="paper-original-title" lang="en">ACTS-SQL: Agentic and Critic-Oriented Tree-Structured SQL Correctness with Large Language Models</p>

<img src="/images/2608.15145v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在大模型落地数据分析与业务查询的各类场景中，Text-to-SQL 一直被寄予厚望。然而，不论底层模型是顶级的闭源通用模型，还是经过针对性微调的领域模型，单次生成 SQL 的正确率始终难以达到工业化上线的严苛要求。哪怕是表现亮眼的基座模型，在面对复杂的跨表连接、多重嵌套逻辑与特定方言语法时，首轮生成的准确率往往也难尽如人意。

> ArXiv URL：https://arxiv.org/abs/2608.15145v1

为了弥补单次生成的不足，学术界和工业界普遍转向了基于 Agent 的自动纠错机制（SQL Correction）。然而，传统的自适应纠错流程大多遵循“线性试错”的逻辑：大模型生成一个 SQL，执行后报错或结果异常，再把错误信息塞回 Prompt 让模型修改，如此单线推进三到五轮。这种看似符合直觉的单链条迭代，在实际业务中暴露出致命缺陷：前期一步走错，后续的修改往往会像滚雪球一样越描越黑，产生严重的“语义漂移”。

针对这一系统性痛点，字节跳动与中国人民大学联合提出了 **ACTS-SQL**（Agentic and Critic-Oriented Tree-Structured SQL Correctness）。该工作完全不需要额外微调，而是将 SQL 的除错与纠正重构为一套“由规划引导、支持显式分支与回溯”的树状推理架构。借助专用的子句级拆解与歧义诊断工具，ACTS-SQL 在学术基准 BIRD-Critic 上相较此前最先进方案实现了 9.42% 的准确率提升；在火山引擎日志服务（Torch Log Service, TLS）的真实生产环境中，成功将线上复杂日志查询的执行准确率从 36.77% 提升至 53.61%。

### 为什么线性的单路径调试总是越修越偏？

理解 ACTS-SQL 的破局点，必须先看清线性修正范式的内在脆弱性。目前多数基于执行反馈的修正系统，本质上都是单线程的 ReAct 或 Reflection 循环。模型基于初始错误的 SQL 及其运行反馈生成 Revision 1，再基于 Revision 1 生成 Revision 2。这种结构隐含了一个致命假设：模型的后续修改能够保留并收敛在正确的业务意图之中。

但真实的数据库查询逻辑极度敏感。一个微小的 `WHERE` 条件收紧，或者一个连接类型的误判，都会对全局语义产生颠覆性破坏。本文给出了一个极为典型的失败案例：

<img src="/images/2608.15145v1/intro_case.jpg" alt="线性单路径纠错的失效模式" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在该案例中，用户的自然语言意图只是希望从形如“$M \times N$”的描述中提取商品数量与品类，并筛选出描述中包含“Electronics”关键词的记录。然而，在模型进行第一轮修正时，产生了一个隐式的错误前置假设——它误以为匹配“Electronics”必须是某种标准化的严格相等。一旦这个假设落地，单路径的 Agent 就会被死死困在这一狭窄的假设空间内，后续轮次所有的思考与努力，全都消耗在如何把字段转为小写、去除首尾空格、或者编写更加复杂的正则表达式来实现精准匹配上。

这直接暴露了线性除错的两个根本短板：

其一是**前期隐式假设的脆弱性**。大模型在面对具有歧义或未充分指定的自然语言时，会下意识地选择某一种解释并固化为代码。在线性管线中，后续修正无法“跳出模型先入为主的认知局限”，导致真正合理的语义空间（如使用子串模糊包含）被彻底屏蔽。

其二是**误差累积的不可逆性**。SQL 具有极强的符号组合性。当模型不断叠加字符串处理、修改聚合过滤条件时，局部的修补会产生全局性的语义偏移。原本可能只是漏写了一个匹配函数，最后整个 SQL 结构却被改得面目全非，彻底偏离了用户的原始意图。线性流程缺乏“推倒重来、回到分叉口换个思路”的机制，导致纠错轨迹滑向不可控的混乱。

### 树状规划与回溯：ACTS-SQL的核心运行逻辑

既然线性单路径的病根在于“不可逆”，解决之道自然是引入可探索、可评估、可回退的分支机制。ACTS-SQL 将 SQL 纠错抽象为一个由规划（Plan）驱动的树状搜索过程。在这一体系中，修正过程不再是盲目的试错，而是围绕决策树展开的系统化勘验。

整个调试空间被组织为一棵动态扩展的树 $T$。树的每一个节点 $n$ 对应一次具体的工具调用或诊断动作，边则代表状态转移。当环境反馈表明当前推导路径陷入死胡同时，系统不仅可以终止该分支，更能精准执行回滚（Rollback），重新从上游决策点选择其他合理的语义分支继续演进。

<img src="/images/2608.15145v1/overview2.jpg" alt="ACTS-SQL 框架全景概览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整体流程由顶层的中央大脑 **Central Agent** 与一组专精的外部工具协同完成。系统接收到自然语言需求、目标 SQL 方言、数据库 Schema、问题 SQL 及其执行报错或返回内容后，并非直接着手写代码，而是首先调用规划逻辑生成初始树状修复蓝图。

Central Agent 依据当前节点执行后的结构化上下文，判断接下来是深入子节点、横向拓展新的候选分支，还是在遭遇逻辑违背时触发回退。整个循环以结构化方式展开：

```text

Algorithm: Tree-Structured SQL Debugging

1: T <- GenerateInitialPlan(UserQuery, FaultySQL, Schema)

2: n <- T.root

3: while not Terminated(T) do

4:     result <- Execute(n)

5:     if NeedsExpansion(result) then

6:         T <- ExpandTree(T, result)

7:     end if

8:     if EvaluationFailed(result) then

9:         n <- Rollback(T, n)

10:    else

11:        n <- SelectNextNode(T, result)

12:    end if

13: end while

```

这一设计的精妙之处在于，它让大模型的探索过程完全白盒化与结构化。Central Agent 始终维护一个包含了全局规划 JSON、已执行节点状态矩阵及上下文游标的完整状态视图。大模型每一次评估不仅关注当前的单步成败，还能站在整棵推导树的全局视角，清醒地判断哪些假设已经被真实数据否决，哪些潜在路径依然值得尝试。

### 专精工具箱：从语义歧义到子句级语法解耦

单有树状结构，如果缺乏精准的探针与干预手段，树的分支与剪枝依然会沦为空谈。大模型在调试 SQL 时，往往在“语义意图理解”和“底层方言语法”两个不同层面上交织受阻。ACTS-SQL 据此设计了五项互补的专用工具，为树的生长提供高质量的决策依据。

第一个关键工具是**歧义检测工具（Detect Ambiguities）**。这是实现主动分支（Branching）的源头。面对用户提问中模糊或省略的业务术语，该工具会结合当前数据库的 Schema，显式挖掘出 2 到 3 个合理的业务解释，并将它们分别实例化为树结构上的平行分支节点。例如，当用户要求筛选“重叠（overlapping）”的数据时，工具会直接枚举出两种完全不同的技术口径：一种是统计层面的重复预订总数，另一种是时间戳区间上的真实重叠。将隐式的歧义显式拆解为并行的子节点，彻底封死了“第一步猜错就全盘皆输”的漏洞。

第二个支撑体系是**数据库探针工具集**，包含执行工具（Run SQL）与列格式检查工具（Inspect Column Format）。单纯让 LLM 进行“脑内静态代码走查”极易产生幻觉，许多连接失败或空值过滤问题本质上由真实数据的分布决定。探针工具允许模型在不书写庞大复杂查询的前提下，通过简单的固定模板快速抽样指定表、列的真实值分布。这些带外数据为剪枝（Pruning）提供了不可辩驳的物理证据：一旦探针发现某列数据格式与某种语义分支的假设不符，该分支就会被立即截断，触发回溯。

<img src="/images/2608.15145v1/workflow_case.jpg" alt="树状 SQL 纠错的真实推导案例" style="width:min(600px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; display:block;">

第三个极具工程价值的工具是**子句拆解与语法修复工具（Split and Fix Syntax Error SQL）**。当分支终于锁定了正确的语义方向，生成的候选 SQL 却往往因为各家数据库复杂的方言差异（如 Presto、Trino、ClickHouse、MySQL 的细微语法差别）导致执行失败。传统的做法是让模型重写整段 SQL，但这种全局重新生成极易顺带改坏此前已经推导正确的业务逻辑。

ACTS-SQL 采用了一种扁平化关系代数分解策略：将一段包含嵌套、联结、过滤的复杂 SQL 拆解成一组互不嵌套、极简的扁平单表 `SELECT` 操作单元，分别对应投影、过滤、关联和聚合。系统对这些原子子句分别进行轻量级校验，把具体的语法违背精准锁定在最小颗粒度的子句范围之内。这种做法实现了“局部语法定向定点切除”，在快速恢复 SQL 可执行性的同时，最大限度冻结了已经通过校验的全局语义结构。

最终，当所有中间凭证闭环，**候选生成工具（Generate SQL）**在叶子节点完成最终代码装配。该工具被赋予了严苛的“最小修改原则”，强制模型必须严格基于从根节点到当前叶子节点累积的探针证据进行修补，严禁毫无根据的自由发挥。

### 评测与工业落地：突破微调上限与生产实战

为了验证纯 Agent 式的树状纠错究竟能达到怎样的高度，研究团队首先在学术界极具公信力的大规模基准 **BIRD-Critic** 上进行了严密评估。该基准不仅涵盖多种复杂数据库方言，更设计了覆盖语法、Schema 映射、多表逻辑以及隐式过滤等多元梯度的错误样本。

评测结果给出了一个极具颠覆性的结论：完全无需针对特定领域数据进行昂贵微调的 ACTS-SQL，在基准上的综合纠错表现直接刷新了 SOTA，相比此前依靠海量高质量标注数据监督微调（SFT）训练出的专用最强基线模型，准确率逆势提升了 9.42 个百分点。

这一结果不仅印证了搜索机制在复杂符号推理上的威力，更直击工业界落地微调方案的软肋：

在现实数据基础设施中，日志与数仓的方言繁复多样，且业务语义瞬息万变。基于特定方言微调的模型在跨方言或遇到新业务 Schema 时，泛化能力往往断崖式下跌。例如，研究团队观察到，在 MySQL 等常规方言上精调的 Xiyan 模型，直接迁移到火山引擎自定义日志查询环境时，执行准确率仅剩 13.40%，甚至远不如未经微调的通用顶尖基座（36.77%）。这确立了一个技术判断：面对异构且动态演进的数据库生态，**基于通用强基座的“测试期计算扩展（Test-Time Compute）+ 结构化决策树”路线，在泛化性与维护成本上显著优于沉重的模型微调路线**。

除了基准测试，ACTS-SQL 更直接接入了字节跳动火山引擎的日志分析服务（Torch Log Service, TLS），赋能线上真实用户的 Text-to-TLS 问答管线。日志分析场景对 SQL 纠错的要求近乎苛刻：用户的问题往往充满口语化省略，字段定义包含海量半结构化 JSON，且查询语言属于深度定制的日志分析方言。

在生产系统以严苛延迟与成本控制为前提的线上盲测中，当搭载代表性的顶尖强基座（GPT-5 级别）时，ACTS-SQL 成功将真实复杂用户请求的执行准确率从原先单次生成的 **36.77% 跃升至 53.61%**，净提升达 16.84 个百分点。在切换至其他通用基座模型时，系统均展现出了高度一致的纠错增益。这证明 ACTS-SQL 的机制优势并不依附于某一家闭源模型的特殊特性，而是系统级拓扑结构带来的通用推理红利。

### 对复杂代码与 Agent 系统的后续启示

从工程落地与系统设计的视角审视，ACTS-SQL 的实践为大模型时代的自动化编程和复杂 Agent 架构设计带来了几点清晰的启示：

其一，**单向链条式 Agent 的范式天花板已现**。无论对 Prompt 做多少次自省提示，线性的状态演进在长链路推理中都无法摆脱“误差单调累积”的马尔可夫困局。引入显式的树状分支与状态回溯，是赋予系统自愈能力的必经之路。

其二，**软性语义推理必须与刚性符号探针强制解耦**。让大模型既充当业务理解者，又在脑中虚拟解析器运行代码，往往两头不讨好。将语法定点修复降解为原子关系代数片段，将 Schema 理解绑定到列级别的真实探针采样，把模型的上下文牢牢锚定在确定性的物理反馈上，才能真正根除幻觉。

在迈向更高难度的企业级复杂数据交互时，大模型需要的往往不是更加庞大的参数，而是一套允许其从容试错、体面撤退并系统求证的推理骨架。ACTS-SQL 在真实工业日志系统中的扎实落地，为此类结合测试期搜索与领域工具链的设计范式提供了一份极具参考价值的答卷。
