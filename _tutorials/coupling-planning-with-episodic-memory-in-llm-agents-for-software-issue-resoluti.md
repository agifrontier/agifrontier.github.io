---
layout: default
title: "PMCoder：规划与情境记忆双向耦合，SWE-bench Verified多解25题"
description: "来自范德堡大学（Vanderbilt University）的研究团队提出了名为 PMCoder 的软件修复智能体，核心机制在于将分层阶段规划器（Hierarchical Phase Planner）与情境记忆（Episodic Memory）进行深度双向耦合。"
arxiv_id: "2608.06811"
paper_published: "2026-08-07"
published_at: "2026-09-09T13:15:08.606362+08:00"
topics:
  - "知识系统"
  - "AI Agent"
tags:
  - "LLM agents"
  - "PMCoder"
  - "SWE-bench Verified"
  - "episodic memory"
  - "hierarchical phase planner"
  - "issue reproduction verdicts"
related_tutorials:
  - "socratic-swe-self-evolving-coding-agents-via-trace-derived-agent-skills"
  - "memrl-self-evolving-agents-via-runtime-reinforcement-learning-on-episodic-memory"
  - "treewriter-ai-assisted-hierarchical-planning-and-writing-for-long-form-documents"
  - "evomal-self-poisoning-in-self-evolving-coding-agents"
---

<p class="paper-original-title" lang="en">Coupling Planning with Episodic Memory in LLM Agents for Software Issue Resolution</p>

在真实的软件代码仓库中自动修复一个 Bug，大语言模型（LLM）Agent 面临的往往是一场持久战。一次修复任务可能包含数十乃至上百个交互步骤，跨越代码探索、假设定位、补丁编写与测试验证等多个阶段。在如此漫长的执行轨迹中，阻碍智能体成功的往往并非底层模型缺乏代码生成能力，而是其内部状态管理能力的溃败：遗忘早期定位的关键线索、陷入重复报错的编辑循环、上下文窗口被冗余日志挤爆，或者在没有任何真实运行证据的情况下“自言自语”宣称问题已解决并提前退出。

> ArXiv URL：https://arxiv.org/abs/2608.06811v1

来自范德堡大学（Vanderbilt University）的研究团队提出了名为 PMCoder 的软件修复智能体，核心机制在于将分层阶段规划器（Hierarchical Phase Planner）与情境记忆（Episodic Memory）进行深度双向耦合。它摒弃了以往将规划和记忆模块割裂设计的传统思路，让当前规划阶段指导记忆的检索策略，同时利用记忆中沉淀的历史操作统计数据驱动卡滞检测与回溯重规划。此外，系统通过离线生成的轻量级 Bug 复现脚本引入执行接地（Execution Grounding），彻底斩断模型单凭文本声称修复完毕的盲目自信。

在业界公认严苛的 SWE-bench Verified 基准评测中，PMCoder 在相同基准环境下相比强基线模型平均多解决了 25 个真实 GitHub 缺陷，解决率绝对值提升 5.0 个百分点（+5.0pp）。更为关键的是，即便在没有复现脚本触发的“无武装”状态下，这种规划与记忆的双向协同依然带来了显著收益；而在 DeepSeek-V4-Flash、Claude Haiku 4.5 以及 OpenHands 框架上的跨系统迁移实验，进一步验证了该状态管理架构的通用韧性。

### 为什么长轨迹修复总会失控？

当前处理仓库级软件工程任务的 Agent 普遍采用“观察-行动”的循环模式（Observe-Act Loop）。智能体阅读缺陷报告与代码，输出 Shell 命令，读取标准输出与返回码，直到最终提交 Git 补丁。然而，在这条长达上百步的链路中，智能体能依靠的内部演进状态十分薄弱。

最根本的症结在于长程修复任务中的“成功信号缺失问题”（Success-Signal Problem）。在真实的调试过程中，官方测试集是被隔离的，模型只能依赖它执行命令后观察到的反馈。但这种反馈充满欺骗性：首先，Shell 返回码极其粗糙，执行 `ls`、`grep`、一段无效的探针输出与成功的测试脚本，返回码可能都是 0；其次，如果让模型自己写测试来验证自己的改动，模型往往会写出刚好契合其错误补丁行为的断言，而非符合真实缺陷规范的断言；最后，大模型在长上下文中极易产生确认偏差，在输出文本中信誓旦旦地声明“修复已完成验证”，这只是一句空洞的修辞，并非客观事实。

<img src="/images/2608.06811v1/pmcoder_architecture.webp" alt="PMCoder总体架构图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

如果智能体缺乏严密的结构化状态，它很快就会迷失。以往的研究往往孤立地增强某个环节：要么引入显式的 To-do 列表规划，要么引入历史经验的 RAG 记忆库。但孤立的规划器不知道历史命令执行了几次、同一个文件被修改了多少遍，极易制定脱离实际的虚高计划；而孤立的记忆库不知道当前处于“大范围搜寻线索”还是“精细化对准行号修改”，只能盲目地根据文本相似度召回一堆过时信息。

PMCoder 的切入点正是消除这种脱节。如架构图所示，它在观察-行动循环中植入了一套共享的双向控制流，将整个轨迹的行为证据锚定在客观事实之上。

### 分层阶段规划与情境记忆的双向联动

PMCoder 的核心系统设计建立在两个相互咬合的齿轮上：一个是基于确定性状态机的分层阶段规划器，另一个是带有图结构感知的带预算情境记忆。

规划器把一个复杂的修复任务约束在四个严格递进的阶段中：探索（Exploration）、假设（Hypothesis）、实现（Implementation）与验证（Verification）。这四个阶段具有标量排序，但系统允许出现带有回溯性质的事件转移。值得强调的是，整个生命周期中，规划器仅在任务启动之初调用一次 LLM，用于将 Issue 描述分解为带有阶段标签的子任务列表。此后的每一步转移、状态更新、卡滞检测与回溯，全部由无模型开销的确定性规则引擎执行。

<img src="/images/2608.06811v1/planner_statemachine.webp" alt="规划器状态机转移逻辑" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

每执行一步 Shell 命令，规则检测器会根据执行的命令类型及伴随的解释文本判断当前意图。例如，文件巡检与搜索归入探索，运行诊断打印或根因排查归入假设，文件修改归入实现，运行测试归入验证。为了抵御检测器偶尔的误判，状态机采用了滞后效应（Hysteresis）机制：前向推进只需单次明确证据，但想要横向滑动或后退回溯，则必须积累多次持续的动作证据。

与此同时，情境记忆维护着一个事件图。每个节点不仅记录对话内容，更抽取了丰富的结构化元数据：执行的角色、时间戳、内容压缩摘要、前置命令是否发生了文件变更、具体触碰了哪些代码文件。这些关于文件修改的元数据直接从系统实际执行的命令中捕获，杜绝了模型胡言乱语对记忆的污染。在进行记忆检索时，PMCoder 并不依赖黑盒的 Embedding 向量，而是采用最大边际相关性（MMR）结合带有代码结构图的混合打分机制：




{% raw %}$$ g(v)\;=\;\lambda\cdot\mathit{rel}(v)\;-\;(1-\lambda)\cdot\max_{u\in S}\mathit{sim}(v,u) $${% endraw %}



其中相关性评分 $\mathit{rel}(v)$ 融合了词汇特征与代码拓扑特征：




{% raw %}$$ \mathit{rel}(v)=w_{c}\cdot\mathit{lex}(v)+w_{g}\cdot\mathit{graph}(v) $${% endraw %}



$w_{c}$ 与 $w_{g}$ 分别代表词汇权重和代码图权重。$\mathit{lex}$ 针对当前子任务锚点计算逆文档频率（IDF）加权得分，而 $\mathit{graph}$ 则利用 AST 语法树解析 Python 文件的导入关系（Import Edges）与同文件共现关系。这意味着，记忆召回不仅仅看“文字像不像”，更看“两个文件在工程调用链路中是否紧邻”。

这两者之间的双向耦合体现在四个具象通道上：

1. **阶段决定检索（Phase $\rightarrow$ Retrieval）**：规划器所处的阶段实时调节记忆控制器的检索预算、多样性惩罚系数 $\lambda$ 与图结构权重。探索阶段给予大预算、高多样性，帮助模型广泛铺开视野；实现阶段则收紧为小预算、高图结构权重，死死锚定待编辑文件及其直接依赖。

2. **子任务驱动锚点（Sub-task $\rightarrow$ Retrieval）**：当前活跃子任务的关键词被直接作为检索锚点，彻底打破了仅看最近对话的时间偏差（Recency Bias）。

3. **记忆统计反哺规划（Memory Statistics $\rightarrow$ Plan）**：情境记忆持续追踪聚合统计值，包括单一文件编辑次数、只读命令是否饱和、标准化指令是否连续重复。一旦触碰阈值，记忆子系统立即向规划器举起红旗，触发回溯（Backtrack），打断模型的无脑循环，强制将“回滚代码并重新审视”（revert-then-refix）的恢复子任务压入栈顶。

4. **判据纠偏验证（Verdict $\rightarrow$ Plan）**：将下文详述的复现脚本执行判定嵌入规划状态，未通过复现测试前，验证阶段绝不允许标记为完成。

### 零扰动的上下文注入与执行落地

即便规划器与记忆计算出了极其精妙的上下文状态，工程上还有一个致命难题：如何把这些信息喂给大模型？

以往很多工作选择“重写历史”（History Rewriting），例如将早期的观察结果压缩替换掉，或者把长对话重排。但在主流的工具调用（Tool-Calling）模式下，API 协议严格要求每一次助手发起的 `tool_call` 必须紧跟着环境返回的对应 `tool_result`。强行打乱顺序或篡改消息角色，极易使 API 抛出异常，或者让模型陷入分布外（Out-of-Distribution）的困惑状态。

<img src="/images/2608.06811v1/injection_channel.webp" alt="工具调用结果追加注入通道" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

PMCoder 采取了一种非常优雅且符合分布的工程设计：它完全不动历史对话列表，而是将格式化的规划-记忆控制块直接作为一个特殊标记界定的尾缀，追加在最新一条由环境返回的 `tool_result` 之后。对于底层模型而言，它感知到的只是一次正常的命令执行返回结果附带了系统的诊断状态，这完全处于它的预训练与对齐分布之内。此外，注入通道配备了差分触发与新颖性过滤：仅在规划状态发生转移时激活，且自动过滤掉已经暴露过的节点，转而输出一行简明扼要的高密度建议（例如某个子任务失败的单行原因或文件过度修改警告）。

在推进验证阶段时，PMCoder 引入了执行落地（Execution Grounding）。在面对具体缺陷时，系统利用模型在离线状态下仅基于 Issue 文本提取出独立的 Bug 复现脚本（Bash 或 Python），并严格校验该脚本在未经修改的代码库上必定报错。在运行时，这个复现判定（Verdict）充当了规划器的校准门禁。模型声称自己搞定了没有用，只有复现脚本在打了补丁的环境中执行通过，验证子任务才被许可闭环。即便复现脚本因为超时或环境问题不可用，系统也不会崩溃，而是平稳降级到纯粹的规划-记忆基座运行。

### Django-13516 深度复盘：一次起死回生的修复追踪

为了具象化这套机制如何在代码战争中拯救一个缺陷，论文细致解剖了针对 Django 仓库第 13516 号 Issue 的修复对比。

<img src="/images/2608.06811v1/case_django13516.webp" alt="Django 13516 修复轨迹对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

该 Bug 的机理非常微妙：Django 的管理命令输出包装类 `OutputWrapper` 继承自 `io.TextIOBase`，它的设计初衷是将缺失的属性委托给被包装的底层输出流。然而，Python 标准库中的 `TextIOBase` 本身就自带了一个空操作（No-op）的 `flush()` 方法。这就导致当外部调用 `flush()` 时，由于父类已有该方法，调用永远不会被委托给底层流，使得数据库迁移（`migrate`）过程中的输出信息一直卡在缓冲区中无法即时打印。真正的金牌修复（Gold Fix）是在 `base.py` 中为 `OutputWrapper` 显式重写 `flush()` 并透传给底层流。

在测试中，基线 Agent 和 PMCoder 在早期都正确识别出了这一根因。然而在随后的长轨迹交互中，基准 Agent 产生了典型的“认知漂移”，它在上下文堆叠后忽然脑补出一个虚假约束，认为“不能修改 Django 的核心类”，进而转去修改完全无关的 `migrate.py`，最终提交了一个毫无用处的补丁。

PMCoder 在中间阶段同样遭遇了挫折：它在修改 `base.py` 时不慎破坏了文件结构。但在此刻，双向状态管理机制立刻发挥了挽狂澜于既倒的作用。记忆系统检测到 `base.py` 的反复修改异常与语法错误，向规划器报送了编辑异常统计；规划器立刻判定卡滞，触发回溯机制，生成了 `revert-then-refix` 信号，调用 Git 命令将文件还原到干净的初始状态。更为精妙的是，在接下来的上下文注入中，基于子任务锚点的情境记忆成功将几十步之前模型自己做出的“`OutputWrapper` 缺乏透传”的诊断结论重新打捞回最新提示词中。

重新看清地图的 PMCoder 没有重蹈覆辙，它精准地在 `base.py` 处添加了仅有几行代码的正确代理函数，并在该用例的三次重复评测中成功解决了两次；而未配备该机制的基准 Agent 在三次测试中全部饮恨。

### 实验评测：不仅是得分，更是轨迹行为的质变

实验首先在软件工程智能体领域金标测试集 SWE-bench Verified（500个经过人工严格校验的真实 Issue）上展开。底模采用以代码能力著称的开源模型 Qwen3-Coder，并通过 vLLM 部署。为了消除 FP8 量化服务带来的非确定性噪音，核心实验均严格执行了三次全量评测取平均。

从宏观解题量来看，基线系统在 500 个问题中平均解决 148.7 个（解决率 29.7%），而搭载了完整机制的 PMCoder 平均解决数跃升至 173.7 个（解决率 34.7%）。平均多解出的 25.0 个缺陷带来了整整 5.0 个百分点的绝对增益，在配对统计检验中表现出高度的显著性。难度分层分析显示，PMCoder 的收益主要集中在基座模型本身具有一定认知理解能力的中低难度区间，这印证了作者的判断：该系统并不是通过魔法提升基座模型的数理逻辑上限，而是通过精细的状态管理，把模型“本来有能力解出、却在长途奔袭中走丢了”的那部分成功率切实收割了回来。

更具洞察力的指标体现在智能体底层行为模式（Behavioral Signatures）的剧烈变化上：

* **失败后重复下达相同命令的比例**：基线 Agent 经常陷入在同一处报错上反复重试同一条指令的僵局，而 PMCoder 将这一指标降低了 40% 以上（比值仅为 0.58 倍）。

* **空补丁或放弃退出率**：很多长任务模型因失去方向最终直接输出空的 Diff 或提早投降，PMCoder 将此类绝望退出的概率压缩到了基线的 0.38 倍。

* **上下文窗口耗尽率**：超长上下文字符溢出是 Agent 崩溃的常见原因，PMCoder 的上下文耗尽频率下降为基线的 0.60 倍。

* **干净回滚再修复（Revert-then-refix）次数**：基线 Agent 极少主动撤销自己的错误提交，而 PMCoder 在三次运行中累计执行了 62 次成功回退并重修，展现出了极高韧性的自我纠错意愿。

为了验证各个组件的独立贡献，论文进行了严格的 $2\times 2$ 因子消融实验。在相同测试条件下，单加规划器带来了 +6.3 个问题的微弱提升（+1.3pp），单加记忆机制带来了 +8.3 个问题的提升（+1.7pp）。然而，当规划与记忆双向耦合时，总收益直接跳升至 +25.0 个（+5.0pp）。这一统计学上的正向交互作用有力证明了“双向联动”并非简单堆砌功能，规划失去了记忆的统计依据便成了盲人摸象，记忆失去了规划的阶段约束便成了大海捞针，两者结合产生了 $1+1 > 2$ 的突变。

在拓展验证中，系统展示了强大的泛化边界。当基座模型切换至 DeepSeek-V4-Flash 时，解决率提升了 3.2 个百分点（多解 16 题）；换用商业闭源模型 Claude Haiku 4.5 时，同样提升了 2.8 个百分点（多解 14 题）。进一步地，将 PMCoder 的整套双向基座无缝移植到当下最流行的开源框架 OpenHands 中，智能体在 Verified-500 上依然实现了 +4.6 个百分点（多解 23 题）的惊人增益。甚至在完全脱离 Bug 修复场景的通用命令行交互基准 TerminalWorld（20 个官方人工验证任务）上，PMCoder 也将成功任务数从 5 个推高至 7 个。

### 工程反思与长程智能体的新范式

软件工程 Agent 正在从“拼模型直觉”步入“拼系统状态控制”的深水区。过去一年中，行业习惯于将希望寄托在更大的上下文窗口（从 32k 到 1M）或更暴力的模型参数规模上。然而 PMCoder 清楚地表明：在没有有效状态治理的情况下，上下文窗口越大，反而越容易成为模型存储过时推理和幻觉噪音的垃圾场。

PMCoder 在成本与效益上的平衡极具工程参考价值。它不仅没有引入昂贵的多智能体博弈（Multi-Agent Debate），也没有在每一步使用重型 LLM 去担任评审员。除了一次性生成的测试复现脚本和仅在第一步执行的子任务拆分外，整个规划推进、记忆检索与卡滞检测全由轻量级、确定性的规则和图算法完成，几乎没有增加推理延迟和 Token 成本。

这种“外挂轻量确定性骨架，约束长程非确定性模型”的范式，为开发更可靠的终端 Agent 提供了极其清晰的技术路线。智能体不需要时时刻刻扮演深谋远虑的哲学家，它只需要一个在恰当时刻提醒它“你已经改错三次了、先把代码回滚”、并在它迷失时把最初线索重新推到它眼前的坚实支架。
