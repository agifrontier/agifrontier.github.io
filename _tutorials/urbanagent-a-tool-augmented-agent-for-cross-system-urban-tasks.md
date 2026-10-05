---
layout: default
title: "UrbanAgent：打破跨系统服务孤岛，让城市智能体任务成功率达到71%"
description: "近期由多位研究者提出的 UrbanAgent ，试图用统一的工具增强 Agent 框架攻克这一瓶颈。它不仅能够调度跨系统的异构城市服务，还引入了“行动前主动澄清、依赖驱动执行、多源证据对齐校验”的完整自适应闭环。"
arxiv_id: "2608.03018"
paper_published: "2026-08-04"
published_at: "2026-10-05T13:15:07.965576+08:00"
topics:
  - "AI Agent"
  - "AI工程"
tags:
  - "Model Context Protocol"
  - "Urban-Eval"
  - "UrbanAgent"
  - "adaptive closed loop"
  - "cross-system urban tasks"
  - "dependency validity"
related_tutorials:
  - "self-evolving-neuro-symbolic-skills-for-tool-augmented-spatial-reasoning"
  - "recevolve-a-knowledge-driven-autonomous-agent-system-for-recommender-systems"
  - "auto-research-for-materials-auditable-ai-scientist-workflows-with-held-out-trans"
  - "synact-a-reasoning-acting-large-language-model-agent-for-adaptive-synthesis-opti"
seo_title: "UrbanAgent：打破跨系统服务孤岛，让城市智能体任务成功率达到71%"
---

<p class="paper-original-title" lang="en">UrbanAgent: A Tool-Augmented Agent for Cross-System Urban Tasks</p>

<img src="/images/2608.03018v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在高度数字化的现代城市中，居民获取公共与商业服务的门槛反而变得愈发繁杂。日常生活中一个看似简单的出行需求——比如“规划一条避开降雨路段的自驾路线，并顺路找一家人均 100 元以内且有无障碍通道的餐厅”——在现实操作中往往需要频繁在天气预警、地图导航、生活消费和政务服务等多个 App 之间来回切换。用户必须手动充当不同系统之间的数据“搬运工”和逻辑校验者，在高度割裂的接口与数据流之间反复比对、筛选甚至回退。

> ArXiv URL：https://arxiv.org/abs/2608.03018v1

现有的解决方案大多停留在各自的系统边界内。类似城市大脑的大型中枢平台通常只面向宏观治理或单一机构，无法承接居民个体的端到端细粒度需求；城市基础模型擅长时空预测与宏观推理，却不具备直接调用实时业务接口的能力；GUI 操作智能体或垂直领域 Agent 往往受限于单一应用界面或固定的垂直任务。近期由多位研究者提出的 **UrbanAgent**，试图用统一的工具增强 Agent 框架攻克这一瓶颈。它不仅能够调度跨系统的异构城市服务，还引入了“行动前主动澄清、依赖驱动执行、多源证据对齐校验”的完整自适应闭环。实验表明，在专门构建的跨系统评测基准 **UrbanEval** 上，UrbanAgent 将复杂城市任务的成功率提升至 71%，在可执行任务上相比最强基线实现了 10.5 个百分点的提升。

<img src="/images/2608.03018v1/intro.webp" alt="割裂的城市服务与UrbanAgent工作流对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么通用的工具调用机制在城市任务中频频碰壁？

将大型语言模型与外部工具接口结合已经成为构建自主智能体的主流范式。在软件工程、网页浏览或简单信息检索等场景中，通用 Agent 依靠 ReAct（Reasoning and Acting）或 Plan-and-Execute 等编排策略，已经能较好地拆解多步调用。然而，当这些通用框架被直接置于真实的城市空间任务中时，往往暴露出严重的脆弱性。

城市物理空间与多源商业生态之间存在极其复杂的隐性依赖。一个真实的城市需求通常具有高度组合性和长链路依赖特征。更棘手的是，这些请求在自然语言表述上往往是高度抽象的高层意图，缺乏直接可执行的结构化参数。例如，当用户发起“找一家最近的营业药房”的指令时，指令内部实际上省略了至关重要的基准空间位置。

通用工具调用 Agent 在面对这种欠约束输入时，极易产生“静默脑补”（Silent Assumption）。模型会擅自假设一个默认位置或使用自身预训练参数中残留的城市坐标，随后顺畅地调取高精度的地图 API、查询当地降水情况并生成详尽的驾车指引。从单步调用的技术指标来看，每一次 API 响应都状态正常、参数合规，但最终拼装出的整条工作流从原点开始就是完全错误的。这种“每一步都是对的，组合起来却毫无用处”的假性成功，是目前城市服务 Agent 最致命的隐患。

此外，城市 API 具有显著的空间异构性和不确定性。不同服务商在地理覆盖边界、数据时效性和数据精度上差异巨大。某些坐标地理编码服务可能只能覆盖国内特定省市，境外或跨行政区查询会返回空值或越界错误；有时即便是合法的接口返回，其内容在现实空间中也是无效的。如果智能体缺乏对工具链执行依赖的动态感知和针对性重试机制，整个工作流便会在下游链路中发生雪崩式的级联失败。

### UrbanAgent 的系统架构：四大支柱构建执行闭环

为了应对上述挑战，UrbanAgent 没有采用静态生成完整调用序列的僵化方式，而是建立了一套由认知澄清、推理执行核心、工具调用与接地、证据对齐综合四个组件构成的动态闭环。整个系统统一接入代码执行、常规 API 与模型上下文协议（Model Context Protocol，MCP），将异构服务抽象在标准工具空间中。

<img src="/images/2608.03018v1/framework.webp" alt="UrbanAgent框架总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### 认知与澄清（Cognition and Clarification）

在任何外部工具被激活之前，UrbanAgent 部署了专用的认知映射机制 $f_{\text{cog}}^{\pi}$，其作用是将非结构化的自然语言请求 $\mathcal{Q}$ 结构化解析为形式化任务元组：




{% raw %}$$ \mathcal{U}=\langle\mathcal{I},\mathcal{E},\mathcal{C}_{\text{hard}},\mathcal{C}_{\text{soft}},\mathcal{M}\rangle $${% endraw %}



在这个元组中，$\mathcal{I}$ 精确锚定任务的核心意图，$\mathcal{E}$ 记录后续调用所需的初始实体与参数，$\mathcal{C}_{\text{hard}}$ 代表硬性约束（如“步行距离在 1 公里以内”、“营业至午夜”），$\mathcal{C}_{\text{soft}}$ 记录排序偏好（如“人均消费优先从低到高”）。

最为关键的是集合 $\mathcal{M}$，它专门收集那些无法在不篡改用户初衷的前提下被合理推断的关键缺失变量（如起始地理位置、明确的出行时间等）。一旦判定 $\mathcal{M}\neq\varnothing$，UrbanAgent 会直接拦截后续的所有工具调用操作，立即向用户发起澄清反问，明确终止盲目的推断过程。这一设计从物理源头上阻断了“凭空捏造参数”所引发的链式错误。

#### 推理-执行核心（Reasoning–Execution Core）

对于参数齐备的结构化任务，系统进入逐步推理与执行循环。UrbanAgent 放弃了一次性预先规划所有动作图谱的策略，转而采用依赖驱动的渐进式策略 $\pi$。在第 $t$ 步时，智能体根据当前的结构化任务 $\mathcal{U}$、截止到前一步的累积观测历史 $\mathcal{O}_{<t}$ 以及工具注册中心 $\operatorname{Schema}(\mathcal{T})$，决定下一步的具体操作：




{% raw %}$$ a_{t} =\pi\!\left(\mathcal{U},\mathcal{O}_{<t}\mid\operatorname{Schema}(\mathcal{T})\right), \quad a_{t} \in\{(\tau_{k},p_{t}),\operatorname{Commit}(\hat{\mathcal{R}})\} $${% endraw %}



这种逐步执行策略的核心在于严格确保“下游调用必须以真实消费上游观测值为前提”。例如，只有当高德或谷歌地图服务真正返回并解析出目标兴趣点（POI）的经纬度浮点数后，该数值才会被注入到下一步的路线规划算法或天气网格查询接口中，避免在模型记忆中进行未经核验的虚构传参。

#### 工具调用与接地（Tool Invocation and Grounding）

在运行时层面，大语言模型的 Schema 约束仅能保证参数的格式合规，无法保证语义的有效性。UrbanAgent 在工具注册与调度层建立了严密的空间校验与容错接地机制。

每一次工具执行都会捕获底层返回的原始响应、状态码及延迟，存入观测记录 $o_t$。若底层服务抛出异常或参数语义错误，错误信息将被封装并反馈给推理核心以触发自适应参数修正或工具切换。如果同一工具连续遭遇 8 次不可恢复的失败，系统将强制告警并标记该子任务不可达。

针对城市时空数据的特性，系统内置了地理一致性检查（Geographic-Consistency Check）和空结果校验。如果逆地理编码接口返回的省市与用户要求的行政区划产生冲突，或者检索返回了结构完整但内容为空的对象，该观测都会被立即标记为失败并通知规划器重试。同时，调度策略中硬编码了服务商覆盖范围规则（如避免在海外坐标调用只支持国内的地理引擎），从而确保只有经过空间验证的有效观测数据才能被持久化到有效事实空间中。

#### 证据对齐综合（Evidence-Aligned Synthesis）

在执行核心结束动作并生成初步草稿 $\hat{\mathcal{R}}$ 后，UrbanAgent 并不会直接将其呈现给终端用户，而是通过证据对齐综合模块执行严格的后验过滤：




{% raw %}$$ \mathcal{R}=f_{\text{syn}}^{\pi}(\mathcal{U},\mathcal{O},\hat{\mathcal{R}}) $${% endraw %}



该模块由三道紧密相连的工序组成：

1. **清洗（Clean）**：剔除草稿中冗余暴露的内部系统调用逻辑、调试轨迹以及类似“请问还需要我帮您做什么”的非必要口语对话，使回答聚焦于决策方案本身；

2. **对齐（Align）**：在观测历史 $\mathcal{O}$ 中精确定位支撑力最强、粒度最细的物理证据，并将这些事实按照用户最初设定的格式进行组织；

3. **核验（Verify）**：将最终结论对照硬约束 $\mathcal{C}_{\text{hard}}$ 进行反向校对，严厉禁止在最终答复中出现任何未被有效观测 $\mathcal{O}$ 支撑的事实性断言。如果经过多轮检索依然无法获取某些特定细节，综合模块会被强制要求明确陈述信息的缺失，而不是用概率拟合的假象去误导用户。

### UrbanEval：打破仅凭“最终答案”评测的盲区

过去评估大模型工具调用能力的基准（如 ToolBench 等）多侧重于通用 Web 场景或虚拟代码环境；而城市计算领域的现有基准（如 CityGPT 评测等）又大多聚焦于静态的时空预测能力或知识库问答，缺乏对真实在线异构服务调用的度量。更核心的问题在于，在复杂的城市多跳任务中，很多智能体仅凭预训练知识就能拼接出一个看似煞有介事的推荐方案，掩盖其底层并未真实调用工具或跳过了依赖步骤的作弊行为。

为了真正衡量城市 Agent 的实际执行质感，作者团队构建了 **UrbanEval** 评测基准。该基准包含 250 个精心设计的城市真实复杂需求，涵盖跨城出行、复合生活规划、应急响应与事件调度等多种真实维度，划分为 5 大任务类别与 3 种难度级别。

<img src="/images/2608.03018v1/Figure3.webp" alt="UrbanEval基准总览与任务执行示例" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

为了杜绝“蒙对结果”的虚假评测，UrbanEval 引入了三层评测机制，将任务求解、执行合法性与系统开销清晰分离：

- **结果层（Result Layer）**：由大模型评判员根据各查询的硬性标准判定任务成功率（Task Success Rate, TSR），并专门针对意图模糊查询评估缺失输入检索率（Missing Input Retrieval, MIR）。

- **过程层（Process Layer）**：直接通过运行轨迹计算底层确定性指标。工具意图覆盖率（Tool Intent Coverage, TIC）检验必须调用的最小工具集合是否齐备；输入输出依赖准确率（IO-Dependency Accuracy, IOA）严密检测下游参数是否真实引用了上游输出；验证有效参数率（Valid Parameter Action Rate, VPAR）则评估每一次工具调用的语义参数合法性。

- **效率层（Efficiency Layer）**：如实统计各策略在完成任务时的交互轮次、耗时以及 Token 消耗量。

值得注意的是，由于真实城市服务的返回结果具有强时效性（例如实时天气会随时刻变化，餐厅的空闲席位动态变动），UrbanEval 在评分时没有采用僵化的静态标注答案，而是将每次评测系统实际获得的实时 API 返回日志作为动态真实基准（Ground Truth）。这种设计彻底消除了因外部环境波动对模型评分公允性造成的干扰。

### 实验对比：系统性增益从何而来？

在控制变量的对比实验中，UrbanAgent 与四类主流 Agent 编排机制（Native tool-calling、ReAct、Plan-and-Execute、AutoGen）以及十种无工具调用的裸大模型（Single LLMs）展开了系统性评测。所有工具增强基线均被置于完全一致的环境配置下：共享相同的 MCP 工具注册中心、相同的 JSON Schema 描述、相同的单次查询预算上限（最多 25 次 LLM 调用或 600 秒限时）。

实验结果揭示了显著的性能分化：

在基准测试的全部 250 个请求中，当采用 **GPT-5-mini** 作为基础底座时，UrbanAgent 取得了 **71%** 的端到端任务成功率（TSR），而业内表现最好的通用基线方案（Native tool-calling 与 AutoGen）仅达到 61%。更具说服力的数据体现在剔除了 50 个纯澄清意图任务的 200 个全流程可执行复杂任务中：UrbanAgent 成功完成了 128 个（成功率 **64.0%**），大幅超出 Native tool-calling 成功完成的 107 个（成功率 **53.5%**），实现了 10.5 个绝对百分点的性能代差。这一领先优势在更换基础模型（包括 Gemini-2.5-flash、DeepSeek-V4-flash 和 Qwen3-235B-A22B）后依然保持了高度的稳定性。

通过 UrbanEval 提供的过程层精细度量，研究团队得以剖析这一优势背后的深层原因，破除了一般性的技术直觉误区：

第一，在基础工具链的覆盖和依赖顺序上，各高级工具基线其实并没有拉开悬殊差距。各基线在工具意图覆盖率（TIC）上普遍达到了 0.86 到 0.89，而在输入输出依赖有效性（IOA）上更是均接近 1.00。这意味着在主流底层大模型已经具备扎实单步调用能力的当下，基线系统“知道该调哪个接口，也知道参数按顺序传”。

第二，在线澄清能力本身也已基本饱和。面对模糊任务时，各大基线的 MIR 均处在 0.90 到 0.98 的高位，表明模型大多能够识别出参数缺失。

真正的断层分化发生在此类临界情境的处理哲学上：基线方案在面对 50 个意图不完整的模糊查询时，虽然指出了参数缺失，但随后仍有 5 到 8 次“忍不住”基于臆想的假设参数继续调用真实 API，展开了无效的盲目操作；而 UrbanAgent 的认知拦截机制实现了 **0 次** 越权盲猜，严格将系统边界定格在确认动作之内。

另一个更为显著的差距，在于**长链路强约束任务的处理韧性**。在涉及多重条件嵌套（例如需要同时满足距离半径、营业时段、天气变化和多地途经次序）的复杂请求中，通用基线系统往往在某个中间节点调用返回空数据或报错时就草草收场，甚至直接依据残留的历史记忆在最终回答中敷衍应付；UrbanAgent 则依靠其内建的地理合法性拦截与多源证据对齐综合，能够自主进行多轮备选工具路径的探索与回退。

当然，这种高度的鲁棒性并非没有代价。系统开销数据表明，UrbanAgent 在单次成功任务中平均消耗约 118.3k Token，约为 Native tool-calling 和 ReAct 等基线系统（37.3k 至 42.8k）的三倍左右。这种计算资源的开销集中投入在多阶段的验证、反思和跨工具语义清洗上。在实际工程落地中，这体现了一种明确的系统取舍：在容错率极低的物理城市服务面前，耗费更多推理算力去换取一个在现实空间中真实可用、经得起时空验证的确定性方案，其业务价值远高于廉价却频繁出错的不可行建议。

### 从单点工具调用走向城市空间计算的真正闭环

从技术演进的视角审视，UrbanAgent 的突破不仅在于提供了一套高成功率的 Agent 实现架构，更在于它清晰界定了大模型在介入物理世界服务时所必须建立的防御体系。

当前的许多 Agent 研究容易陷入一种盲目乐观的误区，以为只要为语言模型配备足够丰富的 API 描述，模型就能自行涌现出复杂的协调逻辑。然而城市系统的复杂性在于，空间距离的非欧几里得特征、多源系统之间的数据口径冲突、以及自然语言高层意图与底层物理坐标之间的巨大语义鸿沟，都会将普通规划器的逻辑脆弱性成倍放大。

UrbanAgent 证明了在模型自身参数能力之外，通过系统级的工程范式重构——将意图解析时的认知约束、执行过程中的地理语义接地，以及输出阶段针对客观证据的强力对齐机制进行耦合——是构建高可用物理空间 Agent 的必经之路。伴随 Model Context Protocol（MCP）等工具标准化协议的普及，这种让 AI 真正“在严密约束下行事、在确凿证据上发言”的严谨架构，为未来自动驾驶调度、数字孪生城市运营以及个性化居民生活助理的可靠落地，提供了极具参考价值的工程蓝本。
