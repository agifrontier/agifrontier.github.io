---
layout: default
title: "OmegaUse-OfficeVal：大模型干Office长任务差在哪？百度给出经济学基准"
description: "OmegaUse-OfficeVal：整个任务集的构建经历了严格的工程与伦理闭环： 1. 真实需求采集与脱敏 ：研究团队从不同岗位的办公从业者处征集真实的业务委托与初始输入素材，模拟一名初级行政助理、财务实习生或数据分析员接到的工作。"
arxiv_id: "2607.27155"
paper_published: "2026-07-29"
published_at: "2026-09-26T13:15:07.411720+08:00"
topics:
  - "AI Agent"
tags:
  - "LLM agents"
  - "OmegaUse-OfficeVal"
  - "code-based verifiers"
  - "fine-grained rubrics"
  - "human labor time"
  - "long-horizon office-suite tasks"
related_tutorials:
  - "startupbench-benchmarking-general-purpose-agents-on-market-validated-end-to-end-"
  - "learning-on-the-job-an-experience-driven-self-evolving-agent-for-long-horizon-ta"
  - "repurposing-synthetic-data-for-fine-grained-search-agent-supervision"
  - "factscore-fine-grained-atomic-evaluation-of-factual-precision-in-long-form-text-"
seo_title: "OmegaUse-OfficeVal: Benchmarking LLM Agents on Long-Horizon Office-Suite Tasks with Economic Grounding"
---

<p class="paper-original-title" lang="en">OmegaUse-OfficeVal: Benchmarking LLM Agents on Long-Horizon Office-Suite Tasks with Economic Grounding</p>

<img src="/images/2607.27155v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从“Vibe Coding”（氛围写代码）走向“Vibe Working”（氛围办公），是大模型落地被寄予厚望的下一步。业界普遍预期，既然 LLM 智能体能够在复杂代码库里修 Bug、做重构，那么处理日常的文档、表格、幻灯片和 PDF 跨模态办公任务，理应水到渠成。

> ArXiv URL：https://arxiv.org/abs/2607.27155v1

然而现实往往骨感得多。不少用户在尝试让智能体处理一份结构复杂的几十页财报、合并多源多格式报表或制作排版严苛的汇报 PPT 时，经常遇到格式错乱、公式失效甚至关键信息被覆盖的窘境。更关键的问题在于：学术界与工业界长期缺乏一把客观、精准的标尺，来衡量智能体在真实高价值办公流中的交付质量，以及它究竟能为企业节省多少真实成本。

为了填补这一空白，来自百度（Baidu Inc.）的研究团队推出了 **OmegaUse-OfficeVal**。这是一个面向长周期（Long-Horizon）日常办公套件任务、且具备任务级“经济学标定（Economic Grounding）”的评测基准。该基准汇集了 100 个源于真实职场需求的复杂办公任务，平均需要人类初级员工耗费 2.32 小时才能完成。

研究全面评测了 GLM-5.2、Kimi K2.6、DeepSeek-V4-Pro、MiniMax M3 以及 Qwen3.7-Plus 等一众主流前沿大模型，并设定了严谨的人类基线。结果显示出一个尖锐的反差：**大模型在耗时和推理成本上已经比人类初级员工快数十倍、便宜数千倍，但在最终交付件的完成质量上，表现最好的模型得分仅为 17.91，远落后于人类基线的 27.79**。模型看似跑得飞快，但交出的“活儿”离真正可用依然存在明显鸿沟。

<img src="/images/2607.27155v1/time-score-cost-comparison.webp" alt="时间、得分与成本综合对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 为什么现有办公评测测不出真实生产力？

现有的智能体评测基准虽然种类繁多，但在评估“职场打工人日常办公”这一特定场景时，往往存在三类系统性断层：

第一，**泛生产力基准粒度过粗且开放度受限**。诸如 GDPVal、Remote Labor Index（RLI）和 Agents' Last Exam（ALE）等基准，虽然将视野投向高经济价值的人类职业工作流，但它们的经济价值估算通常挂钩在宏观行业或职业大类上，缺乏对具体微观任务的工时与价格拆解；同时，出于版权或防数据污染等考虑，这类基准大多只开源极小一部分任务或对标准输出做严格访问限制，研究者难以进行可复现的深度诊断。

第二，**传统办公自动化基准偏向短周期与机械操作**。如 OfficeBench、SpreadsheetBench 和 PPT-Eval 等基准，往往聚焦于单元格公式填写、幻灯片文本框生成等单一应用、固定步骤的简单操作。它们用动作步数或对话轮数来定义复杂度，缺乏真实人类在多应用间反复跳转、排查依赖、反复校对所耗费的长程劳动，更不具备经济学视角的价值锚定。

第三，**GUI 智能体基准偏重轨迹过程而非最终交付件**。诸如 OSWorld 2.0 等前沿操作系统基准，虽然也将任务时长拉长到数小时，但其核心判定逻辑仍然是“环境状态或交互动作是否达标”，即考察鼠标有没有点到指定按钮、窗口有没有处于预期状态。然而在真实职场中，雇主和协同者根本不在乎智能体是用 Python 脚本批处理、走 COM 接口还是通过视觉截屏模拟点击，唯一具有商业价值的是最终提交的文件是否格式完好、内容精准、直接可用。

<img src="/images/2607.27155v1/six_task_example.webp" alt="OmegaUse-OfficeVal 六个典型任务示例" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

针对这些痛点，OmegaUse-OfficeVal 确立了三条核心原则：**职场真实性**（任务源自从业者日常高频委托，指令形式自然，而非生硬的操作序列拆解）、**任务级经济学标定**（每一道题都具备独立的人类耗时和外包市场价格标签，支持按价值加权评估）以及**长周期挑战性**（摒弃几分钟能搞定的微操作，专注考验大模型在长程上下文与复杂依赖下的持久规划能力）。

### 双重经济信号与任务构建管线

为了让评估具备真正的现实指导意义，OmegaUse-OfficeVal 引入了两个关键的经济学信号：**人类劳动工时（Human Labor Time）**与**任务价格代理（Task Price Proxy）**。

<img src="/images/2607.27155v1/pipline.webp" alt="任务构建全流程管线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整个任务集的构建经历了严格的工程与伦理闭环：

1. **真实需求采集与脱敏**：研究团队从不同岗位的办公从业者处征集真实的业务委托与初始输入素材，模拟一名初级行政助理、财务实习生或数据分析员接到的工作。所有材料均经过专家筛选和深度的隐私脱敏重构，确保去除商业机密的同时，完全保留真实业务逻辑的复杂拓扑。

2. **人类劳动工时的精细采集**：团队招募了 20 名通过办公套件能力初测的专职标注员，在不使用任何 LLM 辅助的前提下手工完成任务。每个任务指派至少 2 名标注员（耗时差异较大时引入第 3 名），并设计了结合质量准入门槛的效率激励机制。最终将有效完成时间中最短的两次耗时取平均值，得到该任务的人类劳动工时。统计显示，任务的平均人类耗时长达 2.32 小时，展现出扎实的长周期特征。

3. **任务价格代理的稳健估值**：在数据采集中，约 20% 的任务拥有来自众包平台的真实历史外包成交对价，作为高置信度锚点；其余任务则由三位资深专家独立进行市场发包估值。为了消除极端主观偏差，团队设计了一套基于一致性的聚合算法：将三名专家的估价排序为 $\min \le \mathrm{mid} \le \max$，计算两组相邻差值 $A = \max - \mathrm{mid}$ 与 $B = \mathrm{mid} - \min$。若 $A/B \ge 2$，说明最高价明显偏离共识，直接剔除，取中位数与最低值的均值；反之若 $B/A \ge 2$，则剔除最低价；其余情况下则取三者算术平均。这种机制有效平抑了极端估价对微观经济学评估的扰动。

<img src="/images/2607.27155v1/task-stat.webp" alt="任务输入文件与输出成果分布统计" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

统计表明，OmegaUse-OfficeVal 的输入材料具有强烈的多模态与复合格式特征。任务不仅要求输入 PDF、Word（Docx）、Excel（Xlsx）和 PowerPoint（Pptx），还深度嵌套了图像与音视频文件；而输出目标则全面覆盖了高标准的办公复合文档。

### 告别“大模型当裁判”：确定性代码验证与可用性一票否决

长周期办公任务交付物的评测历来是个棘手难题。人工判卷成本极高且主观方差极大，难以支持大规模高频自动化基准测试；而目前流行的“LLM-as-judge”方案，在面对动辄几万字、上百张工作表、多层嵌套样式的复杂文档时，不仅存在严重的幻觉和位置偏置，更致命的是，随着基座模型的代际升级，评测裁判的打分尺度会发生不可控的漂移，让历史对比失去基准意义。

为此，OmegaUse-OfficeVal 采用了完全确定性的**基于代码的验证协议（Code-based Verification Protocol）**。评测只校验最终生成的文件实体，不干预智能体内部是采用多智能体协同、Python 代码沙箱还是操作系统级 GUI 操作。

整个验证器由两套细粒度规则编译而成：

- **可用性规则（Usability Rubric，共 219 项检查）**：底线指标，检验文件是否物理损坏、能否被 Office 正常解析渲染、是否存在全篇乱码、公式严重失效或无法编辑。这是真实职场的“一票否决制”——如果员工提交了一份打不开或者格式全崩的损坏文件，无论里面的文字推导多精彩，其实用价值均为零。

- **任务完成度规则（Task Completion Rubric，共 2009 项检查）**：细化到段落、单元格数据、格式排版、图表生成、要点覆盖等具体指标。更重要的是，该评分表引入了**负向惩罚机制**，不仅对满足需求的要点给予正向赋权，如果智能体在操作过程中擅自修改了模板中的既有有效内容，破坏了既定版式，增加了下游用户的二次返工修复成本，验证代码将实施严厉的倒扣分。

形式化而言，对于任务 $t$，设可用性检查项集合为 $\mathcal{U}_t$，对应二元指标为 $I_u \in \{0, 1\}$，整体可用性判定为：




{% raw %}$$U_t = \prod_{u \in \mathcal{U}_t} I_u$${% endraw %}



设完成度检查项集合为 $\mathcal{C}_t$，权重为 $w_c$，指标为 $I_c \in \{0, 1\}$。原始得分与正向满分基准分别为：




{% raw %}$$S_t^{\mathrm{raw}} = \sum_{c \in \mathcal{C}_t} w_c I_c, \quad S_t^+ = \sum_{c \in \mathcal{C}_t : w_c > 0} w_c$${% endraw %}



为防止负分过度溢出，引入零下界截断 $S_t^{\mathrm{clip}} = \max(0, S_t^{\mathrm{raw}})$。最终任务总分为：




{% raw %}$$\mathrm{Score}(t) = U_t \cdot \frac{S_t^{\mathrm{clip}}}{S_t^+}$${% endraw %}



只有当所有可用性检查全部通过时（$U_t = 1$），任务才能获得非零分数。所有评分 Python 脚本均由代码智能体初建，随后经由资深业务专家比对人工评分与脚本评分，开展多轮“人类-代码分歧裁决（Human-Code Discrepancy Resolution）”，直到自动化校验逻辑与资深业务专家的标准达到高度一致。

### 实验揭示的核心差距：快和便宜，但交付极不稳定

基于上述基准，研究团队对 GLM-5.2、Kimi K2.6、DeepSeek-V4-Pro、MiniMax M3、Qwen3.7-Plus 以及专职人类标注员进行了全面评估。除了常规的算术平均分（Score），基准还计算了以人类劳动时间加权的得分（Time-weighted Score）与以外包价格代理加权的得分（Price-weighted Score）。

主要评测结果反映出当前智能体办公能力的几个关键事实：

首先，**质量鸿沟显著存在**。初级人类标注员的最佳交付件平均得分为 27.79。尽管这离满分 100 分仍有距离（印证了任务本身极高的严苛度和扣分项的敏感性），但已经大幅拉开了与所有参评大模型的差距。在所有模型中，表现最优的 GLM-5.2 仅拿到 17.91 分，其余模型大多在 10 分至 16 分之间徘徊。这表明，在大体量、多约束的长程办公任务上，目前的顶尖大模型在生成可用交付件方面依然存在巨大的系统性短板。

<img src="/images/2607.27155v1/task-score-distribution.webp" alt="任务得分分布直方图与箱线图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从任务得分的分布图可以看出，模型的得分高度集中在接近 0 的低分段。相当一部分任务因为触发了“可用性一票否决”（如导出的 Excel 格式损毁、排版结构破坏）或出现严重的格式破坏性修改，导致最终有效分为零。

其次，**均分最高不等于商业价值最高**。引入经济学加权后的指标揭示了一个值得注意的现象：大模型的平均分排名与价格加权分排名并不完全同步。某些模型在大量简单的、价格较低的日常排版任务中表现尚可，拉高了平均得分；但在那些真正涉及复杂核算、多表交叉验证、外包市场单价高昂的高价值复杂任务上，模型几乎全部崩溃。按价格加权后，模型的商业价值捕获率进一步被压缩。这提示业界：单纯优化基准测试的平均通过率，可能会掩盖模型在最具商业价值任务上的能力缺陷。

<img src="/images/2607.27155v1/time-score-heatmap.webp" alt="人类耗时与得分相关性热力图分析" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

第三，**长周期是模型交付质量的“头号杀手”**。在按人类耗时划分的任务难度阶梯中，人类标注员和各类大模型的得分均随任务时长的增加而下滑，但大模型的衰减曲线要陡峭得多。当人类工时超过 3 小时以上时，大部分模型的完成度迅速趋近于零。随着任务流程的拉长，大模型在上下文维持、深层依赖推导、前后样式一致性控制上的失误呈指数级累积。一次对原始引用的误读或一个破坏性的覆盖写操作，就会让后续所有步骤功亏一篑。

第四，**效率优势极其巨大，但瓶颈不在速度**。在耗时与财务成本上，LLM 展现了压倒性的优势。人类完成一个任务平均需要 2.32 小时，折算人工时薪具有不可忽视的财务成本；而前沿模型完成单任务的平均端到端时间通常在数分钟内，推理 API 成本更是低至几美分到数毛钱人民币。DeepSeek-V4-Pro 在各模型中运行耗时极低，Qwen3.7-Plus 展现出很高的性价比，但过快的生成速度并没有兑现成更高的最终得分。这清楚地证明：现阶段办公智能体演进的核心矛盾，早已不是推理吞吐量有多大、响应延迟有多低，而是能否在保住极低成本与高速度的同时，彻底攻克长程逻辑不漂移、不破坏既有工程规范的“交付可靠性”。

### 从“自动化玩具”到“数字员工”的演进启示

OmegaUse-OfficeVal 带来的启示远超一份单纯的榜单。它向智能体研究界与办公软件产业界传递了清晰的信号：

长期以来，业界习惯于将办公助手包装成“聊天框里的 Copilot”，通过一轮或几轮对话生成一段文字、润色一个段落。但企业级办公自动化的终极形态，必定是具备自主规划、跨应用调度、多源文件操作能力的 Agent。在这个演进路径上，仅考核生成文本“像不像”是远远不够的。

真正的办公交付物评测，必须具备**物理可用性的确定性检验**与**错误引入的负向代价惩罚**。在企业真实业务中，一个把报表格式改崩了、导致整张表所有联动宏失效的 AI 助手，带给人力的修复负担甚至远远超过纯手工从头制作。OmegaUse-OfficeVal 采用的代码级严格验证与双重经济学锚定，为衡量这一维度的净收益树立了一个兼具学术严谨性与商业现实感的方法论范式。

大模型距离替代一名真正合格的初级打工人还有很长的路要走。通往可靠“Vibe Working”的瓶颈，不在于大模型读写 Token 的速度，而在于面对层层嵌套的复杂业务指令时，它能否像人类新手一样，在长达两三个小时的任务跨度里，始终保持对业务意图的精准把握，不漏项、不添乱，交出一份开箱即用的干净作业。
