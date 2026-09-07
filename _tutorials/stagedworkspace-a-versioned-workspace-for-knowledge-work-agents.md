---
layout: default
title: "StagedWorkspace：双视图版本契约，让办公Agent在OfficeQA提升至63.9%"
description: "哈佛大学、斯坦福大学、华盛顿大学与 Raycaster AI 等机构的研究团队近期联合提出了 StagedWorkspace ，首次将工作区状态明确确立为知识工作 Agent 的核心实验变量，并引入了“工作区状态契约”（Workspace-State Contract）。"
arxiv_id: "2608.18050"
paper_published: "2026-08-18"
published_at: "2026-09-07T13:15:08.344824+08:00"
topics:
  - "AI Agent"
tags:
  - "APEX-Agents"
  - "OfficeQA Pro"
  - "StagedWorkspace"
  - "content hashes"
  - "dual parsed/native access"
  - "knowledge-work agents"
related_tutorials:
  - "shared-selective-persistent-memory-for-agentic-llm-systems"
  - "seedance-15-pro-a-native-audio-visual-joint-generation-foundation-model"
  - "scienceflow-a-long-horizon-agent-for-ml-research-scientific-discovery-and-beyond"
  - "seedance-2-0-advancing-video-generation-for-world-complexity"
---

<p class="paper-original-title" lang="en">StagedWorkspace: A Versioned Workspace for Knowledge-Work Agents</p>

<img src="/images/2608.18050v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在大模型迈向自主智能体的浪潮中，软件工程领域率先取得了实质性突破。以 SWE-bench 为代表的代码评测中，Agent 能够依靠 Git 仓库的确定性状态，自如地进行索引定位、增量修改、运行测试并生成标准 Patch。但在更广阔的通用“知识工作”（Knowledge Work）场景——面对层级复杂的 PDF 财报、嵌入宏与复杂公式的 Excel 表格、多页幻灯片以及杂乱的混合格式目录时，各类办公 Agent 却频繁陷入低级混乱。

> ArXiv URL：https://arxiv.org/abs/2608.18050v1

这种混乱往往不是模型推演能力不足，而是环境底座在“系统工程层面”的缺失：**工作区状态漂移（State Drift）**。Agent 在搜索上下文时依赖的是提取出来的文本解析缓存，在执行修改时触碰的是沙箱内的原生二进制文件，在自我审查修改时缺乏结构化增量记录，最终提交的产物更是经常脱离检索依据。哈佛大学、斯坦福大学、华盛顿大学与 Raycaster AI 等机构的研究团队近期联合提出了 **StagedWorkspace**，首次将工作区状态明确确立为知识工作 Agent 的核心实验变量，并引入了“工作区状态契约”（Workspace-State Contract）。

基于该契约实现的 SW-Agent 展现了惊人的系统增益：在 OfficeQA Pro 基准上，配备 Gemini 3.1 Pro 的 Agent 得分从原先基准报告的 29.3% 飞跃至 63.9%；在企业级复杂任务基准 APEX-Agents 上，搭载紧凑型模型 GPT-5.4 Nano 的得分也从 25.5 提升至 42.1。严谨的严格受控消融实验证实，仅仅通过解耦并实时同步“原生文件”与“解析视图”，就能在相同模型、相同 Prompt、相同解析器与工具预算下，带来 8.3 至 12.1 个百分点的准确率净提升。

<img src="/images/2608.18050v1/intro.webp" alt="工作区状态契约总览" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 知识工作 Agent 隐蔽的“状态不同步”死穴

经典的知识工作定义，本质上是通过解释、判断和修订来生产或转化信息资产。代码、文档、报表与研究简报均属于交付型数字工件（Work Product）。在当前的 Agent 架构设计中，环境往往只能在两个极端之间做妥协。

一种是纯工件访问模式（Artifact-only）。Agent 直接面对原始文件系统，虽然保留了完整的文件层次、单元格公式和排版，但由于缺乏全局文本索引，模型不得不一次次翻页、穷举式滚动读取大体积文件，迅速塞满上下文窗口，丢失长程信息。另一种则是纯解析视图模式（Parsed-only）。环境在初始化阶段将所有 PDF 或表格预处理切块为 Markdown 或文本向量供 Agent 检索，虽然检索定位极其迅速，但在切块过程中，多栏排版层级、单元格公式关系、甚至图表视觉依据彻底丢失，且 Agent 根本无法对底层二进制文件执行精确的就地写入。

更棘手的是状态的可变性。在多轮交互任务中，Agent 往往需要一边查阅资料，一边在工作区内就地修改交付物。缺乏版本契约的系统，无法保证“检索到的内容”与“当前写入的文件版本”完全对应。Agent 完全可能在一个步骤中根据旧版表格的摘要做出推断，在下一个步骤中直接覆盖了原生文件，导致解析缓存失效；随后模型又根据陈旧的检索缓存再次推论，最终提交一份与引证依据自相矛盾的交付物。

代码开发中赖以生存的 Git 仓库契约在办公场景中彻底失效了，因为复杂的表格与多媒体简报无法被简单的行式文本差异（Line-based diff）所描述。要让知识工作 Agent 真正可用，工作区必须在底层重新建立契约机制，让每一次检索、每一次读取、每一次修改审查和最终提交，都严格指向同一个演进中的文件版本。

### StagedWorkspace：用内容哈希筑牢底层状态契约

为了弥合这一系统裂痕，研究团队构建了 StagedWorkspace，其核心是将非代码工作区形式化为一个版本受控的状态机。在任意交互步 $t$ 下，工作区状态被严谨定义为三个协同演进的组件：




{% raw %}$$W_t = \text{当前工作区内的原生文件集合}$${% endraw %}






{% raw %}$$C_t = \text{带有源路径与内容哈希标签的解析记录缓存}$${% endraw %}






{% raw %}$$\Delta_t = \delta(W_0, W_t) = \text{自初始状态演进至今的结构化审查差异}$${% endraw %}



当 Agent 通过工具执行文件改写、新增或删除操作时，状态转移遵循明确的同步逻辑：原生文件从 $W_t$ 变更为 $W_{t+1}$，系统立即对受影响的文件重新计算内容哈希值（Content Hash）。此时，缓存同步函数 $\operatorname{sync}(C_t, W_{t+1})$ 会即刻介入，将所有哈希不匹配的解析记录标记为陈旧状态（Stale），同时在后台触发异步增量重解析与重索引。只有当新的解析记录生成并附带最新哈希后，才重新供检索器调度。

<img src="/images/2608.18050v1/ArtifactState_Workspace.webp" alt="StagedWorkspace状态模型与同步逻辑" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在整个交互循环（Agent Turn Loop）中，这一机制彻底切断了“时空错位”的可能。模型如果在步骤 $t$ 搜索某份财报，它拿到的检索文本必然对应此时磁盘上的确切版本；若它顺藤摸瓜需要核查原始排版细节，它通过工具打开的原生文件也严密吻合同一状态。

在差异追踪机制（Journaled Review）的设计上，StagedWorkspace 拒绝了粗暴的黑盒文件替换，而是针对混合文件类型开发了差异化表面。对于纯文本与代码，保留行级 Diff；对于大型电子表格，差异追踪器细化到行列增删以及具体的单元格数值/公式变更；对于演示幻灯片，系统以幻灯片页面为粒度输出版面与文本的增量变化；而对于不可解析的二进制工件，则退回至改动前后的结构化预览。在交付任务前，Agent 可以调用审查工具显式观察 $\Delta_t$，对自己的改动进行就地校验，就像程序员在提交代码前执行 `git diff` 一样。

### 双重视图协作：信息定位与精确执行的最佳解耦

许多开发者直觉上认为，拥有足够长上下文的多模态旗舰模型，只需把原始 PDF 或整个表格生吞下去即可，无需画蛇添足地维护两套视图。然而消融实验展示了完全相反的系统事实：双重访问（Dual Access）构成了 Agent 稳定表现的绝对支柱。

研究团队设计了两种典型的使用范式，清晰解释了双重视图互补的必要性：

1. **先解析后原生（Parsed-to-Native）**：在 APEX-Agents 的复杂数据清洗任务（如 HarFeast 案例）中，项目文件夹包含极度繁复的跨文件调查说明。Agent 首先在解析记录缓存 $C_t$ 中利用全文检索与语义匹配，迅速穿透海量文件，精确定位到定义调查指标列映射关系的指导指南；随后，模型切入原生文件系统 $W_t$，在包含复杂宏与关联公式的原生 `.xlsx` 电子表格上精准执行 Python 脚本清洗与写入，保证了交付工件的真实可用性。

2. **先原生后解析（Native-to-Parsed）**：在面对高密度扫描件或高度依赖排版的 PDF 时，解析器常常在表格合并单元格或跨页标注上产生结构退化。此时 Agent 在检索到可疑区域后，能调动视觉工具回溯原生文件排版进行人工级校验，确认无误后再把编辑结果沉淀回工作区，触发后台的哈希同步更新。

<img src="/images/2608.18050v1/harfeast_column_guide_parsed_cache.webp" alt="HarFeast案例中通过解析缓存定位列指南并在原生工件中执行" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

正是通过将“用于理解与定位的表征”和“用于沉淀与验证的工件”进行契约解耦，Agent 不必在阅读整个超大工作区时耗尽推理预算，也不会在盲目操作原生文件时丧失全局视野。

### 严密受控消融：系统架构带来的惊人净增益

为了杜绝“刷榜技巧”带来的评估偏差，作者在实验设计上极其克制。研究不仅对比了公开榜单得分，更关键的是设计了严格固定控制变量的切片消融（Fixed-harness Ablations）：在完全相同的底座模型、Prompt 模版、底层解析器（Reducto 等）、检索后端、评估裁判以及 250 次工具调用预算上限下，仅仅切换工作区访问视图：

- **Dual（双重同步访问）**：同时开放原生工作目录 $W_t$ 与带哈希校验的解析缓存 $C_t$；

- **Artifact-only（纯工件访问）**：仅开放原始文件与沙箱操作，关闭解析检索缓存；

- **Parsed-only（纯解析缓存）**：仅开放切块后的解析检索系统，无法操作和直接感知原生文件。

在专注于从复杂 PDF 文档库中进行深层次检索与精细化阅读的 **OfficeQA Pro** 基准上，双重视图在所有测试模型上均斩获了最高的点估计成绩。相比受限的单一视图，Dual 配置为 OfficeQA Pass@1 带来了 8.3 至 12.1 个百分点的显著净提升。

而在涉及多格式混合项目目录、要求产出真实专业交付物的 **APEX-Agents** 基准上，双重视图让任务的平均评分（Mean Rubric Score）净增 4.7 到 9.2 分。更重要的是，在包含明确文件就地编辑任务的 57 个 APEX 子任务配对实验中，打开审查差异工具（Diff Visible）相较于盲改直接提交，成绩出现了一致的向上偏移。这有力印证了一个软件工程早就证明、但在办公智能体中被长期漠视的法则：**交付前对差异的自省能力，是保证长流程可靠性的底线**。

更值得深思的是不同模型在系统介入后的反馈差异。传统认知中，能力较弱的轻量模型在长程任务中几乎无法使用，但在拥有了严密的工作区状态契约后，中端与轻量模型的潜力被大幅激活。Gemini 3.1 Pro 在 OfficeQA Pro 上从原本公开测试的 29.3% 暴涨至 63.9%，GPT-5.4 Nano 在 APEX 上的综合表现翻倍。相比之下，处于第一梯队的顶尖超大模型虽然绝对分数更高，但双重契约带来的相对增幅却显得相对平缓。

这一现象揭示了一个极其核心的学术洞见：**在过往未经控制的评测中，基准榜单上的巨大分数鸿沟，很大程度上混合了“模型自身推理算力”与“对糟糕环境状态管理的容错代偿能力”**。超大模型往往凭借着反复重读、盲目试错和极强的上下文冗余消化能力，生生抗住了环境状态漂移的干扰；而中小模型一旦遇到状态脱节便迅速崩溃。一旦由系统底座接管了版本状态的一致性，中小模型的逻辑能力便能完全释放出来。

甚至在运行效率与调用成本的综合权衡上，StagedWorkspace 也击碎了“功能越多越贵越慢”的刻板印象。实验数据表明，Dual 模式的综合成本并未上升，在 Gemini 3 Flash 和 Gemini 3.1 Pro 上反而是成本最低的配置；在耗时维度上，双视图更是全面快于纯解析视图（例如在 GPT-5.4 下由 17.3 分钟压缩至 6.1 分钟）。原因十分直接：精准的索引配合确定的原生访问，使得 Agent 不再需要在解析器返回的大量无用文本中反复纠错打转，减少了无效轮次。

### 同步机制的边界与未来评测转向

尽管 StagedWorkspace 解决了工件状态错位这一底层隐患，但作者在错误分析中坦率地指出了架构能解决与不能解决的问题边界。在 APEX 评测的 452 个任务切片中，有 188 个任务在所有测试模型和消融配置下得分均为 0，其中投资银行等复杂业务分析场景占据了最大失利份额。

通过对 Agent 行为轨迹的逐行人工复核，这些残留死穴并不归咎于文件状态不同步，而是暴露出当前大模型在深水区任务中的固有局限：长程多阶段规划失控、对特定行业复杂业务逻辑的先验理解缺失、或者未能精准遵循人工评估规范中严苛的细项要求。状态契约是知识工作的地基，它能确保 Agent 不再“拿着旧图纸建新大楼”，但如果模型本身根本读不懂图纸背后的财务勾稽关系，系统底座同样无力回天。

这项研究对未来 Agent 基础设施与基准构建带来了深远的启示：

- **告别黑盒终局评分，转向状态转移评估**：现有的知识工作 Benchmark 大多只在终点线设置一个评分器，要么检查最后一道数学题是否匹配，要么对最终交付的文件打整体分。这种设计完全忽略了 Agent 在中间状态的转移质量。未来的评估必须引入细粒度的阶段验收机制，分别评测信息定位（Retrieval）、文件读取切片（Read-region）以及阶段性增量编辑（Staged Edits）的有效性。

- **系统状态必须被视作独立控制变量**：在比较模型 Agent 性能时，如果各家使用的环境 Harness、文件追踪机制和解析同步方案各不相同，跨系统刷榜的分数对比便毫无科学说服力。将工作区契约与解析检索模块显式解耦，将是后续 Agent 评测科学化的必然路径。

从操作系统管理内存，到软件工程拥抱 Git，计算机系统的每一次阶跃，本质上都在做同一件事：将混乱隐蔽的状态，收敛为严谨确定的契约。StagedWorkspace 证明了这一点在知识工作 Agent 时代的不可替代性——在大模型学会像知识工作者一样思考之前，首先必须让它的工作区学会像知识管理系统一样，诚实、同步且可溯地记录每一个字节的变迁。
