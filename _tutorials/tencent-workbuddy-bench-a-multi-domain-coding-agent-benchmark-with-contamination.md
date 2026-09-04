---
layout: default
title: "腾讯 WorkBuddy Bench：逆向重构真实 PR，破解 Agent 评测污染"
description: "评估代码智能体（CodingAgent）的能力正在陷入一种尴尬的困境：榜单上的分数越来越高，但在真实工程环境里写起代码来依然漏洞百出。这种落差背后的核心症结在于两点。"
arxiv_id: "2607.20911"
published_at: "2026-09-04T13:15:08.122237+08:00"
topics:
  - "AI Agent"
  - "AI评测"
tags:
  - "AI Agent"
  - "AI评测"
  - "AI论文解读"
related_tutorials:
  - "rex-mle-the-autonomous-agent-benchmark-for-medical-imaging-challenges"
  - "a-framework-for-evaluating-agentic-skills-at-scale"
  - "agentbeats-agentifying-agent-assessment-for-openness-standardization-and-reproducibility"
  - "ai-agent-systems-architectures-applications-and-evaluation"
---

<p class="paper-original-title" lang="en">Tencent WorkBuddy Bench: A Multi-Domain Coding-Agent Benchmark with Contamination-Resistant Task Construction</p>

评估代码智能体（Coding Agent）的能力正在陷入一种尴尬的困境：榜单上的分数越来越高，但在真实工程环境里写起代码来依然漏洞百出。这种落差背后的核心症结在于两点。其一，以 SWE-bench 为代表的静态公开基准，其任务描述和解决方案在互联网上完全公开，许多模型跑出高分并非凭借深度的上下文推理，而是直接撞上了预训练阶段背诵下来的 Pull Request 讨论帖；其二，现存的大多数基准将“软件工程”窄化成了单一的修复 Bug，而忽视了真实开发中更庞大的需求实现、前端交互、多格式数据处理以及安全分析等复合任务。针对这一痛点，腾讯推出了全新的多领域代码智能体评测套件 **Tencent WorkBuddy Bench**。

> ArXiv URL：https://arxiv.org/abs/2607.20911

该基准不仅将评测领域拓宽至代码工程、Web 前端、复杂办公流和红蓝对抗安全四大真实工作场景，更关键的是提出了一套“抗污染任务构建方法”：所有任务均逆向自真实 Commit、PR 或业务场景，重构为口语化、角色化且故意“欠说明（underspecified）”的自然语言需求。这从源头上切断了模型通过公网检索原帖作弊的路径，同时全量开源了包含环境镜像、测试断言和参考解答的执行套件，确保了评测的绝对可审计性与可复现性。

<img src="/images/2607.20911/figure1-unified-benchmark-overview.webp" alt="Tencent WorkBuddy Bench 总览架构" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 评测之困：公开背题与闭门造车的两难

在当前的 Agent 评测生态中，开发者主要面对两类基准，但两类各有难以克服的妥协。第一类是以 SWE-bench、SWE-bench Verified 为代表的静态公开数据集。这类数据集抓取开源社区的 Issue 和 PR，其题目表述、排错过程和补丁代码在 GitHub 及各大技术论坛中随处可见。当大模型爬取了海量公开代码库后，在这些基准上的提升很难分清究竟是长程规划与代码推理能力的跃升，还是对已知上下文的记忆复现。

第二类则是以商业工具内部基准为代表的闭源方案，例如 CursorBench。它们直接从线上用户真实的交互会话中抽取任务，分布自然极为贴合实际生产。然而由于涉及用户隐私和商业秘密，外部研究者既无法查看任务的具体分布，也无法复现其实验，甚至难以排除基准对自研工具存在隐性偏置的可能性。

企业组织如果想客观衡量一个将要落地的通用代码智能体，需要的是兼顾两者优点的基准：既要有来自生产一线的真实任务分布，又要在机制设计上斩断搜索泄露的污染通道，同时还要做到完全开源，让任何第三方都能独立复现与审查。Tencent WorkBuddy Bench 正是在这样的取舍下成型的。它没有直接搬运公开的 Issue 原文，也没有挪用受隐私约束的用户会话，而是将内部积累的真实请求分布模式抽离出来，对开源软件真实演进记录与企业业务场景进行逆向重塑。

### 逆向工程：如何从源头封堵 Prompt 污染

评测污染最隐蔽也最致命的途径，是 Prompt 文本本身的可检索性。如果任务描述里包含了原 Issue 中独特的报错堆栈、异常关键字或特定的类名命名，模型往往能直接联想到对应的修复方案。Tencent WorkBuddy Bench 对此给出的解法是“逆向重构与口语化表达”。

每一个任务的原始源头虽然锚定在真实的提交记录或典型场景中，但任务描述被全面重写为同事或客户日常提需求的口吻。指令中严格扣留了底层根因分析（Root Cause）、参考代码差异（Reference Diff）以及任何可能直接“送分”的排障线索。任务要求像真实职场协作一样，充斥着一定程度的意图和约束，而非详尽的系统设计说明书。

这种设计的核心在于“故意欠说明（Deliberate Underspecification）”。在实际工作中，很少有产品经理或业务同事能直接指出需要修改哪一个文件的哪一行，他们往往只表达核心意图与业务边界。因此，基准中的任务常常不显式交代目标文件、模块接口细节以及边缘异常的定义。Agent 必须自主深入到工作区（Workspace）中，检索代码仓库、解析已有数据结构并理清依赖关系，补齐缺失的隐式假设。这把考核重点从单纯的“语法与代码填充”拓展到了真正的“需求歧义消解与环境落地（Grounding）”。

更为严谨的是任务准入的双向门禁机制（Oracle-gated admission）。一个候选任务在正式收录前，必须在沙盒中经历两轮自动化验证：在代码未作任何修改的初始基线工作区下，验证器给出的奖励得分必须满足 $\text{Baseline Reward} \le 0.3$；而在加载了参考修复方案（Gold Patch）之后，验证器奖励必须达到 $\text{Oracle Reward} = 1.0$。这道硬性门禁彻底排除了初始环境本身就已经误打误撞达标的无效任务，也剔除了参考答案无法稳定通过判题的脆弱用例。

### 严密沙盒：执行时态隔离与 Harbor 规范

评测资产如果与 Agent 运行环境混在一起，很容易引发另一种形式的“偷窥”。Tencent WorkBuddy Bench 采用了统一的 Harbor 风格任务目录格式，在物理层面上对智能体的感知边界做出了硬隔离。

任务包中的 `environment/` 目录定义了纯净的 Docker 镜像，构建时仅仅拷入 Agent 所需的工作空间文件，任何用于校验评分的测试资产都不会被打包进容器镜像。在 Agent 与环境交互的整个生命周期内，它能访问的只有任务说明 `instruction.md` 和业务源码。只有当智能体宣布任务结束（Episode Ends）并退出后，沙盒外的评测流水线才会挂载 `tests/` 目录下的评分用例与断言脚本。这种设计保证了“隐蔽测试（Hidden Tests）”针对的是 Agent 运行期的可见性，而非对开源社区的保密，任何第三方均可随时查阅完整的测试套件。

<img src="/images/2607.20911/fig-code-scenario.webp" alt="代码场景工作流与评测流程" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 四大领域并行：重构软件工作的真实全貌

现实中的开发者并非整天沉浸在修复 Bug 这一件事上。一个真正好用的代码智能体，往往要在前端开发、跨文档业务处理以及安全运维等不同边界来回切换。Tencent WorkBuddy Bench 将评测切分为四个平行子集，每个子集拥有独特的任务形态与评判逻辑：

1. **Code（仓库级软件工程）**：包含 80 个高难度任务，彻底打破了传统基准中“80% 以上全是修 Bug”的狭窄分布。该子集覆盖了从 L2（小型项目）到 L5（大型多模块代码库）的复杂度阶梯，涵盖功能实现、重构、算法工程、测试补全和产品分析等场景。任务甚至由开发者、算法工程师、产品经理、质量保障（QA）和运维（Ops）五种不同的业务角色发出，要求 Agent 在大型代码库中自主穿梭定位。

2. **Web（前端与可交互界面）**：包含 70 个任务，恪守“要可运行产物，不要纯对话输出（Artifact-not-chat）”的原则。模型如果只是在聊天框里给出一大段格式漂亮的 HTML/CSS 代码，但在指定输出路径上没有落盘生成可执行产物，将被直接判为 0 分。该子集涵盖页面交互、数据可视化、图表渲染及文档转换。评分机制采用了三层递进判别：规则引擎校验文件结构与必选项，大语言模型与多模态模型（LLM/VLM Judge）评估视觉还原与布局语义，同时引入 Agent-Judge 模拟用户真实点击与交互，检查前端状态在运行时是否产生预期的流转。

<img src="/images/2607.20911/fig-web-scenario.webp" alt="Web 前端任务交互与评测场景" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

3. **Office（企业混合文件工作流）**：包含 50 个真实办公场景。工作区内堆叠着表格、PDF、JSON 导出数据、Markdown 笔记和复杂的文件夹结构。Agent 需要编写脚本或调用工具处理混合格式数据，完成跨文件的信息核对、数据透视并生成可交付物。评分同样融合了针对文件输出的确定性规则，以及基于留存证据链的大模型语义裁判。

4. **Security（红蓝对抗与安全实战）**：收录 60 个安全向任务，跳出“单纯写安全补丁”的窠臼，全面覆盖白盒漏洞复现、逆向恶意样本分析和安全运维操作。该子集舍弃了任何带有主观波动的大模型裁判，全部依赖纯代码编写的确定性判题脚本（`scoring.py`），对漏洞触发凭证和沙盒状态做出非黑即白的严谨评判。

特别值得注意的是，该基准拒绝将这四个领域的得分生硬计算一个“全局平均分”。因为代码子集靠单元测试判据，Web 子集结合了多模态与交互状态驱动，Office 子集依赖证据链判别，而安全子集依赖确定性脚本，评分度量尺度的物理意义截然不同。消解掉这种跨领域的虚假平均，正是为了避免模型厂商在不同维度上“取长补短”拼凑综合高分，要求研究者诚实面对模型在特定工程边界下的短板。

### 跨框架评测与行业启示

为了验证评测框架本身的稳健性，套件不仅支持腾讯自研的 CodeBuddy Code 运行环境，还无缝适配了 Anthropic 推出的 Claude Code 等主流智能体驱动框架（Agent Harness）。在多款顶尖模型家族的实际运行对比中，这种对长上下文理解、多模块依赖探索以及非结构化需求对齐的要求，迅速拉开了不同模型梯队之间的实际差距。

Tencent WorkBuddy Bench 的发布，给大模型时代的代码评估体系带来了明确的启发：

首先，单纯依靠扩大静态公开数据集规模的思路已经难以为继。面对具备全网抓取与记忆能力的新一代大模型，评测集的寿命很大程度上取决于任务构建时对底层线索的脱敏与重组深度。通过“逆向工程真实代码变更 + 口语化角色代入”构建任务，为防止基准快速老化提供了一条可复制的范式。

其次，代码智能体的能力定义必须从“解题”走向“交付”。真实的企业软件工程从来不是给出一个正确的 diff 片段就万事大吉，它伴随着模糊的上下文、缺乏文档的代码库、多模态的界面校验以及复杂的团队协作角色。只有让评测基准的形态与这种复杂生态保持同构，大模型在榜单上的跃迁，才能真正转化为生产线上的真实生产力。
