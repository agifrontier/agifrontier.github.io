---
layout: default
title: "ElasticBack：无需篡改模型权重，单技能条件后门拿下89%攻击成功率"
description: "为了破解上述矛盾，ElasticBack 提出了一种以“门控词”（Gate Words）为枢纽、将触发词视作激活开关的联合优化框架。整个攻击流程的核心思想是： 先利用语义锚定技术在技能文档中精准植入一条带有严格约束边界的休眠规则 ，将其参数与行为锁定后，再依托黑盒代理模型。"
arxiv_id: "2608.09577"
paper_published: "2026-08-10"
published_at: "2026-09-08T13:15:08.162312+08:00"
topics:
  - "AI Agent"
  - "模型优化"
tags:
  - "ElasticBack"
  - "LLM-agent skills"
  - "conditional backdoor"
  - "semantic-anchored rule injection"
  - "skill supply chain"
  - "stealth-constrained genetic search"
related_tutorials:
  - "evaluating-skills-not-just-agents-agentic-continuous-evaluation-of-skills"
  - "rangefactory-scalable-construction-of-multi-hop-cyber-ranges"
  - "a-language-for-describing-agentic-llm-contexts"
  - "skills-are-not-islands-measuring-dependency-and-risk-in-agent-skill-supply-chain"
---

<p class="paper-original-title" lang="en">ElasticBack: Stealthy Conditional Backdoor in LLM-Agent Skills via Coupled Trigger-Rule Optimization</p>

<img src="/images/2608.09577v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在面向任务的大模型智能体（LLM Agent）生态中，“技能”（Skills）正在成为一种核心的基础设施。无论是 Claude Code、Codex CLI 还是各类基于工具调用的 Agent 框架，开发者早已习惯将复杂的外部 API、代码运行环境或格式规范打包成轻量级的技能包。用户只需要在本地或云端安装一个包含自然语言操作指南（`Skill.md`）和脚本模板的技能文件夹，Agent 就能在遇到相应领域任务时自动加载并执行它。

> ArXiv URL：https://arxiv.org/abs/2608.09577v1

然而，这种完全依赖自然语言和开放脚本的“即插即用”特性，正悄然打开一条危险的软件供应链攻击面。来自重庆大学、南京航空航天大学、南洋理工大学与东北大学的联合研究团队，在一项针对 Agent 技能生态的安全研究中，揭示了一种隐蔽且低成本的攻击范式——**ElasticBack**。这项工作设计了首个不需要篡改底层模型权重、仅依靠单一被污染技能即可生效的“条件后门”（Conditional Backdoor）。实验显示，该方法在四大主流 Agent 骨干模型与上百个真实技能测试中，取得了平均 **89% 的攻击成功率（ASR）**，同时将**误报触发率（FPR）压制在 6%** 左右，且对正常任务的完成度几乎没有产生负面干扰。

这项研究最警醒之处在于，后门并非时刻处于激活状态，也不是靠破坏 Agent 常规逻辑来运行；它在绝大部分日常交互和平台安全审查中表现得完全正常，只有当带有特定“开关”特征的查询传入时，隐蔽的恶意载荷才会被精准唤醒。

<img src="/images/2608.09577v1/F1.webp" alt="技能后门攻击管道与既有方案对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 技能生态的信任盲区与攻击困境

现代 LLM Agent 的能力扩充遵循着一种极度轻量化的分发逻辑。为了让模型获得特定能力，系统不会去重新微调百亿或千亿参数的基座，而是直接读取开放社区分发的自然语言说明文档。以 Anthropic 推进的开放 Agent 技能规范为例，模型只需在任务匹配时解析 `Skill.md` 中的说明，并依此调用绑定的辅助脚本。实际审计显示，开源社区中超过 13% 的公开技能包已经携带不同程度的安全隐患，而开发者和终端口径对这些下载即用的技能几乎保持着“默认信任”。

以往针对 Agent 技能的安全攻击大多存在明显的战术缺陷。第一类攻击是“无差别触发型”，例如将恶意指令写死在 `Skill.md` 中，每次只要技能被调用，恶意载荷就会执行。这种方式虽然实现简单，但极其容易在开发者的常规测试或平台的静态文本检测中暴露。第二类攻击试图引入条件触发，但往往代价昂贵：有的需要攻击者在技能包内强行捆绑一个经过特定数据微调过的自定义小模型，有的则需要把加密载荷分散切片到多个互相关联的技能中协同生效。

对于攻击者而言，最理想的场景是：**不修改任何模型参数、不依赖复杂的跨技能拓扑，仅仅发布一个表面功能完全正常的单一技能文件，就能让后门长期潜伏**。但要在纯提示词（Prompt-based）与自然语言指令层面做到这一点，面临着两道极难调和的技术鸿沟。

第一道鸿沟是“规则与触发词的分离失真”。为了保证后门不被日常请求引爆，攻击者必须在技能文档中写下一条激活条件规则 $R$，而在实际受害者发送的查询中植入触发短语 $T$。然而在现实部署环境下，攻击者无法实时窃看受害者的动态查询，两端内容是解耦生成的。如果技能文档里的规则条件写得太宽泛，普通的合法查询就会误触发恶意代码，导致假阳率过高、暴露风险激增；反之，若触发词设计得太抽象，或者规则约束过于刻板，模型在推理时往往难以建立稳固的条件关联，导致攻击成功率大幅下跌。

第二道鸿沟则是“双端隐蔽性与对抗鲁棒性”。无论是技能文档里的规则 $R$，还是查询端的触发词 $T$，都必须经受平台防御机制的检验。直接在文档中追加一段突兀的恶意代码调用指令，容易在困惑度（Perplexity）检测或大模型裁判（LLM-as-a-Judge）审计中被秒杀；而在用户端塞入语法不通的乱码或生硬词块，同样难以逃脱输入过滤网。

### 规则与触发词的耦合优化机制

为了破解上述矛盾，ElasticBack 提出了一种以“门控词”（Gate Words）为枢纽、将触发词视作激活开关的联合优化框架。

<img src="/images/2608.09577v1/F2.webp" alt="ElasticBack 框架架构图" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

整个攻击流程的核心思想是：**先利用语义锚定技术在技能文档中精准植入一条带有严格约束边界的休眠规则 $R$，将其参数与行为锁定后，再依托黑盒代理模型，利用遗传算法在查询侧针对性演化出一个兼具高激活率与自然语感的触发短语 $T$。**

#### 语义锚定注入：把规则藏进注意力的盲区

在技能文档侧，攻击者需要植入恶意规则 $R$ 和伪装成正常工具的恶意脚本 $s$。为了防止文档直接被安全探针标红，ElasticBack 并没有随意追加段落，而是先通过注意力显著性图分析文档内部的注意力流动，筛选出整篇说明文档中语义权重最低、对正常任务逻辑影响最小的位置 $p_R$ 作为插入点。

随后，框架通过跨域语义映射，为恶意脚本 $s$ 挑选一个与该技能原本业务场景高度拟合的伪装名称：




{% raw %}$$ \text{cover} = \arg\max_{c} \cos\big(\mathrm{SBERT}(c_{\text{payload}}), \mathrm{SBERT}(c_{\text{skill}})\big) $${% endraw %}



紧接着，算法从恶意载荷的核心词表中提取出一组具有区分度、但在该技能常规语料中极少出现的“门控词”集合 $\mathcal{G}$：




{% raw %}$$ \mathcal{G} = \{\,w \in \mathcal{W}_{\text{pay}} : \mathrm{sim}(w, D_{\text{core}}) < \theta_g\,\} $${% endraw %}



门控词的作用是充当这扇“后门”的专属钥匙孔。最后，算法根据门控词合成包含三段式逻辑的自然语言条件规则：




{% raw %}$$ R: \underbrace{\text{When } C,}_{\text{条件}} \ \underbrace{\text{you invoke } s \text{ to } Z, \text{ then produce } \delta}_{\text{执行载荷}} \ \underbrace{\text{Otherwise, proceed normally.}}_{\text{负向作用域}} $${% endraw %}



这里的“负向作用域”（Negative Scope）至关重要。它明确要求 Agent 在未满足条件 $C$ 时严格执行常规逻辑。这一步从语义逻辑层面封堵了模型在面对普通指令时的幻觉与误调倾向。在生成多个候选规则后，框架利用代理模型输入良性测试集，挑选出误触率最低且区分度最大的那条规则，并将其在文档中**彻底冻结**。

#### 语料驱动演化：以遗传算法定制自然触发短语

在规则 $R$ 已经固化的前提下，接下来的核心任务是寻找那个能够一击即中、同时看起来毫无破绽的触发词 $T$。ElasticBack 采用了受隐蔽性约束的混合遗传算法（C-HGA）在目标技能语料上进行定向演化。

该算法将触发词优化的适应度函数设计为攻击有效性与隐蔽性指标的复合体：




{% raw %}$$ F(T) = \alpha \widehat{\mathrm{ASR}}(T) + \lambda \mathrm{Stealth}(T) - \Pi(T) $${% endraw %}



其中 $\widehat{\mathrm{ASR}}$ 代表在代理模型上的攻击激活概率，$\Pi(T)$ 是针对语法畸变和格式失真的惩罚项，而隐蔽性评分 $\mathrm{Stealth}(T)$ 则被细化拆解为三个正交维度：




{% raw %}$$ \mathrm{Stealth}(T) = w_s \mathrm{SemFit} + w_r \text{R-Sim} + w_d \mathrm{DepAtt} $${% endraw %}



这三个子项各自承担明确的技术约束：

- **语义契合度（SemFit）**：评估触发短语是否自然归属于当前技能对应的业务语境，避免突兀跨界；

- **规则对齐度（R-Sim）**：衡量短语与文档中休眠条件 $C$ 的语义逼近程度，确保模型能够将其与文档指令产生逻辑关联；

- **依存句法连贯性（DepAtt）**：通过自然语言处理工具包检查短语作为名词修饰成分时的语法合规性，防止生成让困惑度指标飙升的破碎片段。

为了防止演化过程中偶然产生的“幸运触发词”在不同推理实例下失效，算法在收敛阶段引入了置信度下限（Lower Confidence Bound, LCB）筛选机制：




{% raw %}$$ \mathrm{LCB} = \mathrm{mean}(\mathrm{ASR}) - k \cdot \mathrm{std}(\mathrm{ASR}) $${% endraw %}



只有均值高且方差极小的触发词才能脱颖而出。最终，经过确定性句法修复的 $T$ 会被部署到该技能所掌控的下游渠道（如技能附带的示例 Prompt、工具预填模板或 RAG 检索返回内容），在用户执行看似日常的交互时自然注入请求上下文中。

### 实验验证：高成功率、低误报与零权重侵入

为了检验 ElasticBack 的实际破坏力，研究团队在涵盖软件工程、内容生成和数据分析等多个领域的公开基准 ClawHub 上构建了测试集。实验针对三大典型恶意行为（敏感凭证外发、未授权命令执行、篡改业务处理结果），在 50 个自包含的真实技能包上全面铺开，并选择了包括通用开源与商业级大模型在内的 4 种 Agent 骨干。

评估指标严格锚定三个核心维度：攻击成功率 $\mathrm{ASR}$、面对正常用户请求时的误报率 $\mathrm{FPR}$，以及衡量正常任务解决率损耗的 $\Delta_{\mathrm{CAC}}$。一个合格的后门不仅要在输入触发词时精准爆破，更必须保证平时绝对不影响用户的正常工作，否则极易引起怀疑。

跨越 12 组（恶意载荷 $\times$ 受害模型）组合的评测结果证明，ElasticBack 在综合效能上明显拉开了与现有基线方法的差距：

- **杀伤力层面**：ElasticBack 取得了高达 **89% 的平均攻击成功率**，证明利用提示词工程与纯文档优化的条件后门完全能够稳定操控现代复杂 Agent；

- **休眠纯净度**：该方法的平均**误报触发率仅为 6%**，这意味着在绝大部分正常交互中，恶意逻辑处于近乎完全的静默状态；

- **功能无损度**：正常任务的准确率下降值 $\Delta_{\mathrm{CAC}}$ 平均仅为 **3%**，在全部 12 项实验设定中，有 7 项设定的正常任务准确率完全没有产生任何波动（$\Delta_{\mathrm{CAC}} = 0$）。

反观传统的注入基线，它们往往陷入两极分化的尴尬境地：有的虽然能达到相近的 ASR，但由于缺乏精细的作用域收敛机制，FPR 往往高达百分之几十，一经部署便迅速在常规交互中穿帮；另一些基线为了压制误报，导致条件限制过于严苛，在面对具有复杂上下文的提示词时根本无法稳定激发载荷。

<img src="/images/2608.09577v1/alpha_lambda_asr.webp" alt="超参数对攻击成功率与误报率的影响曲线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

<img src="/images/2608.09577v1/alpha_lambda_fpr.webp" alt="超参数对攻击成功率与误报率的影响曲线" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

从权重敏感度分析图可以看出，适应度函数中的效果权重 $\alpha$ 与隐蔽性权重 $\lambda$ 呈现出清晰的博弈特征。当系统调高 $\alpha$ 时，触发词会更激进地迎合代理模型的判别边界，攻击成功率快速饱和；而适当平衡 $\lambda$ 的取值，则能够在保持高 ASR 的同时，将模型的误报风险有效锁死在安全底线之内。

此外，由于整个攻击过程不需要针对受害模型进行梯度回传或微调更新，其计算开销极其微小。整个遗传搜索过程在单块消费级显卡（如 RTX 4060）上，仅需 3 到 5 代演化即可收敛。一旦生成完毕，注入了恶意规则的 `Skill.md` 和伪装脚本就成为了纯粹的静态制品，运行时无需攻击者进行任何额外的计算介入。

### 智能体供应链安全的防御重构

ElasticBack 的提出打破了一个长期以来的安全误区：许多开发者潜意识里认为，只要基座大模型的权重是经过安全对齐的，只要系统中没有引入未经审查的二进制可执行模块，单纯下载几份以 `.md` 格式存在的自然语言指令集是相对安全的。

这项研究直接粉碎了这种侥幸。在以自然语言为交互介质的 Agent 时代，指令就是代码，文本配置就是执行逻辑。当攻击者能够精细化地设计出符合语法分布、语义连贯且仅在极少数语境下才翻转逻辑的“条件开关”时，现有的许多静态防线几乎全面失效：

- **静态困惑度检测失效**：经过遗传算法优化的触发短语和经过注意力低显著区植入的规则文本，在语言模型的困惑度扫描下与常规技术文档没有统计学差异；

- **LLM 裁判脱靶**：直接用大模型审查技能文档时，由于规则尾部显式声明了“若未满足条件则完全正常执行”，审查者往往会将其判定为一种合法的分支处理逻辑，无法识别出其中隐藏的恶意属性；

- **离线沙箱漏检**：沙箱在未拿到确切触发短语的情况下运行测试用例，技能只会展示出完美的良性执行轨迹。

面对这种高度隐蔽的条件后门，防御思路必须从“文档静态过滤”转向更加纵深的“运行时动态防御”。未来的防御体系需要重点推进以下方向：首先是在技能执行层引入严格的沙箱隔离与最小权限原则，禁止自然语言指令无节制地派生系统执行权限；其次是在调用环节实施能力门控，对涉及网络回传、凭证访问和敏感文件写入的操作实施显式的动态鉴权，防止“看似无害”的文档指令以 Agent 的身份在后台静默下发危险调用。

开源社区和应用商店为 Agent 生态带来了繁荣，但也把脆弱的信任链条推到了台前。这项研究所揭示的条件后门机制，向所有正在积极构建自主智能体系统的开发者敲响了警钟：在放权让模型自由加载外部技能之前，安全机制必须先行。
