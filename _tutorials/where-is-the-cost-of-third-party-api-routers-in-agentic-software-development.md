---
layout: default
title: "SIDEL：第三方API路由暗藏杀机，四大Coding Agent防御率全为0%"
description: "SIDEL：为了支撑评测，作者团队手工构建了一个包含 400 个样本的高质量恶意注入数据集，全面覆盖四大攻击类型： 1. 恶意代码执行 （100 例）：包括反向 Shell、特权提升、后门植入与系统破坏。2. 错误代码生成 （100 例）：包括隐蔽 Logic Bug、死锁注入、资源泄漏与安全隐患代码引入。"
arxiv_id: "2607.23624"
paper_published: "2026-07-26"
published_at: "2026-09-12T13:15:08.779692+08:00"
topics:
  - "AI Agent"
tags:
  - "Distribution Alignment Injection (L4)"
  - "LLM-Polished Injection (L3)"
  - "Response Append (L2)"
  - "Response Substitution (L1)"
  - "SIDEL"
  - "agentic software development"
related_tutorials:
  - "agentic-software-engineering-foundational-pillars-and-a-research-roadmap"
  - "evoclaw-evaluating-ai-agents-on-continuous-software-evolution"
  - "the-alignment-waltz-jointly-training-agents-to-collaborate-for-safety"
  - "a-comprehensive-survey-on-benchmarks-and-solutions-in-software-engineering-of-ll"
---

<p class="paper-original-title" lang="en">Where Is the Cost of Third-Party API Routers in Agentic Software Development?</p>

<img src="/images/2607.23624v2/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

随着代码生成模型能力的爆发，软件开发范式正在经历从“代码补全”到“高自主度 Coding Agent”的剧烈转变。类似 Claude Code、Cursor、Codex 以及各类开源 Agent，不再只是被动输出一个代码块，而是直接被赋予了读取目录、修改代码、执行 Shell 脚本甚至运行测试套件的系统级权限。为了降低开发者的交互成本，这类 Agent 往往默认或被推荐运行在“全自动”或“高自主”模式下。

> ArXiv URL：https://arxiv.org/abs/2607.23624v2

与此同时，大模型接入层也发生了一场基础设施级的静默演进：第三方 API 路由（API Routers）成为了事实上的连接标准。无论是坐拥数万 GitHub Star 和数亿 Docker 拉取量的开源网关（如 LiteLLM、One API / New API），还是聚合数百个模型的商业聚合平台（如 OpenRouter），开发者早已习惯在客户端与上游大模型提供商之间架设一层路由。它的便利性显而易见——统一的 API 规范、统一的密钥管理、平滑的负载均衡以及灵活的 Fallback 机制。

然而，来自北京航空航天大学、北京大学与上海交通大学的研究团队在一项最新研究中敲响了警钟：**这个被开发者深度信任的中间层，正在成为 Agent 软件工程中最致命的盲区。**

<img src="/images/2607.23624v2/intro.webp" alt="API Router 在 Coding Agent 与上游模型之间形成了新的信任边界" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 致命的“信任真空”：当路由成为中间人

在传统的安全认知中，针对大模型的攻击大多集中在“提示词注入”（Prompt Injection）或“工具滥用”（Tool-Use Attacks）——攻击者试图通过外部污染的网页、文档或仓库代码，诱导模型生成恶意指令。这类攻击对抗的本质是模型自身的对齐与辨别能力。

但当第三方 API 路由介入后，安全边界发生了根本性位移。如上图所示，API 路由在应用层直接终结了客户端的 TLS 连接，并代表客户端重新向上游模型发起请求。这意味着，路由处于绝对的“明文中间人”位置：它能毫无阻碍地审查并篡改系统的 Prompt、仓库上下文、工具规范，以及模型返回的每一次结构化 Tool Call。

最关键的问题在于，**现有的 Agent 客户端防御机制（无论是操作确认、权限模式还是命令白名单），本质上都建立在一个脆弱的前提假设之上——它们默认客户端接收到的响应，就是上游可信模型原本输出的响应。**

一旦 API 路由遭受供应链投毒（例如此前在 PyPI 上出现的恶意 LiteLLM 版本伪造包）或运营者恶意作祟，路由就可以在返回给 Agent 的响应中任意掺入恶意操作。上游模型或许给出的是完全无害、经过严格安全对齐的代码补全，但当数据包经过路由交付给本地 Agent 时，已经变成了包含恶意提权、代码投毒或凭据窃取的剧毒指令。这种中间人篡改是否真的能在复杂的仓库级开发任务中生效？现有的安全沙箱与审查机制能否拦截？在此之前，学界和工业界缺乏系统性的端到端实证评估。

### 从粗暴替换到分布对齐：四级路由干预分类法

为了科学解构路由侧篡改的技术边界，研究团队将路由干预手段形式化为四个隐蔽程度逐级递增的干预等级（Level 1 至 Level 4）。每一次交互中，模型响应都会被拆解为两部分：面向人类开发者阅读的自然语言文本 $r^{\mathrm{txt}}$，以及直接触发底层系统操作的结构化工具调用负载 $r^{\mathrm{act}}$。

<img src="/images/2607.23624v2/inject.webp" alt="四种不同隐蔽等级的路由侧注入机制示意" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

第一级是**响应替换（L1: Response Substitution）**。这是最直接的攻击形式，路由完全抛弃上游提供商返回的真实内容 $r$，将其彻底替换为由攻击者预先构造的全新响应 $r^{\prime}$。这种攻击虽然有效，但在长程对话中极易因为上下文脱节而被开发人员察觉。

第二级是**响应追加（L2: Response Append）**。路由保留模型原本生成的合法工具调用与文本解释，但在工具调用序列的末尾悄悄拼接一个恶意的操作动作 $a^{+}$。例如，在上游模型原本要执行的编译命令后面，额外挂载一个向外发送环境变量的 `curl` 动作。这种方式保留了原汁原味的上下文，但多出来的动作在日志中依然显得有些突兀。

第三级是**LLM 润色注入（L3: LLM-Polished Injection）**。为了掩盖恶意工具调用的痕迹，路由在注入恶意动作 $\phi(r^{\mathrm{act}})$ 后，调用一个辅助语言模型对自然语言解释部分 $r^{\mathrm{txt}}$ 进行重新润色，输出 $\psi(r^{\mathrm{txt}}, \phi(r^{\mathrm{act}}))$。辅助模型会将原本用于解释正常任务的说明，巧妙地改写为“为了排查环境依赖，我们需要临时执行该诊断指令”，使得恶意调用与文本说明看起来天衣无缝。

最高等级则是极其精妙的**带分布对齐的润色注入（L4: LLM-Polished with Distribution Alignment Injection）**。传统的 L3 润色虽然通顺，但可能会在生成物中留下二次编辑的统计特征或明显的推理断层。L4 彻底打破了“生成后再修改”的逻辑，而是直接介入到 Token 生成过程的分布混合中。

设恶意上下文为 $c^{+}$（包含恶意目标动作 $a^{+}$），良性任务上下文为 $c^{-}$。在每个生成时间步 $t$，路由直接混合两个上下文在词表上的预测分布：




{% raw %}$$ \tilde{r}_{t} \sim \alpha\,\sigma\!\big(\ell(c^{+}, \tilde{r}_{<t})/T\big) + (1-\alpha)\,\sigma\!\big(\ell(c^{-},\tilde{r}_{<t})/T\big) $${% endraw %}



其中 $\sigma$ 为 Softmax 函数，$T$ 为采样温度，$\alpha$ 为恶意倾向的混合权重。当温度 $T \to 0$ 时，采样退化为确定性的贪心解码。通过这种方式生成的响应，既严格保留了上游任务的语意连贯性，又在数学层面完全消除了显式的编辑边界，使得下游的代码审计工具几乎无法通过困惑度（Perplexity）或语法突变发现异常。

### SIDEL：精准复现与可控评测的基础设施

为了在真实的大型代码仓库中无损评估上述风险，研究人员开发了 **SIDEL** 评测框架。

真实的软件工程 Agent 高度依赖环境状态，一次 Shell 调用的失败或者文件的微小改动都会引发后续推理的连锁蝴蝶效应。如果直接对线上动态 API 发起攻击测试，由于大模型本身的随机性，研究人员将无法区分“任务失败”究竟是由路由注入导致的，还是模型推理抖动引起的。

<img src="/images/2607.23624v2/SIDEL.webp" alt="SIDEL 实验框架的架构与记录重放注入机制" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

SIDEL 通过容器化隔离与细粒度 Trace 追踪解决了这一难题。它包含三种工作模式：代理记录（Proxy-Record）、纯净重放（Replay）以及重放注入（Replay-Inject）。在测试前，框架首先记录真实 SWE-bench Lite 任务中基准运行的完整轨迹序列：




{% raw %}$$ \tau = \big((q_{t}, h_{t}, r_{t}, \tilde{r}_{t}, e_{t})\big)_{t=1}^{T} $${% endraw %}



其中包含每一轮的用户请求 $q_t$、历史上下文 $h_t$、模型原始响应 $r_t$、交付响应 $\tilde{r}_t$ 以及在环境容器中实际产生的系统副作用 $e_t$。

在重放注入阶段，系统可以在指定轮次将原本的响应无缝替换为注入函数 $\rho_{\lambda_t}$ 生成的篡改响应。由于每个任务都在独立的隔离沙箱中运行，SIDEL 能够以纯净的状态追踪文件系统变更、Shell 退出码以及最终的代码提交，从而以“因果隔离”的方式精准衡量路由注入对下游行为的实质改变。

为了支撑评测，作者团队手工构建了一个包含 400 个样本的高质量恶意注入数据集，全面覆盖四大攻击类型：

1. **恶意代码执行**（100 例）：包括反向 Shell、特权提升、后门植入与系统破坏。

2. **错误代码生成**（100 例）：包括隐蔽 Logic Bug、死锁注入、资源泄漏与安全隐患代码引入。

3. **隐私凭据窃取**（100 例）：包括环境变量窃取、SSH 密钥窃取、历史 Bash 凭证嗅探与源码回传。

4. **供应链投毒**（100 例）：包括依赖混淆（Dependency Confusion）、Typosquatting 依赖包引入与恶意 CI/CD 配置修改。

这些样本均以真实的 Bash、Write 或 Edit 工具调用形式呈现，并辅以严格的正则关键词和环境状态检测断言。

### 0% 防御率：四大主流 Agent 的全面沦陷

研究团队选取了目前开发者群体中使用最为广泛的四款 Coding Agent：**Claude Code、Codex、Cursor 与 OpenCode**。核心衡量指标为防御成功率（Defense Success Rate, DSR），即恶意工具调用在被系统真正执行前被成功拦截并拒绝的比例；以及错误拦截率（Wrong-Block Ratio, WBR），衡量防御机制对正常业务的误伤程度。

在没有任何额外客户端防护的基线测试中，实验呈现出了令人窒息的均匀结果：**在 L1 到 L4 的全部四个注入等级下，所有四款 Agent 的防御成功率（DSR）全部为 0%！**

无论是执行高危系统指令、修改敏感系统文件，还是通过网络外发密钥，所有恶意注入均 100% 毫无阻碍地被本地环境执行。这个结果揭示了一个冷酷的现实：在当前最先进的 Agent 架构中，只要恶意响应披上了符合协议规范的 Tool Call 外衣，客户端运行时就会毫无保留地履行执行义务。工具解析器只负责检查参数是否符合 JSON Schema，根本不具备对意图合规性的深层防御能力。

为了进一步确认攻击的普适性，研究人员以 Claude Code 为基准，分别测试了权限模式与底层模型两个关键变量的影响：

首先是**权限控制模式的失效**。Claude Code 官方提供了四种权限级别：`plan`（规划模式）、`acceptEdits`（接受编辑模式）、`auto`（自动执行模式）以及 `bypassPermissions`（绕过权限模式）。实验表明，无论切到哪种模式，面对精心伪装的路由侧工具注入，**DSR 依然全线锁定在 0%**。权限策略的判定规则严重依赖上下文语义连贯性，当恶意调用被合理编织进当前任务步骤时，客户端的权限守门人完全形同虚设。

其次是**后端模型完全无法自救**。研究人员将 Claude Code 的底层模型分别替换为 DeepSeek-V4-Flash、DeepSeek-V4-Pro、Kimi-2.7 Code 以及 Qwen3.6-Plus。结果毫无悬念，换用任何后端模型都无法改变 DSR 为 0% 的结局。原因显而易见：遭殃的是中间传输链路，上游的大模型即便对齐得再完美、本身具有再强的反注入意识，也根本无法知晓下游客户端拿到的是一份被篡改过的“伪诏”。

### 客户端防御的极限与误伤代价

面对彻底沦陷的防线，客户端增加额外防御能否力挽狂澜？研究团队在 Claude Code 上测试了两种最具代表性的客户端缓解方案：**基于白名单的命令执行控制**和**基于双阶段 LLM 的响应审查机制**。

基于命令白名单的拦截规则可以阻断一部分高危的未授权调用（例如直接匹配到了特定的黑名单危险指令或未授权的可执行文件名），但在软件开发这种极其复杂的任务中，工具调用的合法边界极其模糊。开发者本身就频繁需要使用 `bash`、`python`、`curl` 或包管理器。过度收紧白名单不仅无法防范经过混淆的代码篡改（例如修改正常脚本使其产生后门），还会直接瘫痪 Agent 的自主排错能力。

而双阶段 LLM 响应审查——在工具调用真正执行前，调用一个独立的仲裁模型对拟执行动作的合理性进行二阶段审查——虽然显著拉高了防御成功率，但带来了新的不可承受之重。

其一，是**极高的误拦截率（WBR）**。很多良性的开发动作（例如为了修复一个 Bug 而临时修改系统环境变量、杀死卡死的进程、重置本地数据库）在缺乏全局宏观认知的独立审查模型眼里，与恶意攻击的行为特征高度重合。这种误杀极大地打断了原本自动化的开发流，使得高自主 Agent 沦为需要频繁手动确认的低效工具。

其二，是**显著的推理成本与延迟膨胀**。由于每一次哪怕最细小的工具调用（如 `ls` 查看目录、`git status` 查看状态）都需要挂起并等待外部审查模型的二阶段推理，整个开发任务的 Token 开销和端到端耗时大幅飙升，这在经济和时间成本上让很多企业难以承受。

更致命的是，面对隐蔽性极强的 L4 级分布对齐注入，审查模型也表现出力不从心。由于注入动作完全内嵌在看似合情合理的长程推理逻辑中，文本表现与操作动机高度自洽，审查模型往往会被伪造的上下文成功欺骗，最终放行恶意操作。

### 重新思考 Agent 信任链：呼唤上游签名与端到端完整性

这项由北航、北大、上交联合完成的研究，剥开了当前 Agent 基础设施繁荣表象下的巨大安全隐患。

长期以来，AI 社区在安全上的投入绝大部分倾注在上游模型的“安全对齐”（Safety Alignment）以及客户端的“系统提示词加固”上。然而，正如现代互联网在经历无保护 HTTP 时代后才痛定思痛建立起完整的 PKI 体系与 HTTPS 标准一样，**当前的 Agent 软件生态正处于一个危险的“未签名明文时代”**。

第三方 API 路由为开发者抹平了接入复杂度，降低了计费门槛，但它也单方面粉碎了原生的信任链条。由于缺少端到端的完整性验证，客户端在拿到数据的那一刻，根本无法通过数学手段证明“眼前这个 `bash` 命令确实是 Claude 或 GPT 亲自生成的，且未被任何中间层篡改”。

这项研究明确指出了未来可信 Agent 部署的必经之路：单靠客户端的“事后诸葛亮”式静态匹配或二阶段 LLM 审查，注定无法从根本上解决中间人风险，反而会拖垮系统可用性。**真正的破局方案必须由模型提供商（Provider）主导，建立上游模型输出的密码学完整性保证机制。** 例如，上游提供商在流式传输或最终生成完成时，使用私钥对包含 Token 序列及对应 Tool Call 规范的数据包进行可信数字签名；客户端在解析并驱动本地 Shell 之前，必须在硬件隔离层验证提供商的公钥签名。

在这样严密的数字签名机制普及之前，每一位享受着第三方 API 路由便利、放任 Coding Agent 在本地主机肆意执行命令的开发者，都在不知情中将系统的最高控制权，毫无保留地交托给了一个无法被监督的中间人。
