---
layout: default
title: "JarvisHub：别让Agent困在对话框，用可编辑画布重构长程多模态创作"
description: "针对这一缺失，开源项目 JarvisHub 给出了一种全新的解题视角：它彻底放弃以线性对话记录为核心的交互范式，提出了一套 以画布为原生载体（Canvas-Native）的开源多模态创作智能体运行底座 。"
arxiv_id: "2607.23588"
paper_published: "2026-07-26"
published_at: "2026-10-03T13:15:07.852414+08:00"
topics:
  - "AI Agent"
  - "多模态&视觉"
tags:
  - "JarvisHub"
  - "agent runtime"
  - "canvas nodes and links"
  - "canvas-native"
  - "external memory"
  - "human-steerable creative automation"
related_tutorials:
  - "a-survey-on-agentic-multimodal-large-language-models"
  - "a-survey-on-multimodal-large-language-models"
  - "act-as-human-multimodal-large-language-model-data-annotation-with-critical-think"
  - "ai-native-games-a-survey-and-roadmap"
seo_title: "JarvisHub: An Open Harness for Canvas-Native Multimodal Creative Agents"
---

<p class="paper-original-title" lang="en">JarvisHub: An Open Harness for Canvas-Native Multimodal Creative Agents</p>

<img src="/images/2607.23588v1/A__title.webp" alt="" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

当大模型的多模态生成能力迅速逼近商业可用水平，文生图、文生视频、UI 界面合成以及音效配乐的单点质量已经不再是最大瓶颈。然而，在真实的专业创作场景中，几乎没有任何优秀作品能够依靠单次 Prompt 一键生成。实际的创作工作流充满反复横跳的试错过程：创作者需要收集参考素材、定义全局角色与视觉风格、排布空间布局或分镜、生成海量候选、比对废弃方案、局部微调像素与代码，并在持续的用户反馈中不断迭代。这些散落的草稿、分支版本和失败尝试，并不是用完即弃的副产品，而是支撑长程创作成败的动态项目状态。

> ArXiv URL：https://arxiv.org/abs/2607.23588v1

遗憾的是，当前主流的 AI 工具链都在割裂这种状态。单次 Prompt 工具直接抛弃了上下文与探索历史；对话式智能体将极其复杂的二维多模态创作压缩进单维线性的聊天气泡中，既无法感知布局，也难以回溯局部修改；节点流工具虽然暴露了执行流，却高度依赖人工硬编码的刚性连线。商业系统如 Claude Design、Google Stitch 和各类故事板产品虽然隐约指向了“画布协同”的新形态，但其架构完全黑盒，学术界根本无法系统研究智能体如何维护外部上下文、如何回滚错误，以及如何在长流程中保持多模态资产的一致性。

针对这一缺失，开源项目 JarvisHub 给出了一种全新的解题视角：它彻底放弃以线性对话记录为核心的交互范式，提出了一套**以画布为原生载体（Canvas-Native）的开源多模态创作智能体运行底座**。在 JarvisHub 中，无限画布不仅是展示给人类的操作面板，更是智能体的外部记忆库、动作空间以及人机共享的演进状态图。通过将资产、版本、依赖关系与反馈显式建模为类型化图节点，JarvisHub 让智能体从“一次性工具调用者”转变为“具备长程规划与局部修复能力的协同创作者”。

<img src="/images/2607.23588v1/introduction.webp" alt="现有创作系统与 JarvisHub 的范式对比" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 画布不是UI组件，而是智能体的外部状态机

理解 JarvisHub 的设计哲学，关键在于弄清“长程创作任务的状态究竟存放在哪里”。在传统以 LLM 为核心的 Agent 体系中，上下文的载体默认是上下文窗口内的对话历史（Chat History）。然而，面对包含排版、分镜逻辑、代码联动和高分辨率图像的复杂创作任务，纯文本上下文会迅速遇到上下文污染、注意力退化和空间感知丢失的严重瓶颈。

JarvisHub 明确指出，人类设计师在创作时从来不依赖单向文字备忘录，而是依赖工作台、情绪板（Moodboard）和无限白板。因此，JarvisHub 将项目状态显式形式化为一个动态图结构 $\mathcal{C}_{t} = (\mathcal{G}_{t}, \mathbf{X}_{t}, \mathbf{M}_{t}, \mathbf{U}_{t}, \mathbf{L}_{t})$，其中图 $\mathcal{G}_{t} = (\mathcal{V}_{t}, \mathcal{E}_{t})$ 由具体节点和有向边构成。每一个节点 $v_i$ 都有全局唯一且可寻址的标识符、节点类别（如文本提示词、参考图像、生成视频、UI 组件或反馈批注）、空间坐标、内部元数据以及当前运行时状态。

这种将项目状态图直接投射到画布上的做法，带来了三项对长程创作至关重要的基础能力：

- **精准可寻址性**：智能体再也不需要通过“请把上一轮生成的第三张图里左上角的角色换掉”这种模糊的自然语言去定位资产，而是可以直接通过节点 ID 锁定精准的操作目标。

- **全生命周期可复用性**：被废弃的初期方案、早期的风格参考或是前序阶段的半成品，在画布上都作为明确的实体被保留下来，随时可以被拉入新的工作分支，成为下游工具调用的输入凭据。

- **依赖可回溯与局部修复**：通过节点之间的显式连线，系统清晰记录了某个资产究竟由哪个 Prompt 触发、参考了哪几张垫图、由哪款模型渲染。当用户要求修改局部设定时，智能体能够顺着依赖图只重算受影响的下游节点，而不是破坏全局重新生成。

<img src="/images/2607.23588v1/framework.webp" alt="JarvisHub 的三层系统架构与交互闭环" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

### 约束与审计：协议桥构建的安全执行防线

如果只是把画布提供给模型任意涂写，长程工作流很容易在数轮迭代后陷入混乱。由于多模态生成模型本身的随机性与不可靠性，智能体在操作复杂图结构时极易出现非法覆盖、误删关键依赖或陷入死循环等问题。为此，JarvisHub 在画布底层与顶层智能体之间架设了一道**协议桥（Protocol Bridge）**。

协议桥的核心职责是实现受控的运行时状态迁移。在每一个交互轮次 $t$，智能体接收到用户指令 $q_t$ 并观察当前画布状态 $\mathcal{C}_t$。但它并不能为所欲为，协议桥会根据当前项目的配置动态生成一份能力清单 $\Gamma_t$ 与权限授权 $\Omega_t$。智能体只能在被允许的操作空间内提议动作 $a_t$。这些动作被严格编码为检查过的工具调用、画布变更突变（Mutation）、评估打分请求或向人类发起澄清的信号。

只有通过协议桥语法与逻辑校验的操作，才会真正作用于画布，触发状态转移算子：




{% raw %}$$ \mathcal{C}_{t+1} = \mathcal{F}(\mathcal{C}_{t}, a_{t}, o_{t}, f_{t}, r_{t}) $${% endraw %}



这里的 $o_t$ 是外部工具执行后返回的多模态证据，$f_t$ 是引入的评价或人类反馈，而 $r_t$ 则代表自愈修复决策。如果一次图像生成失败或返回了违反约束的结果，协议桥会将失败状态明确记录在该节点上，并在轨迹中保留错误信息，促使智能体做出局部的重试或向用户寻求指引，而不是直接崩溃或假装成功。这种机制将原先不可控的黑盒语言推理，转化为了步步可审计、操作可撤销的确定性状态机。

<img src="/images/2607.23588v1/agent-loop.webp" alt="JarvisHub 的智能体执行循环与工具生态" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在执行引擎层面，JarvisHub 的智能体运行时解耦成了多个专业工具族，涵盖画布图拓扑操作、多模态媒体生成后端、原生代码环境执行、自愈恢复算子以及与 Anthropic 推出的模型上下文协议（Model Context Protocol, MCP）兼容的扩展能力。上层的子智能体（Sub-agents）与技能库（Skills）协同工作，既能拆解复杂目标，又能时刻将所有产物锚定在共享画布上。

### 贯通长程链路：三类高复杂度任务实战

为了检验这套画布原生架构在长程多模态创作中的实操表现，研究团队构建了三类极具挑战性的工业级长程创作场景，涵盖叙事媒体制作、交互式前端构建以及结构化演讲幻灯片排版。在实验配置中，系统统一接入主流前沿基座，由 GPT-5.5 负责规划主脑与智能体运行时调度，结合 GPT Image 2 与 Seedance 2.0 分别处理高保真图像与视频生成，并引入 Gemini 3.1 Pro 作为多模态评估裁判。

#### 1. 叙事媒体生成：跨镜头的视觉连续性保持

连续视觉叙事（如短剧、分镜脚本、动态漫画）被公认为多模态领域最具挑战的任务之一，核心难点在于如何让不同镜头中的角色外貌、服饰道具和场景光影保持严格一致。在传统 Chat 界面下，几轮对话过后模型就会彻底遗忘最初设定的面部特征。

<img src="/images/2607.23588v1/results1.webp" alt="叙事媒体生成的工作区轨迹" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在以丧尸末日短剧为原型的实验中，JarvisHub 展示了强大的上下文统御力。智能体首先在画布左侧生成剧本大纲与核心角色设定卡，建立明确的参考资产节点。在推进各个分镜时，每一张新镜头的生成操作都显式地将角色设定节点和前置分镜作为输入依赖拉入。当用户或评估模型指出某一镜头构图偏离时，智能体只需在画布上对对应节点执行局部重绘或重采样，而不会破坏后续镜头的叙事连贯性。

<img src="/images/2607.23588v1/results11.webp" alt="叙事媒体生成最终交付的关键帧资产" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### 2. 交互式前端开发：设计审美与功能代码的双重对齐

第二个典型场景是将一份宽泛的摄影作品集设计需求，转化为可交互、符合现代审美的实际网页。这不仅涉及前端代码编写，更包含视觉资产检索、版式比例权衡与多断点响应式预览。

<img src="/images/2607.23588v1/results2.webp" alt="交互式网页开发的工作区轨迹" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

在这一任务中，智能体把画布变成了动态的设计原型工坊。它将情绪参考图、调色板节点、CSS 样式规则以及实际渲染出的 HTML/JS 沙箱预览并列在画布各区域。智能体并非一次性盲写全量代码，而是一边生成资产，一边通过内嵌的无头浏览器执行截屏，将渲染结果作为观察证据写回画布，并由多模态裁判进行视线流动与排版合理性打分。这种“生成—渲染—回填画布—自查评估—局部修补”的闭环，让最终生成的网页不仅能顺畅运行，而且在字体排印、留白节奏和视觉风格上具备极高的专业水准。

<img src="/images/2607.23588v1/results22.webp" alt="交互式网页开发交付的多页面成品展示" style="width:min(1000px, calc(100vw - 2rem)); max-width:none; height:auto; margin:1.5rem auto; position:relative; left:50%; transform:translateX(-50%); display:block;">

#### 3. 演讲幻灯片生成：宏观结构与微观版式的统一

制作高水平的专业技术演示文稿（Presentation Deck），难点在于宏观层面的叙事逻辑必须清晰层层递进，微观层面的图表排布、强调色提取和图标风格又必须高度统一。



面对复杂的机器学习课件课题，JarvisHub 的智能体首先铺开全局提纲树状图，确立章节依赖；随后并发展开单页细化，生成配套的架构示意图、代码片段与排版样式节点。整个生成过程完全透明暴露在画布上，用户可以在任意环节拖拽节点调整页面次序，或者直接在节点上修改核心结论文字。智能体敏锐地捕获到这些人工干预后，能精准识别受影响的下游幻灯片，在保留既定配色规范的前提下完成自适应重排。



### 从“成果导向”到“轨迹导向”：JarvisHub 的深远价值

如果仅仅把 JarvisHub 视作一个更好用的多模态 AI 交互界面，就低估了这项研究的学术野心。论文作者在讨论部分强调了一个关键洞察：**在长程多模态创作研究中，最核心的分析客体不应当是单张最终产物，而是智能体的决策轨迹（Trajectory）。**

在以往的研究范式中，评价生成模型往往只看最终输出的图像美不美、视频画质高不高。然而在真实的复杂创作中，“生成了一张不错的图”并不能证明智能体具备长程创作能力——它可能完全无视了用户给出的参考图，可能随意抹除了用户刚刚确认的上一版设计，也可能在一个只需要微调文字字号的场景下鲁莽地将全图推倒重来。

JarvisHub 记录下的完整执行轨迹：




{% raw %}$$ \tau = \left\{ (q_t, \mathcal{C}_t, \Gamma_t, \Omega_t, a_t, o_t, f_t, r_t, \mathcal{C}_{t+1}) \right\}_{t=1}^{T} $${% endraw %}



为多模态智能体研究提供了前所未有的全景观测窗口。通过这套结构化的轨迹数据，研究人员第一次能够定性定量地审视过程层面的核心问题：

1. **多模态上下文一致性**：智能体在执行动作时，是否真正引用了画布上已有的正确前序节点？

2. **工具使用精准度**：在面对局部缺陷时，智能体是选择了代价低廉的局部修补工具，还是盲目调用全局生成？

3. **反馈遵从与自愈能力**：当外部评估打出低分或用户提出修改批注时，智能体做出的后续决策能否有效修复缺陷？

不仅如此，这类高质量的画布交互轨迹一旦经过脱敏、合规与质量筛选，将成为构建下一代“创作专用基座模型”的绝佳训练飞轮。现存的大模型指令微调数据几乎全部局限于一问一答的线性对话，完全缺乏在多状态、多资产拓扑图上进行长程统筹与错误修复的样本。JarvisHub 通过开源这套底座，实质上为社区铺设了一条生产高阶多模态 Agent 轨迹数据的流水线。

### 结语与未来演进

从 Midjourney 式的单次 Prompt 抽卡，到 ChatGPT 式的线性对话调试，再到今天以 JarvisHub 为代表的画布原生协同，多模态 AI 的演进路径正日益与专业创作者的真实工作习惯合流。JarvisHub 证明了：当智能体拥有了空间化的外部记忆与可编辑的共享状态图，它才能真正跨越单点素材生成的浅水区，承接具有工业价值的长程复杂创作。

当然，JarvisHub 目前仍处于初期框架阶段。当前呈现的任务更偏向定性能力的概念验证，尚未建立起标准化的全自动化 Benchmark 基准；同时，画布状态机虽然严格规范了动作格式，但依然无法兜底基座多模态模型可能出现的“审美漂移”或语义理解偏差。但毋庸置疑的是，随着 JarvisHub 的开源，多模态 Agent 的研究边界终于被推移到了对话框之外——在更广阔的二维画布上，人机协同创作的新范式才刚刚拉开序幕。
