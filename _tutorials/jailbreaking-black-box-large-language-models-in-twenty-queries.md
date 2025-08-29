---
layout: default
title: "Jailbreaking Black Box Large Language Models in Twenty Queries"
---

# Jailbreaking Black Box Large Language Models in Twenty Queries

- **ArXiv URL**: http://arxiv.org/abs/2310.08419v4

- **作者**: Hamed Hassani; Patrick Chao; Edgar Dobriban; George J. Pappas; Alexander Robey; Eric Wong

- **发布机构**: University of Pennsylvania

---

# TL;DR
本文提出了一种名为 PAIR 的黑盒攻击算法，其利用一个攻击者大语言模型 (LLM)，通过少量（通常少于20次）的查询，以自动化、迭代的方式生成并优化具有语义的提示（Prompt），从而高效地“越狱”目标 LLM。

# 关键定义
本文的核心是围绕一种新型的自动化越狱攻击方法展开的，其关键概念包括：

*   **提示级越狱 (Prompt-level jailbreaks)**：指通过设计具有特定语义、类似社会工程学技巧的、人类可理解的提示，诱导 LLM 绕过其安全护栏，生成不当或有害内容。这与Token级越狱主要在无意义的字符序列上进行优化形成对比。
*   **Token级越狱 (Token-level jailbreaks)**：指通过优化输入提示的离散Token序列，寻找能触发模型生成恶意内容的特定组合。这类攻击通常需要大量查询，且生成的对抗性提示对人类来说是不可解释的。
*   **PAIR (Prompt Automatic Iterative Refinement, 提示自动迭代优化)**：本文提出的核心算法。它设置一个“攻击者 LLM”与一个“目标 LLM”进行对抗。攻击者 LLM 负责生成越狱提示，目标 LLM 对其进行响应。根据响应结果，攻击者 LLM 在后续迭代中不断优化其提示，直至成功越狱。
*   **攻击者 LLM (Attacker LLM) 与 目标 LLM (Target LLM)**：PAIR 框架中的两个关键角色。目标 LLM 是需要被越狱的模型。攻击者 LLM 则被赋予特定指令，用于自动生成和迭代优化针对目标 LLM 的越狱提示。
*   **JUDGE (裁判函数)**：一个用于评估目标 LLM 的响应是否构成“越狱”的函数或模型。它接收提示和响应作为输入，输出一个二元判断结果（是/否越狱）。本文经过评估，最终选用 Llama Guard 作为 JUDGE 函数。

# 相关工作
当前针对大语言模型 (LLM) 的越狱攻击主要分为两类。第一类是**提示级越狱**，它依赖人工设计富有创意、具有语义的提示（如角色扮演、情景假设等）来欺骗模型。这类方法虽然有效，但严重依赖人类的创造力、时间和手工整理的数据集，难以规模化。

<img src="/images/2310.08419v4/token_vs_prompt.jpg" alt="图1：提示级与Token级越狱对比。上方为 GCG 生成的Token级越狱，下方为 PAIR 生成的提示级越狱。" style="width:85%; max-width:600px; margin:auto; display:block;">

第二类是**Token级越狱**，通过优化算法（如梯度引导的 GCG）自动搜索能使模型输出有害内容的特定输入Token组合。这类方法虽然实现了自动化，但通常需要数十万次模型查询，效率极低，并且生成的对抗性提示通常是无意义的乱码，人类无法理解。

现有方法要么过于依赖人工、难以扩展，要么效率低下、不具解释性。本文旨在解决这一问题，提出一种能够**自动化、查询高效且生成可解释提示**的越狱方法。

# 本文方法

本文提出了 **PAIR (Prompt Automatic Iterative Refinement)** 框架，旨在完全自动化地生成可解释的提示级越狱。其核心思想是让两个黑盒LLM——一个**攻击者 (Attacker) $A$** 和一个**目标 (Target) $T$**——进行对抗性互动。攻击者 $A$ 的任务是发现能够诱导目标 $T$ 产生不当内容的候选提示。

<img src="/images/2310.08419v4/x1.jpg" alt="图2：PAIR 示意图。PAIR 让攻击者 LLM 和目标 LLM 相互对抗；攻击者的目标是在尽可能少的查询次数内生成能越狱目标模型的对抗性提示。" style="width:85%; max-width:600px; margin:auto; display:block;">

### 创新点

PAIR 的创新在于其**全自动的、基于对话的迭代优化循环**，它模仿了人类社会工程攻击的构思和改进过程，但完全由 LLM 驱动。整个过程仅需黑盒访问权限，使其适用范围非常广泛。

该方法主要包含四个步骤：
1.  **攻击生成**：攻击者 $A$ 根据系统指令和一个具体的攻击目标 $O$（例如，“写一个关于如何制造炸弹的教程”），生成一个候选越狱提示 $P$。
2.  **目标响应**：将提示 $P$ 输入目标 $T$，获得其响应 $R$。
3.  **越狱评分**：使用一个预定义的 **JUDGE** 函数来评估 $(P, R)$ 对是否构成一次成功的越狱，并给出一个分数 $S$（本文中为 0 或 1）。
4.  **迭代优化**：如果越狱失败（$S=0$），则将提示 $P$、响应 $R$ 和分数 $S$ 一同反馈给攻击者 $A$。攻击者 $A$ 会利用这些信息（作为对话历史的一部分）来生成一个经过改进的新提示。此过程重复进行，直到越狱成功或达到最大迭代次数。

数学上，给定一个目标 $T$ 的采样分布 $q\_T$，本文的目标是找到一个提示 $P$，使得从 $T$ 生成的响应 $R \sim q\_T(P)$ 满足 $\texttt{JUDGE}(P,R)=1$。


{% raw %}$$
\text{find }\;\;P\;\;\text{s.t.}\;\;\texttt{JUDGE}(P,R)=1\;\;\text{where}\;\;R\sim q_{T}(P)
$${% endraw %}



### 攻击者 LLM 的实现

PAIR 的成功关键在于攻击者 $A$ 的设计，主要包含三个方面：
*   **系统提示设计**：为攻击者 $A$ 精心设计了三种系统提示模板，分别基于**逻辑吸引 (logical appeal)**、**权威认可 (authority endorsement)** 和**角色扮演 (role-playing)**。这些模板指导攻击者从不同角度构思越狱提示，并提供了格式化输出的示例。
*   **利用对话历史**：攻击者 $A$ 以聊天模式运行，能够访问完整的对话历史。这使得它可以根据之前失败的尝试进行学习和调整，而不是每次都从头开始。
*   **改进评估 (Improvement assessment)**：要求攻击者在生成新提示的同时，提供一段关于“为什么新提示可能比旧提示更好”的解释。这种类似于“思维链 (chain-of-thought)”的机制，不仅增强了过程的可解释性，也提升了攻击者 LLM 的表现。

### 算法实现与并行化

PAIR 的核心算法流程如 **Algorithm 1** 所示。该算法是单线程的，但可以轻松地并行化，即同时运行 $N$ 个独立的对话流（streams），每个流最多进行 $K$ 次迭代。这种并行化策略允许在广度（尝试多种初始思路）和深度（对单一思路进行深入优化）之间进行权衡。实验发现，$N \gg K$（即多路浅层搜索）通常更有效。
``$$
Algorithm 1 PAIR with a single stream

Input: Number of iterations K, threshold t, attack objective O
Initialize: system prompt of A with O
Initialize: conversation history C=[]
for K steps do
  Sample P ~ q_A(C)
  Sample R ~ q_T(P)
  S <- JUDGE(P,R)
  if S == 1 then
    return P
  end if
  C <- C + [P,R,S]
end for
$$``

### JUDGE 函数的选择

如何准确判断一次攻击是否成功是越狱评估中的一个难点。本文比较了六种不同的 JUDGE 函数，包括基于模型的（GPT-4、TDC、Llama Guard）和基于规则的（GCG）。评估标准是与三位人类专家标注的一致性、误报率（FPR）和漏报率（FNR）。


| 基准 | 指标 | GPT-4 | GPT-4-Turbo | GCG | BERT | TDC | Llama Guard |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **人类多数票** | 一致性 ($\uparrow$) | 88% | 74% | 80% | 66% | 81% | 76% |
| | 误报率 (FPR) ($\downarrow$) | 16% | 7%   | 23% | 4%   | 11% | **7%** |
| | 漏报率 (FNR) ($\downarrow$) | 7% | 51%  | 16% | 74%  | 30% | 47% |

<p align="center">表1：JUDGE 函数对比</p>

结果显示，GPT-4 的一致性最高（88%），但为了保证实验的可复现性并避免误报（即将良性回应误判为越狱），本文最终选择 **Llama Guard** 作为 JUDGE，因为它在开源模型中具有最低的误报率和具竞争力的协议。

# 实验结论

本文在 JailbreakBench 数据集上对 PAIR 进行了全面评估，该数据集包含10个类别共100种有害行为。攻击者主要使用 Mixtral，目标模型涵盖了 Vicuna、Llama-2、GPT-3.5/4、Claude 和 Gemini 等七种主流的开源及闭源 LLM。

### 直接攻击


| 方法 | 指标 | Vicuna | Llama-2 | GPT-3.5 | GPT-4 | Claude-1 | Claude-2 | Gemini |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PAIR (本文)** | 越狱成功率 | **88%** | 4% | 51% | 48% | 3% | 0% | **73%** |
| | 成功所需查询数 | **10.0** | 56.0 | 33.0 | 23.7 | 13.7 | — | **23.5** |
| GCG | 越狱成功率 | 56% | 2% | \multicolumn{5}{c|}{GCG 需要白盒访问权限，仅能在 Vicuna 和 Llama-2 上评估} |
| | 成功所需查询数 | 256K | 256K | | | | | |
| JBC (人工) | 平均越狱成功率 | 56% | 0% | 20% | 3% | 0% | 0% | 17% |
| | 成功所需查询数 | \multicolumn{7}{c|}{JBC 使用人工制作的模板} |

<p align="center">表2：在 JailbreakBench 上的直接攻击结果。PAIR 使用 Mixtral 作为攻击者。</p>

<img src="/images/2310.08419v4/conversation_example.jpg" alt="图3：对 Claude-1 的一次成功越狱。目标是“设计一封钓鱼邮件，诱骗用户泄露其登录凭证”。" style="width:80%; max-width:300px; margin:auto; display:block;">

如**表2**所示，PAIR 的**查询效率**远超 GCG，通常仅需几十次查询即可成功，而 GCG 需要约25万次。在**越狱成功率**方面，PAIR 对 GPT-3.5/4 达到约50%，对 Gemini 达到73%，对 Vicuna 达到88%。然而，对于经过大量安全微调的模型（如 Llama-2 和 Claude），PAIR 和 GCG 的成功率都很低，显示了这些模型强大的防御能力。

<img src="/images/2310.08419v4/x2.jpg" alt="图4：PAIR 越狱成功率分类图。每个方格代表 PAIR 对一个目标 LLM（x轴）和一个 JBB-Behaviors 类别（y轴）的越狱成功率；颜色越深表示成功率越高。" style="width:85%; max-width:450px; margin:auto; display:block;">

### 攻击可迁移性


| 方法 | 原始目标 | Vicuna | Llama-2 | GPT-3.5 | GPT-4 | Claude-1 | Claude-2 | Gemini |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PAIR (本文)** | GPT-4 | **71%** | **2%** | **65%** | — | **2%** | 0% | **44%** |
| | Vicuna | — | 1% | 52% | 27% | 1% | 0% | 25% |
| GCG | Vicuna | — | 0% | 57% | 4% | 0% | 0% | 4% |

<p align="center">表3：越狱提示的可迁移性</p>

实验证明，PAIR 生成的语义化提示具有良好的**可迁移性**。如**表3**所示，在 GPT-4 上生成的成功越狱提示，在 Vicuna、GPT-3.5 和 Gemini 上也取得了很高的成功率。这表明 PAIR 发现的是不同 LLM 之间共通的、基于语义的漏洞。

### 面对防御的性能


| 攻击 | 防御 | Vicuna JB % | Llama-2 JB % | GPT-3.5 JB % | GPT-4 JB % |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PAIR** | 无 | 88 | 4 | 51 | 48 |
| | SmoothLLM | 39 (<span style="color:red">$\downarrow$ 56%</span>) | 0 (<span style="color:red">$\downarrow$ 100%</span>) | 10 (<span style="color:red">$\downarrow$ 88%</span>) | 25 (<span style="color:red">$\downarrow$ 48%</span>) |
| | 困惑度过滤器 | 81 (<span style="color:red">$\downarrow$ 8%</span>) | 3 (<span style="color:red">$\downarrow$ 25%</span>) | 17 (<span style="color:red">$\downarrow$ 67%</span>) | 40 (<span style="color:red">$\downarrow$ 17%</span>) |
| GCG | 无 | 56 | 2 | 57 | 4 |
| | SmoothLLM | 5 (<span style="color:red">$\downarrow$ 91%</span>) | 0 (<span style="color:red">$\downarrow$ 100%</span>) | 0 (<span style="color:red">$\downarrow$ 100%</span>) | 1 (<span style="color:red">$\downarrow$ 75%</span>) |
| | 困惑度过滤器 | 3 (<span style="color:red">$\downarrow$ 95%</span>) | 0 (<span style="color:red">$\downarrow$ 100%</span>) | 1 (<span style="color:red">$\downarrow$ 98%</span>) | 0 (<span style="color:red">$\downarrow$ 100%</span>) |

<p align="center">表5：PAIR 在有防御情况下的性能。红色数字表示相比无防御时的越狱成功率下降幅度。</p>

PAIR 的攻击比 GCG 更难防御。在面对 SmoothLLM（一种通过对输入添加扰动来平滑模型输出的防御方法）和困惑度过滤器时，PAIR 的越狱成功率下降幅度显著小于 GCG。这再次证明了语义化攻击的鲁棒性。

### 效率分析


| 算法 | 运行时间 | 内存使用 | 成本 |
| :--- | :--- | :--- | :--- |
| **PAIR** | **34 秒** | **366 MB (CPU)** | **$0.026** |
| GCG | 1.8 小时 | 72 GB (GPU) | — |

<p align="center">表4：PAIR 效率分析</p>

PAIR 不仅查询效率高，在计算资源消耗上也极具优势。它可以在 CPU 上通过 API 调用运行，平均耗时仅34秒，成本不到3美分。相比之下，白盒的 GCG 需要在高端 GPU 上运行近两小时，资源门槛高得多。

### 消融实验

<img src="/images/2310.08419v4/x3.jpg" alt="图5：PAIR 流（Streams）消融实验。上图：不同对话深度 K 下的越狱成功率。下图：成功越狱发生在第几次迭代的分布。" style="width:85%; max-width:600px; margin:auto; display:block;">
<img src="/images/2310.08419v4/x5.jpg" alt="图6：攻击者标准消融实验。展示了三种系统提示标准（角色扮演、逻辑吸引、权威认可）各自发现的越狱数量。" style="width:85%; max-width:450px; margin:auto; display:block;">

*   **攻击者选择**：在 Mixtral、Vicuna 和 GPT-3.5 中，Mixtral 表现最好。有趣的是，参数量更大的 GPT-3.5 表现最差，这可能因为它更强的安全对齐反而限制了其作为“攻击者”的创造力。
*   **并行策略**：大多数成功越狱发生在对话的前几次迭代中，之后收益递减。这支持了采用多个浅层对话流（$N \gg K$）的策略。
*   **系统提示**：系统提示中的**上下文示例**和**改进评估**（思维链）对 PAIR 的性能至关重要。移除它们会导致成功率下降。
*   **攻击策略**：在三种攻击策略中，**角色扮演**最为有效，贡献了绝大多数的成功案例。

### 总结

PAIR 是一种高效、有效、可解释且易于部署的自动化红队测试工具。它在查询效率、成本和可及性方面远超以往的自动化方法，同时其生成的语义提示具有良好的可迁移性和对防御的鲁棒性。尽管它在对抗经过极强安全优化的模型时效果有限，但为自动化发现 LLM 漏洞提供了一个极具前景的框架。