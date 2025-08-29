---
layout: default
title: "Educational data mining and learning analytics: An updated survey"
---

# Educational data mining and learning analytics: An updated survey

- **ArXiv URL**: http://arxiv.org/abs/2402.07956v1

- **作者**: C. Romero; S. Ventura

- **发布机构**: University of Cordoba

---

# TL;DR
本文是一篇关于教育数据挖掘（EDM）和学习分析（LA）领域的更新版综述，系统性地回顾了该领域的关键术语、发展历程、知识发现流程、主要应用环境、方法论、工具集以及未来发展趋势，旨在为研究人员和实践者提供一份全面的最新知识图谱。

# 引言
随着电子学习资源、教学软件和学生信息数据库的普及，教育领域积累了海量数据。手动分析这些数据已不可能，因此需要自动化工具来发掘其中蕴含的巨大教育价值，以理解学生如何学习，并帮助教育机构做出更好的决策。

在此背景下，两个密切相关的研究社区应运而生：
*   **教育数据挖掘 (Educational Data Mining, EDM)**: 专注于为来自教育环境的独特数据类型开发探索性方法，应用数据挖掘（Data Mining, DM）技术解决重要的教育问题，更侧重于技术挑战和发现新模式。
*   **学习分析 (Learning Analytics, LA)**: 定义为对学习者及其背景数据进行测量、收集、分析和报告，旨在理解和优化学习及其环境。它更侧重于教育挑战、数据驱动决策和应用已知模型。

尽管两者侧重点不同，但目标和方法高度重叠。此外，还涌现出一系列相关术语：
*   **学术分析 (Academic Analytics, AA)** 和 **机构分析 (Institutional Analytics, IA)**: 关注课程、学位项目、科研、财务等宏观层面，以提供机构层面的洞察。
*   **教学分析 (Teaching Analytics, TA)**: 从教师视角分析教学活动与表现。
*   **数据驱动教育 (Data-Driven Education, DDE)** 和 **数据驱动决策 (Data-Driven Decision Making in Education, DDDM)**: 指系统性地收集和分析数据，以改善学生和学校的成功。
*   **教育中的大数据 (Big Data in Education, BDE)**: 应用大数据技术处理教育数据。
*   **教育数据科学 (Educational Data Science, EDS)**: 综合运用统计学、数据分析和机器学习等方法解决教育问题。

<img src="/images/2402.07956v1/page_2_Figure_0.jpg" alt="EDM/LA相关的主要领域" style="width:85%; max-width:600px; margin:auto; display:block;">

本综述是对2013年同名综述的更新与改进，反映了过去几年该领域的指数级增长、新术语的出现、对新兴教育环境（如MOOCs、虚拟现实）的关注、工具与数据集的丰富以及研究前沿的演进。

# 背景
EDM和LA领域分别由两个独立的国际会议（IEDM和LAK）推动发展。下表列出了该领域最相关的会议。


| 标题 | 缩写 | 类型 | 首次年份 |
| --- | --- | --- | --- |
| 国际教育人工智能会议 | AIED | 双年 | 1982 |
| 国际智能辅导系统会议 | ITS | 双年 | 1988 |
| IEEE国际先进学习技术会议 | ICALT | 年度 | 2000 |
| 欧洲技术增强学习会议 | EC-TEL | 年度 | 2006 |
| 国际教育数据挖掘会议 | EDM | 年度 | 2008 |
| 国际用户建模、自适应与个性化会议 | UMAP | 年度 | 2009 |
| 国际学习分析与知识会议 | LAK | 年度 | 2011 |
| 大规模学习会议 | L@S | 年度 | 2014 |
| 学习与学生分析会议 | LSAC | 年度 | 2017 |

<center>表1. 与EDM/LA最相关的会议</center>

自2006年第一本相关著作出版以来，该领域的书籍数量不断增加，尤其在近年，“学习分析”成为书名中最常用的术语。其中，两本里程碑式的著作是《教育数据挖掘手册》和《学习分析手册》。

<details>
<summary>表2. 已出版的EDM/LA相关书籍（部分）</summary>


| 封面 | 标题 | 作者 | 年份 | 出版社 |
| --- | --- | --- | --- | --- |
|  | Data Mining in Education | C. Romero & S. Ventura | 2006 | Wit Press |
|  | Handbook of Educational Data Mining | C. Romero, S.Ventura., M.Pechenizky, R. Baker | 2010 | CRC Press, Taylor & Francis Group |
|  | Education Data Mining: Applications and Trends | A. PeñaAyala | 2014 | Springer |
|  | Learning Analytics: From research to practice. | J.A.Larusson, B.White | 2014 | Springer |
|  | Data Science in Higher Education: A Step-by-Step Introduction to Machine Learning for Institutional Researchers | J. Lawson | 2015 | CreateSpace |
|  | Big Data and Learning Analytics in Higher Education: Current Theory and Practice | B.k. Daniel | 2016 | Springer |
|  | Data Mining and Learning Analytics: Applications in Educational Research | S. ElAtia, D.Ipperciel, O.R. Zaïane | 2016 | Wiley |
|  | Big Data in Education: The digital future of learning, policy and practice | B.Williamson | 2017 | SAGE Publications |
|  | Learning Analytics Explained | Niall Sclater | 2017 | Routledge |
|  | Handbook of Learning Analytics | C. Lang, G.Siemens, A.Wise, D.Gašević | 2017 | SOLAR |
|  | Learning Analytics Goes to School | A. Krumm ,B. Means, M.Bienkowski | 2018 | Routledge |
|  | Learning Analytics in the Classroom | J. Horvath, J.Lodge, L.Corrin | 2018 | Routledge |
|  | The Analytics Revolution in Higher Education | J. S.Gagliardi, et al. | 2018 | Stylus Publishing |
|  | Learning Analytics in Education | D. Niemi, et al. | 2018 | Information Age Publishing |
|  | Learning Analytics in Higher Education | J. Lester, et al. | 2018 | Routledge |
|  | Utilizing Learning Analytics to Support Study Success | D. Ifenthaler, et al. | 2019 | Springer |
|  | Machine Learning Paradigms: Advances in Learning Analytics | M. Virvou, et al. | 2019 | Springer |
|  | Utilizing Educational Data Mining Techniques for Improved Learning | C. Bhatt, et al. | 2019 | IGI Global |

</details>

该领域的研究成果主要发表在《Journal of Learning Analytics》和《Journal of Educational Data Mining》等专业期刊上。

<details>
<summary>表3. EDM/LA领域顶级相关期刊</summary>


| 期刊标题 | 论文数 | 影响因子 (2018) | 是否免费开放 |
| --- | --- | --- | --- |
| Journal of Learning Analytics | 143 | - | 是 |
| Computers and Education | 81 | 5.627* | 否 |
| British Journal of Educational Technology | 65 | 2.588 ** | 否 |
| Journal of Educational Data Mining | 48 | - | 是 |
| Journal of Artificial Intelligence in Education | 47 | - | 否 |
| IEEE Transactions on Learning Technologies | 33 | 2.315* | 否 |
| Journal of Computer Assisted Learning | 32 | 2.451 ** | 否 |
| International Journal on Technology Enhanced Learning | 31 | - | 否 |
| User Modeling and User-Adapted Interaction | 27 | 3.400* | 否 |
| Internet and Higher Education | 26 | 5.284** | 否 |
| Computer Applications in Engineering Education | 26 | 1.435 * | 否 |

</details>

如下图所示，从2000年到2018年，Google Scholar中“Educational Data Mining”和“Learning Analytics”的论文数量呈指数级增长，后者自2011年起在数量上超越前者，显示出该领域蓬勃发展的态势。

<img src="/images/2402.07956v1/page_8_Figure_0.jpg" alt="EDM和LA领域的论文数量年度变化趋势" style="width:85%; max-width:600px; margin:auto; display:block;">

高被引论文（见下表）中，综述类文章占比较高，表明对该领域进行系统性梳理和总结的需求十分重要。

<details>
<summary>表4. EDM/LA领域Top-10高被引论文</summary>


| 论文标题 | 参考文献 | Google Scholar引用 | Scopus引用 |
| --- | --- | --- | --- |
| Educational data mining: A survey from 1995 to 2005 | (Romero and Ventura 2007) | 1489 | 662 |
| Educational data mining: a review of the state of the art | (Romero and Ventura 2010a) | 1367 | 631 |
| The state of educational data mining in 2009: A review and future visions | (Baker and Yacef 2009) | 1199 | - |
| Penetrating the fog: Analytics in learning and education | (Siemens and Long, 2011) | 1138 | - |
| Data mining in course management systems: Moodle case study and tutorial | (Romero et al., 2008) | 1105 | 470 |
| Learning analytics: drivers, developments and challenges | (Ferguson, 2012) | 691 | 328 |
| Learning analytics and educational data mining: towards communication and collaboration | (Siemens and Baker 2012) | 589 | 224 |
| Course signals at Purdue: Using learning analytics to increase student success | (Arnold and Pistilli, 2012) | 569 | 206 |
| Translating learning into numbers: A generic framework for learning analytics | (Greller and Drachsler, 2012) | 547 | 221 |
| Mining educational data to analyze students' performance | (Baradwaj and Pal, 2012) | 543 | - |

</details>

# EDM/LA 知识发现周期
应用EDM/LA的过程遵循通用的知识发现与数据挖掘 (Knowledge Discovery and Data Mining, KDD) 流程，但在每个环节都具有教育领域的独特性。

<img src="/images/2402.07956v1/page_9_Figure_4.jpg" alt="EDM/LA知识发现流程图" style="width:85%; max-width:600px; margin:auto; display:block;">

### 教育环境
数据的来源和类型取决于教育环境，如传统课堂、基于计算机的学习或混合式学习，以及所使用的信息系统，如学习管理系统 (Learning Management System, LMS)、智能辅导系统 (Intelligent Tutoring System, ITS)、大规模开放在线课程 (Massive Open Online Course, MOOC) 等。

### 教育数据
教育数据来源广泛，包括师生与教育系统的交互数据（如导航行为、测验输入、论坛发帖）、管理数据（如学校和教师信息）、人口统计数据（如性别、年龄）以及学生情感数据（如动机、情绪状态）。这些数据具有多源、多格式、多粒度（从按键级到学校级）的特点，如下图所示。

<img src="/images/2402.07956v1/page_10_Figure_4.jpg" alt="教育数据的多层次结构" style="width:90%; max-width:700px; margin:auto; display:block;">

### 预处理
数据预处理是耗时且复杂的关键步骤，通常占据整个项目一半以上的时间。它包括：
*   **特征工程 (Feature engineering)**：生成和选择有信息量的学生相关变量。
*   **数据转换/离散化**：将连续属性转换为分类属性，以增强模型的可解释性。
*   **数据匿名化**：删除个人敏感信息，保护学生隐私，并遵循伦理准则。

### 方法与技术
大多数传统的数据挖掘技术，如可视化、分类、聚类和关联分析，已成功应用于教育领域。同时，教育数据的层次化和纵向性等特点也催生了特定的处理方法。

### 新知识的解释与应用
EDM/LA的最终目标是采取行动。发现的知识必须被教师和管理者用于干预和决策，以改善学生学习。因此，模型的可解释性至关重要，“白盒”模型（如决策树）通常优于“黑盒”模型（如神经网络）。可视化技术和推荐系统也是向非专业用户（如教师、学生）呈现结果和建议的有效工具。

# 教育环境与数据
教育环境多种多样，每种环境都提供不同的数据源。

<img src="/images/2402.07956v1/page_12_Figure_0.jpg" alt="教育环境分类图" style="width:85%; max-width:600px; margin:auto; display:block;">

### 传统面授教育
这是最普遍的教育形式，主要依赖师生间的面对面互动。其数据源包括学生出勤、成绩、课程信息等。这些传统系统也可以结合计算机系统作为辅助工具。

<center>表5. 国际教育标准分类</center>


| 等级 | 主要特征 |
| --- | --- |
| 学前教育 | 面向3岁至小学入学年龄的儿童设计。 |
| 初等教育或基础教育第一阶段 | 通常从5-7岁开始。 |
| 初中教育或基础教育第二阶段 | 为8-14岁儿童设计，完成基础教育。 |
| 高中教育 | 从15或16岁开始的更专业的教育。 |
| 高中后非高等教育 | 介于高中和高等教育之间的项目。 |
| 高等教育第一阶段 | 教育内容比之前阶段更高级的课程。 |
| 高等教育第二阶段 | 授予高级研究资格（如博士学位）的课程。 |


### 基于计算机的教育系统
指使用计算机进行教学或管理教学的系统。随着互联网和人工智能技术的发展，涌现出大量新型的、基于网络的智能教育系统。

<center>表6. 基于计算机的教育系统示例</center>


| 系统 | 描述 |
| --- | --- |
| 自适应和智能超媒体系统 (AIHS) | 通过构建个体学生模型（目标、偏好、知识）来提供自适应内容。 |
| 智能辅导系统 (ITS) | 通过对学生行为建模，提供直接的、定制化的指导或反馈。 |
| 学习管理系统 (LMS) | 提供课程交付、管理、跟踪和报告等功能的软件包。 |
| 大规模开放在线课程 (MOOC) | 支持大量参与者的网络课程，参与人数无上限。 |
| 测试和测验系统 | 主要通过一系列问题来衡量学生的知识水平。 |
| 其他类型 | 可穿戴学习系统、学习对象库、概念图、社交网络、严肃游戏、虚拟/增强现实系统等。 |

### 混合式学习系统
混合式学习 (Blended Learning, BL) 环境将面授教学与计算机辅助教学相结合，提供了更大的灵活性和便利性。这类系统的数据源涵盖了前两种环境的数据。

# 工具与数据集
尽管存在许多通用的数据挖掘工具（如Rapidminer、Weka），但它们对教育工作者来说使用门槛较高。因此，一些专门为EDM/LA设计的软件工具应运而生，但它们通常功能特定，只能处理特定类型的数据和问题。

<center>表7. EDM/LA专用工具示例</center>


| 名称 | URL | 描述 |
| --- | --- | --- |
| DataShop | https://pslcdatashop.web.cmu.edu/ | 提供研究数据存储库和一系列分析报告工具。 |
| GISMO | http://gismo.sourceforge.net/ | 为教师提供在线课程中学生活动的可视化监控。 |
| Inspire | https://moodle.org/plugins/tool_inspire | Moodle的分析API，提供描述性和预测性分析引擎。 |
| LOCO Analyst | http://jelenajovanovic.net/LOCO-Analyst/ | 为教师提供网络学习环境中学习过程相关方面的反馈。 |
| Meerkat ED | http://www.reirab.com/MeerkatED | 分析学生在协作学习工具支持的课程中的活动。 |
| MDM Tool | http://www.uco.es/kdis/research/software/ | 在Moodle 2.7中应用数据挖掘技术的框架。 |
| Performance Plus | https://www.d2l.com/higher-education/products/performance/ | 为管理者、教育者和学习者提供强大的分析工具。 |
| SNAPP | ... | 可视化论坛帖子和回复中形成的交互网络。 |
| Solutionpath StREAM | https://www.solutionpath.co.uk/ | 利用预测模型实时确定学生参与度的方方面面。 |

除了使用自有数据，研究者也可以利用一些公开的数据集。然而，目前公开的数据集数量有限，且大多来自电子学习系统。建立一个类似UCI机器学习库的EDM/LA专用数据集仓库将非常有价值。

<center>表8. EDM/LA公开数据集示例</center>


| 数据集 | URL | 描述 |
| --- | --- | --- |
| ASSISTments Competition Dataset | https://sites.google.com/view/assistmentsdatamining/home | 利用真实世界教育数据预测重要的纵向结果。 |
| Canvas Network dataset | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/1XORAL | 来自Canvas Network开放课程的匿名数据。 |
| DataShop | https://pslcdatashop.web.cmu.edu/index.jsp?datasets=public | 主要存储和提供来自ITS的研究数据。 |
| Educational Process Mining Dataset | ... | 数字电子学仿真环境中的学生会话日志。 |
| HarvardX-MITx dataset | https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/26147 | MITx和HarvardX在edX平台第一年MOOC课程的匿名数据。 |
| KDD Cup 2010 Dataset | https://pslcdatashop.web.cmu.edu/KDDCup/ | 根据ITS交互日志预测学生在数学问题上的表现。 |
| LearnMoodle dataset | https://research.moodle.net/158/ | 来自learn.moodle.net某课程的匿名数据。 |
| Open University Learning Analytics Dataset | https://analyse.kmi.open.ac.uk/open_dataset | 包含七门课程的学生及其与Moodle交互的数据。 |
| Student Performance Dataset | https://archive.ics.uci.edu/ml/datasets/Student+Performance | 葡萄牙两所中学学生的学业成就数据。 |
| xAPI-Educational Mining Dataset | https://www.kaggle.com/aljarah/xAPI-Edu-Data | 从名为Kalboard 360的电子学习系统收集的学生学业表现数据。 |

# 方法与应用
## 分类体系
本文从**方法**、**用户**和**应用主题**三个维度对EDM/LA领域进行了梳理和分类。

### 1. 按方法目标分类
EDM/LA领域采用的方法多种多样，本文根据其核心目标和关键应用，将其归纳为以下几类。这个分类体系是理解该领域技术版图的核心。


| 方法 | 目标/描述 | 关键应用 |
| --- | --- | --- |
| **因果挖掘 (Causal Mining)** | 发现数据中的因果关系或识别因果效应。 | 发现哪些学生行为特征导致了学习、学业失败或辍学等。 |
| **聚类 (Clustering)** | 识别相似观察值的群体。 | 根据学习和互动模式对相似的学习材料或学生进行分组。 |
| **基于模型的发现 (Discovery with models)** | 将一个已验证的现象模型作为另一项分析的组成部分。 | 识别学生行为、特征与情境变量之间的关系。 |
| **为人判断提炼数据 (Distillation of data for human judgment)** | 使用摘要、可视化和交互界面以易于理解的方式呈现数据。 | 帮助教师可视化和分析学生的持续活动和信息使用情况。 |
| **知识追踪 (Knowledge tracing)** | 估计学生对技能的掌握程度。 | 随时间推移监控学生的知识状态。 |
| **非负矩阵分解 (Non-negative matrix factorization)** | 将包含学生测试结果的正数矩阵分解为项目矩阵和技能掌握矩阵。 | 评估学生技能。 |
| **离群点检测 (Outlier detection)** | 指出显著不同的个体。 | 检测有困难或学习过程异常的学生。 |
| **预测 (Prediction)** | 从其他变量的组合中推断目标变量，包括分类、回归等。 | 预测学生表现和检测学生行为。 |
| **过程挖掘 (Process mining)** | 从事件日志中获取过程知识。 | 基于学生在教育系统中的演化轨迹来反映其行为。 |
| **推荐 (Recommendation)** | 预测用户可能对某个项目给出的评分或偏好。 | 向学生推荐活动、任务、链接、练习题或课程等。 |
| **关系挖掘 (Relationship mining)** | 研究变量间的关系并编码成规则，包括关联规则、序列模式等。 | 识别学习者行为模式中的关系，诊断学生困难。 |
| **统计学 (Statistics)** | 计算描述性统计和推断性统计。 | 分析、解释教育数据并从中得出结论。 |
| **社交网络分析 (Social network analysis)** | 分析网络信息中实体间的社会关系。 | 解释协作活动和交流工具互动中的结构与关系。 |
| **文本挖掘 (Text mining)** | 从文本中提取高质量信息。 | 分析论坛、聊天、网页和文档的内容。 |
| **可视化 (Visualization)** | 以图形方式呈现数据。 | 生成数据可视化，帮助向教育者传达EDM/LA研究成果。 |

<center>表9. 最流行的EDM/LA方法</center>

### 2. 按用户/利益相关者分类
不同的用户群体对EDM/LA有着不同的目标和诉求。


| 用户/利益相关者 | 目标 |
| --- | --- |
| **学习者/学生** | 了解自身需求，寻求改善学习体验和表现的方法。 |
| **教育者/教师** | 理解学习过程，寻求改进教学方法。 |
| **科研人员** | 开发和评估EDM/LA技术的有效性。 |
| **管理者/学术权威** | 负责为机构内的实施分配资源。 |

<center>表10. 用户/利益相关者及其目标示例</center>

### 3. 按当前热点应用主题分类
除了上述分类，本文还列举了当前EDM/LA研究社群中一些最受关注的热点话题和应用方向。

<details>
<summary>表11. EDM/LA研究的当前热点应用或主题</summary>


| 兴趣主题 | 描述 |
| --- | --- |
| 分析教育理论 | 将学习理论与学习分析相结合进行教育研究。 |
| 分析教学策略 | 使用EDM/LA技术分析和探索教学策略的应用与效果。 |
|分析编程代码 | 专注于分析编程课程中的代码、作业提交等。 |
| 协作学习与团队合作 | 分析协作学习，预测团队合作中的小组评分。 |
| 课程挖掘/分析 | 分析课程结构、成绩和管理数据，以改进课程开发和项目质量。 |
| 仪表盘与可视化学习分析 | 应用可视化技术探索和理解用户在（在线）环境中的轨迹。 |
| 深度学习 | 在EDM/LA研究中应用多层处理单元的神经网络架构。 |
| 发现因果关系 | 在教育数据集中寻找属性间的因果关系。 |
| 预警系统 | 尽早预测学生表现和风险学生，以便及早干预。 |
| 情感学习分析 | 研究学习过程中的情感及其对学习的重要性。 |
| 评估干预效果 | 评估干预措施、数据驱动的学生反馈和可操作建议的有效性。 |
| 特征工程方法 | 使用机器学习技术自动构建属性或学生特征。 |
| 游戏学习分析 | 对严肃游戏中的玩家互动应用数据挖掘和可视化技术。 |
| 可解释的学习者模型 | 开发“白盒”、可解释、可用且高度易懂的学习者模型。 |
| 外语学习 | 应用EDM/LA技术改进外语学习。 |
| 测量自我调节学习 | 应用EDM/LA技术测量学生的自我调节学习特征和行为。 |
| 多模态学习分析 | 利用机器学习和传感器技术，提供跨多个情境的新型学习洞见。 |
| 编排学习分析 | 研究学习分析在课堂层面的采纳、实践影响及其他因素。 |
| 提供个性化反馈 | 自动或半自动生成个性化反馈以支持学生学习。 |
| 情感发现 | 自动识别学习者和学习资源中潜在的态度、情感和主观性。 |
| 迁移学习 | 开发可迁移或应用于其他相似课程/机构的模型。 |
| 理解导航路径 | 从电子学习系统的事件日志中发现与过程相关的知识和导航学习路径。 |
| 写作分析 | 对来自论坛、聊天、论文等的文本数据应用文本挖掘和分析工具。 |

</details>

# 结论与未来趋势
EDM和LA作为交叉学科领域，在过去二十年中迅速发展，已成为一个日益成熟的领域，并逐渐从实验室走向市场。尽管研究在分析的深度和复杂性上取得了进展，但对实践、理论和框架的实际影响仍然有限。这需要研究组织、资助机构和合作项目的共同推动，以促进其从探索性模型向更全面的系统级研究转变。

关于未来的发展趋势，本文提出以下几点：

1.  **开发更通用和可移植的工具**：尽管已有一些专用工具，但仍需开发能够在一个界面中执行多种任务、解决不同教育问题的通用EDM/LA工具。同时，需要提高从这些工具中获得的模型的可移植性。
2.  **培养数据驱动的决策文化**：大多数教育机构仍未充分认识到分析大规模学生学习数据所带来的好处。需要建立新的领导模式来推动系统性变革，并促进数据驱动的教学决策文化。
3.  **克服规模化应用的挑战**：在高等教育机构中推广学习分析面临七大类挑战：
    *   **目标与收益 (Purpose and gain)**：必须明确LA项目的目标，并使其透明化。
    *   **呈现与行动 (Representation and actions)**：为学习者反馈选择合适的环境，并使用正确的可是化技术提供建议。
    *   **数据 (Data)**：制定与组织核心原则一致的数据政策，确保数据安全、隐私和使用的透明度。
    *   **IT基础设施 (IT infrastructure)**：尽早规划和建立必要的内部或外部IT基础设施。
    *   **开发与运营 (Development and operation)**：可扩展性是一个经常被低估的问题，需要在项目初期就予以考虑。