# 2026 学术检索高手项目与前沿研究调研

调研时间：2026-08-24

## 1. 重点参考对象

### Ai2 Asta Paper Finder

- 代码：[allenai/asta-paper-finder](https://github.com/allenai/asta-paper-finder)
- 不是单一检索式，而是先把查询转成结构化约束，再由执行规划器路由到“特定论文、作者、广泛主题、引文关系”等工作流。
- 提供 fast 与 diligent 两种执行预算；候选论文经过独立重排和逐篇 LLM 相关性判断。
- 对“最早、最有影响力、某时间之前”等元数据条件采用专门排序，而不是固定使用同一权重。

可吸收经验：意图路由、执行预算、相关性判定与元数据排序必须分层。

### Ai2 ScholarQA / PaperFinder

- 代码：[allenai/ai2-scholarqa-lib](https://github.com/allenai/ai2-scholarqa-lib)
- 论文：[Ai2 Scholar QA: Organized Literature Synthesis with Attribution](https://aclanthology.org/2025.acl-demo.49/)
- 默认先召回约 256 个段落与关键词结果，再用 `mxbai-rerank-large-v1` 重排，保留约 50 个证据段落。
- 生成过程不是一次回答，而是分解、组织、写作三个阶段；每个结论都附论文原文片段。

可吸收经验：候选召回要宽，精排要晚；结果页应展示证据，不应只展示模型推荐理由。

### OpenScholar

- 论文：[Nature 2026：Synthesizing scientific literature with retrieval-augmented language models](https://www.nature.com/articles/s41586-025-10072-4)
- 在 ScholarQABench 上进行多领域文献检索与综合评测，强调检索、引用和长答案综合的整体质量。
- 论文报告 OpenScholar-8B 在其多论文综合任务中超过 GPT-4o 与 PaperQA2，说明领域检索与证据组织比单纯增大模型更重要。

可吸收经验：比赛项目不能只展示“搜到了哪些论文”，还应展示研究脉络、证据覆盖和可验证综合。

### PaperQA2

- 代码：[Future-House/paper-qa](https://github.com/Future-House/paper-qa)
- 把工作流拆为论文搜索、证据收集、引文遍历和答案生成；重点能力包含证据评分、引用定位和矛盾检测。

可吸收经验：论文卡片应区分元数据、摘要证据和模型判断，避免把生成文本伪装成论文证据。

### ByteDance PaSa

- 代码：[bytedance/pasa](https://github.com/bytedance/pasa)
- 论文：[PaSa: An LLM Agent for Comprehensive Academic Paper Search](https://arxiv.org/abs/2501.10120)
- 使用 Crawler 负责查询和引文扩展，Selector 逐篇判断相关性；可进行多层参考文献扩展。
- 在 AutoScholarQuery 上训练，并使用 RealScholarQuery 检验真实研究人员查询。

可吸收经验：搜索代理与论文判别器应分工；引文扩展必须由高质量种子驱动，不能无条件扩散。

### SAGE 与 SIGIR 2026 检索结论

- [SAGE: Benchmarking and Improving Retrieval for Deep Research Agents](https://arxiv.org/abs/2602.05975)发现，在深度研究代理产生关键词型子查询时，BM25 可明显超过部分大模型稠密检索器；其核心原因是查询表达与检索器训练目标不匹配。
- [Revisiting Text Ranking in Deep Research](https://eprints.gla.ac.uk/383313/)指出，词项、学习型稀疏和多向量检索适合代理生成的查询；段落级检索更节省上下文，重排对最终质量非常重要。

可吸收经验：不能用“语义检索替代关键词检索”，而应做稀疏与语义双通道互补，并记录每条通道的贡献。

### 工程型优秀项目

- [paper-pilot](https://github.com/aytzey/paper-pilot)：六数据源、开放获取 PDF、证据片段、引文图和 Zotero 同步。
- [ScholarRAG](https://github.com/Wh1esky/ScholarRAG)：多粒度切块、Dense + Sparse + BM25、交叉编码器重排和进度式界面。
- [ThesisAgent](https://github.com/Integration-Automation/ThesisAgent)：多数据源检索后导出 PPT、Excel 与 BibTeX。

## 2. 与研问智搜原系统的差距

1. 原系统只有固定的双轮流程，不区分“找首篇论文”和“找最新综述”等意图。
2. 原系统主要展示综合分数和推荐理由，缺少可逐字回查的摘要证据。
3. 原系统只有一个 Top-20 列表，没有奠基—发展—桥接—前沿的研究全景。
4. 原系统只有原始查询语义召回，没有对第二轮改写查询再次进行语义检索。
5. 原系统只有一个运行预算，不适合现场快速演示和赛后深度研究两种场景。

## 3. 本轮形成的原创组合：TRACE-Rank v4 + TRACE-Loop v2

### 3.1 Intent-TRACE 意图自适应排序

先把问题路由为：特定论文、奠基工作、前沿发现、主题综述、方法对比、技术演化或开放主题。不同意图动态调整历史影响、时效性、标题假设、语义召回和引文来源的权重。

### 3.2 Dual-Semantic Follow-up

保留 OpenAlex 词项相关性通道，同时使用 GTE-Large 语义通道检索原始问题；完成盲区分析后，再对第二轮改写查询执行语义召回。它不是简单增加一个向量搜索，而是把语义检索放入代理反馈环。

### 3.3 Verifiable Evidence Matrix

从真实摘要中选择与查询约束重合最高的句子，保存原文字符区间、命中词和证据强度。界面以矩阵展示每篇论文对研究主题、方法、任务和领域约束的覆盖度。

### 3.4 Research-role Landscape

结合发表时间、年均引用、多检索路径命中和引文来源，把论文划分为奠基、发展、桥接和前沿四类。该视图用于解释“这篇论文在研究脉络中的位置”，而不是再生成一段笼统摘要。

### 3.5 Budgeted Search Modes

- 快速：3 条核心检索式、1 条标题探针、1 次语义召回，不执行引文扩展。
- 均衡：完整双轮词项检索、原问题语义召回、1 次二轮语义追检和按需引文扩展。
- 深度：在均衡模式上增加第二条语义追检，适合综述与系统调研。

### 3.6 API Key Pool Failover

多个合法配置的 OpenAlex Key 组成额度池。系统按请求轮换，检测到单 Key 当日额度耗尽后自动摘除，不再让整条检索链降级到低质量数据源。

## 4. 创新边界

本项目不声称训练了新的基础检索模型。创新集中在可复现的系统算法：意图路由、双通道反馈检索、预算控制、证据矩阵、研究角色建模和可观测指标。所有展示论文仍由真实学术 API 返回；LLM 生成的标题假设必须经过学术索引验证后才能进入结果。
