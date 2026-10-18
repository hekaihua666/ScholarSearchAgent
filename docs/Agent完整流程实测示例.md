# Agent 完整流程实测示例

实测时间：2026-10-18 17:05；检索模式：标准；真实模型和论文接口，首次计算未命中响应缓存。此文记录实际执行过程，不是预设示意流程。

## 1. 输入问题

> 2022-2025年图神经网络用于分子性质预测的代表性论文有哪些？

原问题贯穿检索过程。模型可以选择关键词搜索、语义搜索、引文扩展和结束；程序负责执行工具、过滤年份、融合去重、排序与展示。

本次标准模式上限为8次模型决策、10次工具尝试、120秒；实际只用了4次模型决策。一次模型决策可以提出多个工具调用，程序逐项执行并回传结果。

## 2. 实际动作

| 模型决策 | 实际工具与检索方向 | 工具返回记录 | 去重后新增 | 当前候选总数 |
| --- | --- | ---: | ---: | ---: |
| 第1次 | 关键词：graph neural network molecular property prediction | 25 | 23 | 23 |
| 第1次 | 语义：representative graph neural network architectures for molecular property prediction benchmarks 2022-2025 | 11 | 8 | 31 |
| 第2次 | 关键词：survey benchmark graph neural networks molecular property prediction review | 20 | 12 | 43 |
| 第2次 | 语义：self-supervised pretraining graph transformer molecular representation learning foundation model property prediction | 14 | 13 | 56 |
| 第3次 | 语义：3D geometric equivariant graph neural network molecular property prediction quantum chemistry dataset | 13 | 11 | 67 |
| 第4次 | finish_search：模型请求结束，结束信息格式超限，进入保底结束路径 | — | — | 67 |

每次工具执行后，程序将真实论文编号、标题、年份、摘要片段、已有操作和剩余预算传回模型。模型再决定接下来做什么。表中的方向是对实际检索式的解释，不代表读取了模型的私有推理。

本次没有选择引文扩展；系统不要求每次都使用所有工具。5次实际检索返回83条记录，融合去重后保留67条候选，年份均在2022—2025年。

## 3. 模型怎样请求调用工具

第一次决策包含如下真实调用（为便于阅读，参数从原生协议字符串展开）：

```json
{
  "name": "search_papers",
  "arguments": {
    "query": "graph neural network molecular property prediction",
    "limit": 25
  }
}
```

模型提出工具名和参数，程序校验后执行检索，再将返回的真实数据作为工具消息交回模型。模型文本本身不会被直接作为论文记录展示。

## 4. 本次如何结束

实际耗时约31.4秒，4次模型决策、5次检索动作，另有1次结束工具尝试。此次结束由模型主动发起，未耗尽标准模式预算。

本次还出现一个实际问题：结束调用给出的证据编号数量超过20个的接口上限，结束信息未通过校验。程序保留已检索的真实候选，记录为 `model_finished`，页面提示“本次检索已结束，研究条件的完整覆盖仍需核对。”不能把这个结果描述成严格验证了全部研究要求。

## 5. 排序与网页展示

候选在收集过程中按标题、摘要与问题的匹配、被引信息、检索来源和多路径召回等信号进行融合评分，最终按评分展示。页面提供摘要、年份、来源、原文入口以及排序、翻页、引用导出。

本次结果前列包含以下记录，标题和年份来自实际接口返回：

- Graph Neural Networks for the Prediction of Molecular Structure–Property Relationships（2023）。该记录的DOI对应书籍章节，应在人工筛选时核对文献类型。
- Applying graph neural network models to molecular property prediction using high-quality experimental data（2024）。
- GeomGCL: Geometric Graph Contrastive Learning for Molecular Property Prediction（2022）。
- Hierarchical Molecular Graph Self-Supervised Learning for property prediction（2023）。
- Molecular property prediction based on graph structure learning（2024）。

这67条是文献候选，代表性、相关性和文献类型仍需人工筛选；本次没有阅读全文或完成召回率评测。

网页检查返回HTTP 200，展示67条候选，无JavaScript异常。截图阶段复用了刚刚算出的真实结果缓存，页面已标明历史结果，未再发起一次付费模型检索。

## 6. 答辩讲解示例

> 用户输入研究问题后，系统让模型根据问题选择检索工具。每次工具返回真实论文，系统都会更新候选集合，把结果反馈给模型，由模型决定继续补查还是结束。本次示例中，模型先查基本方法，再补查综述、预训练和三维几何方向，最终得到67条去重后的文献候选。系统将结果排序，供用户查看摘要、访问原文和导出引用。

注意将“模型结束检索”和“已经找全论文”区分开。此次结束说明格式超限，实际已保留结果并提示完整性待核对。

## 证据文件

- `tmp/agent-flow-demo/response.json`：完整内部响应。
- `tmp/agent-flow-demo/audit.json`：实际工具轨迹、计数和停止原因。
- `tmp/agent-flow-demo/review.json`：本次检查、前五条与原问题解析。
- `tmp/agent-flow-demo/web-response.json`：网页实际接收的数据。
- `tmp/agent-flow-demo/browser-check.json`：网页检查结果。
- `tmp/agent-flow-demo/results.png`：网页实际截图。
