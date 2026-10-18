# 论文修订说明

Nature-writing 路由：`algorithmic`；章节为 `title + intro + related work + method + experiments + discussion + conclusion`；来源语言为中文；目标格式为通用竞赛论文。

## 一句话论证

在复杂学术查询中，研问智搜通过多源召回、TRACE 分层排序和冻结可复现评测实现可运行检索闭环；dev20 与独立 validation30 表明语义检索和标题探测提供稳定边际收益、主要瓶颈仍是候选召回，同时验证结果限制了对 HELIOS-Lite 的质量提升主张。

## 术语表

- `TRACE-Rank`：多路径融合与约束覆盖排序模块。
- `TRACE-Loop`：查询规划、扩展与多轮检索流程。
- `HELIOS-Lite`：训练无关的约束缺口/成本感知动作控制器。
- `dev20`：修复 arXiv 日期协议后的 20 条开发查询。
- `validation30`：与 dev20 不重叠的 30 条固定未见验证查询。
- `LIVE/FROZEN/REPLAY`：外部抓取、不可变快照、离线确定性重放。

## Claim--evidence map

| 主张 | 证据 | 状态 |
|---|---|---|
| 当前主要瓶颈是 retrieval | dev20 候选命中 21/58，Top20 命中 18；37 篇检索缺口、3 篇排序缺口 | supported |
| semantic 与 title probe 有稳定边际收益 | validation30 候选 gold 分别 +11、+8 | supported |
| balanced 是当前质量 champion | validation30 四项 Top20 质量指标均高于 efficient/HELIOS | supported |
| efficient 是无损低成本模式 | API -38.9%，但 Recall@20 -0.0642 | rejected |
| HELIOS-Lite 提升检索质量 | validation30 未保持，且 30 条全部 stop | rejected |
| HELIOS-Lite 已实现且可解释/可消融 | 生产模式、trace、单元测试与消融均已完成 | supported |

## 仍需补充

1. 赛方正式论文模板、篇幅和引用格式尚未提供；当前采用通用中文论文结构。
2. 三个 validation30 模式是独立 LIVE 抓取，严格配对因果对比仍需在同一候选快照上补做。
3. 若要写“优于现有系统”，必须取得相同数据、相同 cutoff、相同身份匹配下的可复现实测结果；当前稿件没有作此主张。

