# Selector 数据审计

| 项目 | train500 | calibration150 |
|---|---:|---:|
| 合法 gold | 1212 | 390 |
| materialized positives | 1212 | 390 |
| 原生产/代理候选已召回正例 | 833 | 262 |
| metadata 缺失 | 0 | 0 |
| 日期排除 | 10 | 3 |
| identity 失败 | 0 | 0 |

- 旧 v4 冻结 production50 只有 35 个已召回正例；本轮 train500 合法正例全部物化。
- 30 个训练查询、20 个校准查询直接使用冻结生产 Top60；其余查询使用同一份缓存元数据语料构建固定 TF-IDF Top60 hard-negative proxy，没有新增 API/LLM 调用。
- 非 gold 候选统一解释为 `benchmark-negative`，不解释为现实世界确定不相关。
- split overlap：`{'train_calibration': 0, 'train_dev': 0, 'train_validation': 0, 'train_blind': 0, 'calibration_dev': 0, 'calibration_validation': 0, 'calibration_blind': 0, 'dev_blind': 0}`；dev20、validation30、blind50 的 gold 均未读取或进入训练。
- `gold_materialized` 仅用于数据审计，`FEATURE_NAMES=['query_title_coverage', 'query_abstract_coverage', 'title_jaccard', 'all_text_jaccard', 'required_constraint_coverage', 'preferred_constraint_coverage', 'relation_compatibility', 'method_task_dataset_coverage', 'time_validity', 'venue_type_compatibility', 'abstract_missing', 'trace_relative', 'lexical_relative', 'llm_relevance', 'llm_missing', 'source_count_saturation', 'best_channel_reciprocal_rank', 'rank_percentile', 'score_gap_to_query_top', 'metadata_completeness']` 中不含该字段或任何标签派生字段。
