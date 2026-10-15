# 评测报告：L2

- 模式：retrieval
- 样本数：18（含模糊配对；检索主指标看下面的独立正例）
- 完成/失败：18/0

## 独立正例（explicit + cross_doc）

不含 vague 配对题，也不含负样本

- 样本数：18

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| document_recall_at_5 | 0.7685 | 100.0% |
| exact_candidate_recall_at_20 | 0.3796 | 100.0% |
| exact_candidate_recall_at_50 | 0.3796 | 100.0% |
| mrr | 0.1389 | 100.0% |
| precision_at_5 | 0.0333 | 100.0% |
| quote_coverage_at_5 | 0.0278 | 100.0% |
| recall_at_5 | 0.1389 | 100.0% |
| required_fact_coverage_at_5 | 0.4074 | 100.0% |

- 父块还原 parent_restore_rate：1.0000 （命中 3 块，还原 3，未替换 0）

## 分层

| 层 | n | 指标 | 均值 | 覆盖率 |
|---|---:|---|---:|---:|
| cross_doc | 10 | document_recall_at_5 | 0.5833 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_20 | 0.2833 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_50 | 0.2833 | 100.0% |
| cross_doc | 10 | mrr | 0.1000 | 100.0% |
| cross_doc | 10 | precision_at_5 | 0.0200 | 100.0% |
| cross_doc | 10 | quote_coverage_at_5 | 0.0500 | 100.0% |
| cross_doc | 10 | recall_at_5 | 0.0500 | 100.0% |
| cross_doc | 10 | required_fact_coverage_at_5 | 0.2333 | 100.0% |
| explicit | 8 | document_recall_at_5 | 1.0000 | 100.0% |
| explicit | 8 | exact_candidate_recall_at_20 | 0.5000 | 100.0% |
| explicit | 8 | exact_candidate_recall_at_50 | 0.5000 | 100.0% |
| explicit | 8 | mrr | 0.1875 | 100.0% |
| explicit | 8 | precision_at_5 | 0.0500 | 100.0% |
| explicit | 8 | quote_coverage_at_5 | 0.0000 | 100.0% |
| explicit | 8 | recall_at_5 | 0.2500 | 100.0% |
| explicit | 8 | required_fact_coverage_at_5 | 0.6250 | 100.0% |

## 父块还原（全部命中证据块）

- rate：1.0000
- hits / restored / not_replaced / missing / no_parent：3 / 3 / 0 / 0 / 0

## 全量均值（含模糊配对，不作为检索主指标）

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| document_recall_at_5 | 0.7685 | 100.0% |
| exact_candidate_recall_at_20 | 0.3796 | 100.0% |
| exact_candidate_recall_at_50 | 0.3796 | 100.0% |
| mrr | 0.1389 | 100.0% |
| precision_at_5 | 0.0333 | 100.0% |
| quote_coverage_at_5 | 0.0278 | 100.0% |
| recall_at_5 | 0.1389 | 100.0% |
| required_fact_coverage_at_5 | 0.4074 | 100.0% |

## 检索侧客观指标（Recall@5 / MRR 等）

块级 Recall/MRR 按 chunk id 精确匹配，文档 Recall 按 source 匹配；
quote/required-fact coverage 按归一化后的块内文字与数值边界匹配。
与上面的 RAGAS judge 指标分开读。

| 指标 | 均值 |
|---|---:|
| document_recall_at_5 | 0.7685 |
| exact_candidate_recall_at_20 | 0.3796 |
| exact_candidate_recall_at_50 | 0.3796 |
| mrr | 0.1389 |
| precision_at_5 | 0.0333 |
| quote_coverage_at_5 | 0.0278 |
| recall_at_5 | 0.1389 |
| required_fact_coverage_at_5 | 0.4074 |

## 检索阶段诊断

| 阶段字段 | 均值 |
|---|---:|
| coarse_candidate_count | 15.0000 |
| coarse_source_count | 2.6111 |
| coarse_duplicate_source_slots | 12.3889 |
| rerank_candidate_count | N/A |
| rerank_source_count | N/A |
| rerank_duplicate_source_slots | N/A |
| final_candidate_count | 5.0000 |
| final_source_count | 1.7778 |
| final_duplicate_source_slots | 3.2222 |
- 全部 resolved fusion modes：{"weighted": 18}；cliff enabled：0
- 全部 dense evidence survival recall：N/A；覆盖率：0.0%
- 全部 fusion evidence survival recall：0.3796；覆盖率：100.0%
- 全部 rerank evidence survival recall：N/A；覆盖率：0.0%
- 全部 final_top5 evidence survival recall：0.1389；覆盖率：100.0%
- 独立正例 resolved fusion modes：{"weighted": 18}；cliff enabled：0
- 独立正例 dense evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 fusion evidence survival recall：0.3796；覆盖率：100.0%
- 独立正例 rerank evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 final_top5 evidence survival recall：0.1389；覆盖率：100.0%
- cross_doc resolved fusion modes：{"weighted": 10}；cliff enabled：0
- cross_doc dense evidence survival recall：N/A；覆盖率：0.0%
- cross_doc fusion evidence survival recall：0.2833；覆盖率：100.0%
- cross_doc rerank evidence survival recall：N/A；覆盖率：0.0%
- cross_doc final_top5 evidence survival recall：0.0500；覆盖率：100.0%
- explicit resolved fusion modes：{"weighted": 8}；cliff enabled：0
- explicit dense evidence survival recall：N/A；覆盖率：0.0%
- explicit fusion evidence survival recall：0.5000；覆盖率：100.0%
- explicit rerank evidence survival recall：N/A；覆盖率：0.0%
- explicit final_top5 evidence survival recall：0.2500；覆盖率：100.0%

- sparse survival：unavailable（Milvus hybrid API 未返回内部独立榜单；使用 L0/L1 跨层对比）。
- coarse/rerank chunk IDs 与逐条 evidence survival 见 records.jsonl / per_query.csv。

## 运行可复现配置

- dataset_fingerprint：3fe3ac6664c5df04286efb955c6dda88f82111857acaad6aa211e83c3af77d4e
- kb_fingerprint：25f3f726000cf48467a663347d984133c954058dad115debe4a1395b7886d17e
- embedding_model：BAAI/bge-m3
- embedding_version：BAAI/bge-m3
- embedding_dim：1024
- embedding_pooling：cls
- chunk_max_tokens：256
- chunk_overlap_tokens：25
- fusion_mode：weighted
- dense_weight：0.7
- sparse_weight：0.30000000000000004
- git_head：41ca0191f9f014535ecd3c452bb85c66a6f206f2
- git_dirty：True
