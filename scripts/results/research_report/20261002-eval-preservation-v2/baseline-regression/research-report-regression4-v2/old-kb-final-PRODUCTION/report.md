# 评测报告：PRODUCTION

- 模式：retrieval
- 样本数：4（含模糊配对；检索主指标看下面的独立正例）
- 完成/失败：4/0

## 独立正例（explicit + cross_doc）

不含 vague 配对题，也不含负样本

- 样本数：4

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.2500 | 100.0% |
| context_required_fact_coverage | 0.0000 | 100.0% |
| document_recall_at_5 | 0.2500 | 100.0% |
| evidence_group_recall_at_5 | 0.0000 | 100.0% |
| exact_candidate_recall_at_20 | N/A | 0.0% |
| exact_candidate_recall_at_50 | N/A | 0.0% |
| exact_candidate_recall_at_actual_depth | N/A | 0.0% |
| fact_mrr | 0.0000 | 100.0% |
| mrr | 0.0000 | 50.0% |
| precision_at_5 | 0.0000 | 50.0% |
| quote_coverage_at_5 | 0.0000 | 100.0% |
| recall_at_5 | 0.0000 | 50.0% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.0000 | 100.0% |

## 分层

| 层 | n | 指标 | 均值 | 覆盖率 |
|---|---:|---|---:|---:|
| explicit | 4 | all_required_documents_at_5 | 0.2500 | 100.0% |
| explicit | 4 | context_required_fact_coverage | 0.0000 | 100.0% |
| explicit | 4 | document_recall_at_5 | 0.2500 | 100.0% |
| explicit | 4 | evidence_group_recall_at_5 | 0.0000 | 100.0% |
| explicit | 4 | exact_candidate_recall_at_20 | N/A | 0.0% |
| explicit | 4 | exact_candidate_recall_at_50 | N/A | 0.0% |
| explicit | 4 | exact_candidate_recall_at_actual_depth | N/A | 0.0% |
| explicit | 4 | fact_mrr | 0.0000 | 100.0% |
| explicit | 4 | mrr | 0.0000 | 50.0% |
| explicit | 4 | precision_at_5 | 0.0000 | 50.0% |
| explicit | 4 | quote_coverage_at_5 | 0.0000 | 100.0% |
| explicit | 4 | recall_at_5 | 0.0000 | 50.0% |
| explicit | 4 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| explicit | 4 | structured_required_fact_coverage_at_5 | 0.0000 | 100.0% |

## 全量均值（含模糊配对，不作为检索主指标）

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.2500 | 100.0% |
| context_required_fact_coverage | 0.0000 | 100.0% |
| document_recall_at_5 | 0.2500 | 100.0% |
| evidence_group_recall_at_5 | 0.0000 | 100.0% |
| exact_candidate_recall_at_20 | N/A | 0.0% |
| exact_candidate_recall_at_50 | N/A | 0.0% |
| exact_candidate_recall_at_actual_depth | N/A | 0.0% |
| fact_mrr | 0.0000 | 100.0% |
| mrr | 0.0000 | 50.0% |
| precision_at_5 | 0.0000 | 50.0% |
| quote_coverage_at_5 | 0.0000 | 100.0% |
| recall_at_5 | 0.0000 | 50.0% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.0000 | 100.0% |

## 检索侧客观指标（Recall@5 / MRR 等）

块级 Recall/MRR 按 chunk id 精确匹配，文档 Recall 按 source 匹配；
quote/required-fact coverage 按归一化后的块内文字与数值边界匹配。
与上面的 RAGAS judge 指标分开读。

| 指标 | 均值 |
|---|---:|
| all_required_documents_at_5 | 0.2500 |
| context_required_fact_coverage | 0.0000 |
| document_recall_at_5 | 0.2500 |
| evidence_group_recall_at_5 | 0.0000 |
| exact_candidate_recall_at_20 | N/A |
| exact_candidate_recall_at_50 | N/A |
| exact_candidate_recall_at_actual_depth | N/A |
| fact_mrr | 0.0000 |
| mrr | 0.0000 |
| precision_at_5 | 0.0000 |
| quote_coverage_at_5 | 0.0000 |
| recall_at_5 | 0.0000 |
| required_fact_coverage_at_5 | 0.0000 |
| structured_required_fact_coverage_at_5 | 0.0000 |

## 检索阶段诊断

| 阶段字段 | 均值 |
|---|---:|
| coarse_candidate_count | 150.0000 |
| coarse_source_count | 1.0000 |
| coarse_duplicate_source_slots | 149.0000 |
| rerank_candidate_count | 50.0000 |
| rerank_source_count | 1.0000 |
| rerank_duplicate_source_slots | 49.0000 |
| final_candidate_count | 1.2500 |
| final_source_count | 0.2500 |
| final_duplicate_source_slots | 1.0000 |
- 全部 resolved fusion modes：{"weighted": 1}；cliff enabled：0
- 全部 dense evidence survival recall：N/A；覆盖率：0.0%
- 全部 fusion evidence survival recall：0.0000；覆盖率：25.0%
- 全部 rerank evidence survival recall：0.0000；覆盖率：25.0%
- 全部 final_top5 evidence survival recall：0.0000；覆盖率：100.0%
- 独立正例 resolved fusion modes：{"weighted": 1}；cliff enabled：0
- 独立正例 dense evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 fusion evidence survival recall：0.0000；覆盖率：25.0%
- 独立正例 rerank evidence survival recall：0.0000；覆盖率：25.0%
- 独立正例 final_top5 evidence survival recall：0.0000；覆盖率：100.0%
- explicit resolved fusion modes：{"weighted": 1}；cliff enabled：0
- explicit dense evidence survival recall：N/A；覆盖率：0.0%
- explicit fusion evidence survival recall：0.0000；覆盖率：25.0%
- explicit rerank evidence survival recall：0.0000；覆盖率：25.0%
- explicit final_top5 evidence survival recall：0.0000；覆盖率：100.0%

- sparse survival：unavailable（Milvus hybrid API 未返回内部独立榜单；使用 L0/L1 跨层对比）。
- coarse/rerank chunk IDs 与逐条 evidence survival 见 records.jsonl / per_query.csv。

## 运行可复现配置

- dataset_fingerprint：46b060b267470cba54de49c0a2241927c03286d58296cb5ed931bd030f5e5e1d
- kb_fingerprint：25f3f726000cf48467a663347d984133c954058dad115debe4a1395b7886d17e
- metrics_version：2.0
- annotation_schema_versions：['2.0']
- pdf_set_fingerprint：ddc1e68c63656b46110cd06179ac3e674a0986d37c9f4871a81973104927740b
- chunk_content_fingerprint：22d55430b5ba0f355e71b5e4aca39ddba26d2257a09fef124aeb71f5a4e6f1ff
- parser_clean_chunk_rules：11184507f70129696792593fcdd360dee8dea7849bb350bc2f1a53fa82bbd406
- code_worktree_sha256：360e7f922bafd83018fc53077be441508fc54fe8e57320569f09d6d24f74f8d3
- code_diff_sha256：f4706b849b4ef0e94b9673d3f6ac72c37a3dbcc6f912108b274eb030fad729b5
- embedding_model：BAAI/bge-m3
- embedding_version：BAAI/bge-m3
- embedding_dim：1024
- embedding_pooling：cls
- chunk_max_tokens：256
- chunk_overlap_tokens：25
- fusion_mode：weighted
- dense_weight：0.8
- sparse_weight：0.2
- git_head：41ca0191f9f014535ecd3c452bb85c66a6f206f2
- git_dirty：True
