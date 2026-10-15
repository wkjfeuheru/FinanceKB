# 评测报告：PRODUCTION

- 模式：retrieval
- 样本数：20（含模糊配对；检索主指标看下面的独立正例）
- 完成/失败：20/0

## 独立正例（explicit + cross_doc）

不含 vague 配对题，也不含负样本

- 样本数：18

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.3889 | 100.0% |
| context_required_fact_coverage | 0.3796 | 100.0% |
| document_recall_at_5 | 0.5000 | 100.0% |
| evidence_group_recall_at_5 | 0.4074 | 100.0% |
| exact_candidate_recall_at_20 | 0.5015 | 61.1% |
| exact_candidate_recall_at_50 | 0.7625 | 44.4% |
| exact_candidate_recall_at_actual_depth | 0.8045 | 61.1% |
| fact_mrr | 0.3366 | 100.0% |
| mrr | 0.3333 | 100.0% |
| negative_rejection | N/A | 0.0% |
| precision_at_5 | 0.1667 | 100.0% |
| quote_coverage_at_5 | 0.3358 | 100.0% |
| recall_at_5 | 0.3093 | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.3796 | 100.0% |

## 分层

| 层 | n | 指标 | 均值 | 覆盖率 |
|---|---:|---|---:|---:|
| cross_doc | 10 | all_required_documents_at_5 | 0.4000 | 100.0% |
| cross_doc | 10 | context_required_fact_coverage | 0.4833 | 100.0% |
| cross_doc | 10 | document_recall_at_5 | 0.6000 | 100.0% |
| cross_doc | 10 | evidence_group_recall_at_5 | 0.5333 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_20 | 0.6062 | 80.0% |
| cross_doc | 10 | exact_candidate_recall_at_50 | 0.7286 | 70.0% |
| cross_doc | 10 | exact_candidate_recall_at_actual_depth | 0.7312 | 80.0% |
| cross_doc | 10 | fact_mrr | 0.4058 | 100.0% |
| cross_doc | 10 | mrr | 0.4000 | 100.0% |
| cross_doc | 10 | negative_rejection | N/A | 0.0% |
| cross_doc | 10 | precision_at_5 | 0.2200 | 100.0% |
| cross_doc | 10 | quote_coverage_at_5 | 0.4044 | 100.0% |
| cross_doc | 10 | recall_at_5 | 0.3567 | 100.0% |
| cross_doc | 10 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| cross_doc | 10 | structured_required_fact_coverage_at_5 | 0.4833 | 100.0% |
| explicit | 8 | all_required_documents_at_5 | 0.3750 | 100.0% |
| explicit | 8 | context_required_fact_coverage | 0.2500 | 100.0% |
| explicit | 8 | document_recall_at_5 | 0.3750 | 100.0% |
| explicit | 8 | evidence_group_recall_at_5 | 0.2500 | 100.0% |
| explicit | 8 | exact_candidate_recall_at_20 | 0.2222 | 37.5% |
| explicit | 8 | exact_candidate_recall_at_50 | 1.0000 | 12.5% |
| explicit | 8 | exact_candidate_recall_at_actual_depth | 1.0000 | 37.5% |
| explicit | 8 | fact_mrr | 0.2500 | 100.0% |
| explicit | 8 | mrr | 0.2500 | 100.0% |
| explicit | 8 | negative_rejection | N/A | 0.0% |
| explicit | 8 | precision_at_5 | 0.1000 | 100.0% |
| explicit | 8 | quote_coverage_at_5 | 0.2500 | 100.0% |
| explicit | 8 | recall_at_5 | 0.2500 | 100.0% |
| explicit | 8 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| explicit | 8 | structured_required_fact_coverage_at_5 | 0.2500 | 100.0% |
| negative | 2 | all_required_documents_at_5 | N/A | 0.0% |
| negative | 2 | context_required_fact_coverage | N/A | 0.0% |
| negative | 2 | document_recall_at_5 | N/A | 0.0% |
| negative | 2 | evidence_group_recall_at_5 | N/A | 0.0% |
| negative | 2 | exact_candidate_recall_at_20 | N/A | 0.0% |
| negative | 2 | exact_candidate_recall_at_50 | N/A | 0.0% |
| negative | 2 | exact_candidate_recall_at_actual_depth | N/A | 0.0% |
| negative | 2 | fact_mrr | N/A | 0.0% |
| negative | 2 | mrr | N/A | 0.0% |
| negative | 2 | negative_rejection | 0.0000 | 100.0% |
| negative | 2 | precision_at_5 | N/A | 0.0% |
| negative | 2 | quote_coverage_at_5 | N/A | 0.0% |
| negative | 2 | recall_at_5 | N/A | 0.0% |
| negative | 2 | required_fact_coverage_at_5 | N/A | 0.0% |
| negative | 2 | structured_required_fact_coverage_at_5 | N/A | 0.0% |

## 全量均值（含模糊配对，不作为检索主指标）

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.3889 | 90.0% |
| context_required_fact_coverage | 0.3796 | 90.0% |
| document_recall_at_5 | 0.5000 | 90.0% |
| evidence_group_recall_at_5 | 0.4074 | 90.0% |
| exact_candidate_recall_at_20 | 0.5015 | 55.0% |
| exact_candidate_recall_at_50 | 0.7625 | 40.0% |
| exact_candidate_recall_at_actual_depth | 0.8045 | 55.0% |
| fact_mrr | 0.3366 | 90.0% |
| mrr | 0.3333 | 90.0% |
| negative_rejection | 0.0000 | 10.0% |
| precision_at_5 | 0.1667 | 90.0% |
| quote_coverage_at_5 | 0.3358 | 90.0% |
| recall_at_5 | 0.3093 | 90.0% |
| required_fact_coverage_at_5 | 0.0000 | 90.0% |
| structured_required_fact_coverage_at_5 | 0.3796 | 90.0% |

## 检索侧客观指标（Recall@5 / MRR 等）

块级 Recall/MRR 按 chunk id 精确匹配，文档 Recall 按 source 匹配；
quote/required-fact coverage 按归一化后的块内文字与数值边界匹配。
与上面的 RAGAS judge 指标分开读。

| 指标 | 均值 |
|---|---:|
| all_required_documents_at_5 | 0.3889 |
| context_required_fact_coverage | 0.3796 |
| document_recall_at_5 | 0.5000 |
| evidence_group_recall_at_5 | 0.4074 |
| exact_candidate_recall_at_20 | 0.5015 |
| exact_candidate_recall_at_50 | 0.7625 |
| exact_candidate_recall_at_actual_depth | 0.8045 |
| fact_mrr | 0.3366 |
| mrr | 0.3333 |
| negative_rejection | 0.0000 |
| precision_at_5 | 0.1667 |
| quote_coverage_at_5 | 0.3358 |
| recall_at_5 | 0.3093 |
| required_fact_coverage_at_5 | 0.0000 |
| structured_required_fact_coverage_at_5 | 0.3796 |

## 检索阶段诊断

| 阶段字段 | 均值 |
|---|---:|
| coarse_candidate_count | 107.8182 |
| coarse_source_count | 2.0909 |
| coarse_duplicate_source_slots | 105.7273 |
| rerank_candidate_count | 43.6364 |
| rerank_source_count | 2.0909 |
| rerank_duplicate_source_slots | 41.5455 |
| final_candidate_count | 3.0556 |
| final_source_count | 1.0000 |
| final_duplicate_source_slots | 2.0556 |
- 全部 resolved fusion modes：{"weighted": 11}；cliff enabled：0
- 全部 dense evidence survival recall：N/A；覆盖率：0.0%
- 全部 fusion evidence survival recall：0.7258；覆盖率：55.0%
- 全部 rerank evidence survival recall：0.6182；覆盖率：55.0%
- 全部 final_top5 evidence survival recall：0.2815；覆盖率：90.0%
- 独立正例 resolved fusion modes：{"weighted": 11}；cliff enabled：0
- 独立正例 dense evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 fusion evidence survival recall：0.7258；覆盖率：61.1%
- 独立正例 rerank evidence survival recall：0.6182；覆盖率：61.1%
- 独立正例 final_top5 evidence survival recall：0.2815；覆盖率：100.0%
- cross_doc resolved fusion modes：{"weighted": 8}；cliff enabled：0
- cross_doc dense evidence survival recall：N/A；覆盖率：0.0%
- cross_doc fusion evidence survival recall：0.6854；覆盖率：80.0%
- cross_doc rerank evidence survival recall：0.5375；覆盖率：80.0%
- cross_doc final_top5 evidence survival recall：0.3567；覆盖率：100.0%
- explicit resolved fusion modes：{"weighted": 3}；cliff enabled：0
- explicit dense evidence survival recall：N/A；覆盖率：0.0%
- explicit fusion evidence survival recall：0.8333；覆盖率：37.5%
- explicit rerank evidence survival recall：0.8333；覆盖率：37.5%
- explicit final_top5 evidence survival recall：0.1875；覆盖率：100.0%
- negative resolved fusion modes：{}；cliff enabled：0
- negative dense evidence survival recall：N/A；覆盖率：0.0%
- negative fusion evidence survival recall：N/A；覆盖率：0.0%
- negative rerank evidence survival recall：N/A；覆盖率：0.0%
- negative final_top5 evidence survival recall：N/A；覆盖率：0.0%

- sparse survival：unavailable（Milvus hybrid API 未返回内部独立榜单；使用 L0/L1 跨层对比）。
- coarse/rerank chunk IDs 与逐条 evidence survival 见 records.jsonl / per_query.csv。

## 运行可复现配置

- dataset_fingerprint：200d32fa8b87fdc25606d83aab8b694fc27236197f958606ac29f74ea3b8e422
- kb_fingerprint：e1421cd54105de5e02d69ef234a3b0087a15fc5fef9b301be3230c5282e6e545
- metrics_version：2.0
- annotation_schema_versions：['2.0']
- pdf_set_fingerprint：ddc1e68c63656b46110cd06179ac3e674a0986d37c9f4871a81973104927740b
- chunk_content_fingerprint：ac243d4db9ed2611751f06d22c49d8bdc052ee2d7c08c0333cce31747ff34ed8
- parser_clean_chunk_rules：11184507f70129696792593fcdd360dee8dea7849bb350bc2f1a53fa82bbd406
- code_worktree_sha256：bd480210e0546adbce6035f1c407391e804e68ea56f54586243735a3726d81a6
- code_diff_sha256：8cd6e3f5dffb8e272304d15308844a4f4632b6986e7d8b999c0de7ea077ba668
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
