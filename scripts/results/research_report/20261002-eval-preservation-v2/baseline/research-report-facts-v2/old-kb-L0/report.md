# 评测报告：L0

- 模式：retrieval
- 样本数：18（含模糊配对；检索主指标看下面的独立正例）
- 完成/失败：18/0

## 独立正例（explicit + cross_doc）

不含 vague 配对题，也不含负样本

- 样本数：18

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.7222 | 100.0% |
| document_recall_at_5 | 0.8704 | 100.0% |
| evidence_group_recall_at_5 | 0.3380 | 100.0% |
| exact_candidate_recall_at_20 | 0.7056 | 83.3% |
| exact_candidate_recall_at_50 | 0.8333 | 83.3% |
| exact_candidate_recall_at_actual_depth | 0.9333 | 83.3% |
| fact_mrr | 0.2107 | 100.0% |
| mrr | 0.3778 | 83.3% |
| precision_at_5 | 0.1867 | 83.3% |
| quote_coverage_at_5 | 0.2853 | 100.0% |
| recall_at_5 | 0.4056 | 83.3% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.3009 | 100.0% |

- 父块还原 parent_restore_rate：0.8800 （命中 25 块，还原 22，未替换 3）

## 分层

| 层 | n | 指标 | 均值 | 覆盖率 |
|---|---:|---|---:|---:|
| cross_doc | 10 | all_required_documents_at_5 | 0.5000 | 100.0% |
| cross_doc | 10 | document_recall_at_5 | 0.7667 | 100.0% |
| cross_doc | 10 | evidence_group_recall_at_5 | 0.3833 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_20 | 0.6583 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_50 | 0.8500 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_actual_depth | 1.0000 | 100.0% |
| cross_doc | 10 | fact_mrr | 0.2667 | 100.0% |
| cross_doc | 10 | mrr | 0.4917 | 100.0% |
| cross_doc | 10 | precision_at_5 | 0.2200 | 100.0% |
| cross_doc | 10 | quote_coverage_at_5 | 0.3719 | 100.0% |
| cross_doc | 10 | recall_at_5 | 0.4417 | 100.0% |
| cross_doc | 10 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| cross_doc | 10 | structured_required_fact_coverage_at_5 | 0.3167 | 100.0% |
| explicit | 8 | all_required_documents_at_5 | 1.0000 | 100.0% |
| explicit | 8 | document_recall_at_5 | 1.0000 | 100.0% |
| explicit | 8 | evidence_group_recall_at_5 | 0.2812 | 100.0% |
| explicit | 8 | exact_candidate_recall_at_20 | 0.8000 | 62.5% |
| explicit | 8 | exact_candidate_recall_at_50 | 0.8000 | 62.5% |
| explicit | 8 | exact_candidate_recall_at_actual_depth | 0.8000 | 62.5% |
| explicit | 8 | fact_mrr | 0.1406 | 100.0% |
| explicit | 8 | mrr | 0.1500 | 62.5% |
| explicit | 8 | precision_at_5 | 0.1200 | 62.5% |
| explicit | 8 | quote_coverage_at_5 | 0.1771 | 100.0% |
| explicit | 8 | recall_at_5 | 0.3333 | 62.5% |
| explicit | 8 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| explicit | 8 | structured_required_fact_coverage_at_5 | 0.2812 | 100.0% |

## 父块还原（全部命中证据块）

- rate：0.8800
- hits / restored / not_replaced / missing / no_parent：25 / 22 / 3 / 0 / 0

## 全量均值（含模糊配对，不作为检索主指标）

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.7222 | 100.0% |
| document_recall_at_5 | 0.8704 | 100.0% |
| evidence_group_recall_at_5 | 0.3380 | 100.0% |
| exact_candidate_recall_at_20 | 0.7056 | 83.3% |
| exact_candidate_recall_at_50 | 0.8333 | 83.3% |
| exact_candidate_recall_at_actual_depth | 0.9333 | 83.3% |
| fact_mrr | 0.2107 | 100.0% |
| mrr | 0.3778 | 83.3% |
| precision_at_5 | 0.1867 | 83.3% |
| quote_coverage_at_5 | 0.2853 | 100.0% |
| recall_at_5 | 0.4056 | 83.3% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.3009 | 100.0% |

## 检索侧客观指标（Recall@5 / MRR 等）

块级 Recall/MRR 按 chunk id 精确匹配，文档 Recall 按 source 匹配；
quote/required-fact coverage 按归一化后的块内文字与数值边界匹配。
与上面的 RAGAS judge 指标分开读。

| 指标 | 均值 |
|---|---:|
| all_required_documents_at_5 | 0.7222 |
| document_recall_at_5 | 0.8704 |
| evidence_group_recall_at_5 | 0.3380 |
| exact_candidate_recall_at_20 | 0.7056 |
| exact_candidate_recall_at_50 | 0.8333 |
| exact_candidate_recall_at_actual_depth | 0.9333 |
| fact_mrr | 0.2107 |
| mrr | 0.3778 |
| precision_at_5 | 0.1867 |
| quote_coverage_at_5 | 0.2853 |
| recall_at_5 | 0.4056 |
| required_fact_coverage_at_5 | 0.0000 |
| structured_required_fact_coverage_at_5 | 0.3009 |

## 检索阶段诊断

| 阶段字段 | 均值 |
|---|---:|
| coarse_candidate_count | 150.0000 |
| coarse_source_count | 8.5556 |
| coarse_duplicate_source_slots | 141.4444 |
| rerank_candidate_count | N/A |
| rerank_source_count | N/A |
| rerank_duplicate_source_slots | N/A |
| final_candidate_count | 5.0000 |
| final_source_count | 1.3333 |
| final_duplicate_source_slots | 3.6667 |
- 全部 resolved fusion modes：{"dense": 18}；cliff enabled：0
- 全部 dense evidence survival recall：0.5491；覆盖率：100.0%
- 全部 fusion evidence survival recall：N/A；覆盖率：0.0%
- 全部 rerank evidence survival recall：N/A；覆盖率：0.0%
- 全部 final_top5 evidence survival recall：0.2426；覆盖率：100.0%
- 独立正例 resolved fusion modes：{"dense": 18}；cliff enabled：0
- 独立正例 dense evidence survival recall：0.5491；覆盖率：100.0%
- 独立正例 fusion evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 rerank evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 final_top5 evidence survival recall：0.2426；覆盖率：100.0%
- cross_doc resolved fusion modes：{"dense": 10}；cliff enabled：0
- cross_doc dense evidence survival recall：0.7633；覆盖率：100.0%
- cross_doc fusion evidence survival recall：N/A；覆盖率：0.0%
- cross_doc rerank evidence survival recall：N/A；覆盖率：0.0%
- cross_doc final_top5 evidence survival recall：0.3367；覆盖率：100.0%
- explicit resolved fusion modes：{"dense": 8}；cliff enabled：0
- explicit dense evidence survival recall：0.2812；覆盖率：100.0%
- explicit fusion evidence survival recall：N/A；覆盖率：0.0%
- explicit rerank evidence survival recall：N/A；覆盖率：0.0%
- explicit final_top5 evidence survival recall：0.1250；覆盖率：100.0%

- sparse survival：unavailable（Milvus hybrid API 未返回内部独立榜单；使用 L0/L1 跨层对比）。
- coarse/rerank chunk IDs 与逐条 evidence survival 见 records.jsonl / per_query.csv。

## 运行可复现配置

- dataset_fingerprint：105cca64827c0ad58a20d43cbdb098eba7cad4828beeed7451b54290b5b9c9ae
- kb_fingerprint：25f3f726000cf48467a663347d984133c954058dad115debe4a1395b7886d17e
- metrics_version：2.0
- annotation_schema_versions：['2.0']
- pdf_set_fingerprint：ddc1e68c63656b46110cd06179ac3e674a0986d37c9f4871a81973104927740b
- chunk_content_fingerprint：22d55430b5ba0f355e71b5e4aca39ddba26d2257a09fef124aeb71f5a4e6f1ff
- parser_clean_chunk_rules：11184507f70129696792593fcdd360dee8dea7849bb350bc2f1a53fa82bbd406
- code_worktree_sha256：79beedf324b1a25c6a7a7215606ba111781992b31e54c80d03229d115f2647d5
- code_diff_sha256：6d77020dbddd5e4bcfb4c6028969ba4324b640b77c1af3ac59f4f818e3e6a2f7
- embedding_model：BAAI/bge-m3
- embedding_version：BAAI/bge-m3
- embedding_dim：1024
- embedding_pooling：cls
- chunk_max_tokens：256
- chunk_overlap_tokens：25
- fusion_mode：dense
- dense_weight：0.8
- sparse_weight：0.2
- git_head：41ca0191f9f014535ecd3c452bb85c66a6f206f2
- git_dirty：True
