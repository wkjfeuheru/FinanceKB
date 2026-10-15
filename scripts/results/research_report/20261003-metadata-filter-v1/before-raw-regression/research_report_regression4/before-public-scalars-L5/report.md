# 评测报告：L5

- 模式：retrieval
- 样本数：4（含模糊配对；检索主指标看下面的独立正例）
- 完成/失败：4/0

## 独立正例（explicit + cross_doc）

不含 vague 配对题，也不含负样本

- 样本数：4

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 1.0000 | 100.0% |
| document_recall_at_5 | 1.0000 | 100.0% |
| evidence_group_recall_at_5 | 0.5000 | 100.0% |
| exact_candidate_recall_at_20 | 1.0000 | 50.0% |
| exact_candidate_recall_at_50 | 1.0000 | 50.0% |
| exact_candidate_recall_at_actual_depth | 1.0000 | 50.0% |
| fact_mrr | 0.5000 | 100.0% |
| mrr | 1.0000 | 50.0% |
| precision_at_5 | 0.2000 | 50.0% |
| quote_coverage_at_5 | 0.5000 | 100.0% |
| recall_at_5 | 1.0000 | 50.0% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.5000 | 100.0% |

- 父块还原 parent_restore_rate：0.5000 （命中 4 块，还原 2，未替换 2）

## 分层

| 层 | n | 指标 | 均值 | 覆盖率 |
|---|---:|---|---:|---:|
| explicit | 4 | all_required_documents_at_5 | 1.0000 | 100.0% |
| explicit | 4 | document_recall_at_5 | 1.0000 | 100.0% |
| explicit | 4 | evidence_group_recall_at_5 | 0.5000 | 100.0% |
| explicit | 4 | exact_candidate_recall_at_20 | 1.0000 | 50.0% |
| explicit | 4 | exact_candidate_recall_at_50 | 1.0000 | 50.0% |
| explicit | 4 | exact_candidate_recall_at_actual_depth | 1.0000 | 50.0% |
| explicit | 4 | fact_mrr | 0.5000 | 100.0% |
| explicit | 4 | mrr | 1.0000 | 50.0% |
| explicit | 4 | precision_at_5 | 0.2000 | 50.0% |
| explicit | 4 | quote_coverage_at_5 | 0.5000 | 100.0% |
| explicit | 4 | recall_at_5 | 1.0000 | 50.0% |
| explicit | 4 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| explicit | 4 | structured_required_fact_coverage_at_5 | 0.5000 | 100.0% |

## 父块还原（全部命中证据块）

- rate：0.5000
- hits / restored / not_replaced / missing / no_parent：4 / 2 / 2 / 0 / 0

## 全量均值（含模糊配对，不作为检索主指标）

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 1.0000 | 100.0% |
| document_recall_at_5 | 1.0000 | 100.0% |
| evidence_group_recall_at_5 | 0.5000 | 100.0% |
| exact_candidate_recall_at_20 | 1.0000 | 50.0% |
| exact_candidate_recall_at_50 | 1.0000 | 50.0% |
| exact_candidate_recall_at_actual_depth | 1.0000 | 50.0% |
| fact_mrr | 0.5000 | 100.0% |
| mrr | 1.0000 | 50.0% |
| precision_at_5 | 0.2000 | 50.0% |
| quote_coverage_at_5 | 0.5000 | 100.0% |
| recall_at_5 | 1.0000 | 50.0% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.5000 | 100.0% |

## 检索侧客观指标（Recall@5 / MRR 等）

块级 Recall/MRR 按 chunk id 精确匹配，文档 Recall 按 source 匹配；
quote/required-fact coverage 按归一化后的块内文字与数值边界匹配。
与上面的 RAGAS judge 指标分开读。

| 指标 | 均值 |
|---|---:|
| all_required_documents_at_5 | 1.0000 |
| document_recall_at_5 | 1.0000 |
| evidence_group_recall_at_5 | 0.5000 |
| exact_candidate_recall_at_20 | 1.0000 |
| exact_candidate_recall_at_50 | 1.0000 |
| exact_candidate_recall_at_actual_depth | 1.0000 |
| fact_mrr | 0.5000 |
| mrr | 1.0000 |
| precision_at_5 | 0.2000 |
| quote_coverage_at_5 | 0.5000 |
| recall_at_5 | 1.0000 |
| required_fact_coverage_at_5 | 0.0000 |
| structured_required_fact_coverage_at_5 | 0.5000 |

## 检索阶段诊断

| 阶段字段 | 均值 |
|---|---:|
| coarse_candidate_count | 150.0000 |
| coarse_source_count | 9.5000 |
| coarse_duplicate_source_slots | 140.5000 |
| rerank_candidate_count | 50.0000 |
| rerank_source_count | 7.7500 |
| rerank_duplicate_source_slots | 42.2500 |
| final_candidate_count | 5.0000 |
| final_source_count | 3.2500 |
| final_duplicate_source_slots | 1.7500 |
- 全部 resolved fusion modes：{"weighted": 4}；cliff enabled：4
- 全部 dense evidence survival recall：N/A；覆盖率：0.0%
- 全部 fusion evidence survival recall：0.5000；覆盖率：100.0%
- 全部 rerank evidence survival recall：0.5000；覆盖率：100.0%
- 全部 final_top5 evidence survival recall：0.5000；覆盖率：100.0%
- 独立正例 resolved fusion modes：{"weighted": 4}；cliff enabled：4
- 独立正例 dense evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 fusion evidence survival recall：0.5000；覆盖率：100.0%
- 独立正例 rerank evidence survival recall：0.5000；覆盖率：100.0%
- 独立正例 final_top5 evidence survival recall：0.5000；覆盖率：100.0%
- explicit resolved fusion modes：{"weighted": 4}；cliff enabled：4
- explicit dense evidence survival recall：N/A；覆盖率：0.0%
- explicit fusion evidence survival recall：0.5000；覆盖率：100.0%
- explicit rerank evidence survival recall：0.5000；覆盖率：100.0%
- explicit final_top5 evidence survival recall：0.5000；覆盖率：100.0%

- sparse survival：unavailable（Milvus hybrid API 未返回内部独立榜单；使用 L0/L1 跨层对比）。
- coarse/rerank chunk IDs 与逐条 evidence survival 见 records.jsonl / per_query.csv。

## 运行可复现配置

- dataset_fingerprint：bb6a3e9eea03229a4f8e694382908de9970ac33e793d6ad08a35190a80ecb324
- kb_fingerprint：e1421cd54105de5e02d69ef234a3b0087a15fc5fef9b301be3230c5282e6e545
- metrics_version：2.0
- annotation_schema_versions：['2.0']
- pdf_set_fingerprint：ddc1e68c63656b46110cd06179ac3e674a0986d37c9f4871a81973104927740b
- chunk_content_fingerprint：ac243d4db9ed2611751f06d22c49d8bdc052ee2d7c08c0333cce31747ff34ed8
- parser_clean_chunk_rules：11184507f70129696792593fcdd360dee8dea7849bb350bc2f1a53fa82bbd406
- code_worktree_sha256：35f0574c312e0da6b0befe534a061f7d46cc456e3871ec06a2f4586a77ca128b
- code_diff_sha256：53b1f8c3ba57245857775d20d8d6376083e749db86af5eecd5ef6a502fefe0bc
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
