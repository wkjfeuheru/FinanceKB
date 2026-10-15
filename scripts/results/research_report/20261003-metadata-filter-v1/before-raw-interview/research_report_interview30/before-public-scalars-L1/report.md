# 评测报告：L1

- 模式：retrieval
- 样本数：28（含模糊配对；检索主指标看下面的独立正例）
- 完成/失败：28/0

## 独立正例（explicit + cross_doc）

不含 vague 配对题，也不含负样本

- 样本数：18

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.7222 | 100.0% |
| document_recall_at_5 | 0.8611 | 100.0% |
| evidence_group_recall_at_5 | 0.2917 | 100.0% |
| exact_candidate_recall_at_20 | 0.5519 | 100.0% |
| exact_candidate_recall_at_50 | 0.7787 | 100.0% |
| exact_candidate_recall_at_actual_depth | 0.9639 | 100.0% |
| fact_mrr | 0.2222 | 100.0% |
| mrr | 0.3472 | 100.0% |
| precision_at_5 | 0.1444 | 100.0% |
| quote_coverage_at_5 | 0.2800 | 100.0% |
| recall_at_5 | 0.2769 | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.2546 | 100.0% |

- 父块还原 parent_restore_rate：0.7083 （命中 24 块，还原 17，未替换 7）

## 分层

| 层 | n | 指标 | 均值 | 覆盖率 |
|---|---:|---|---:|---:|
| cross_doc | 10 | all_required_documents_at_5 | 0.5000 | 100.0% |
| cross_doc | 10 | document_recall_at_5 | 0.7500 | 100.0% |
| cross_doc | 10 | evidence_group_recall_at_5 | 0.3000 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_20 | 0.7267 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_50 | 0.8017 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_actual_depth | 0.9350 | 100.0% |
| cross_doc | 10 | fact_mrr | 0.2375 | 100.0% |
| cross_doc | 10 | mrr | 0.4250 | 100.0% |
| cross_doc | 10 | precision_at_5 | 0.1800 | 100.0% |
| cross_doc | 10 | quote_coverage_at_5 | 0.3211 | 100.0% |
| cross_doc | 10 | recall_at_5 | 0.2983 | 100.0% |
| cross_doc | 10 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| cross_doc | 10 | structured_required_fact_coverage_at_5 | 0.2333 | 100.0% |
| explicit | 8 | all_required_documents_at_5 | 1.0000 | 100.0% |
| explicit | 8 | document_recall_at_5 | 1.0000 | 100.0% |
| explicit | 8 | evidence_group_recall_at_5 | 0.2812 | 100.0% |
| explicit | 8 | exact_candidate_recall_at_20 | 0.3333 | 100.0% |
| explicit | 8 | exact_candidate_recall_at_50 | 0.7500 | 100.0% |
| explicit | 8 | exact_candidate_recall_at_actual_depth | 1.0000 | 100.0% |
| explicit | 8 | fact_mrr | 0.2031 | 100.0% |
| explicit | 8 | mrr | 0.2500 | 100.0% |
| explicit | 8 | precision_at_5 | 0.1000 | 100.0% |
| explicit | 8 | quote_coverage_at_5 | 0.2286 | 100.0% |
| explicit | 8 | recall_at_5 | 0.2500 | 100.0% |
| explicit | 8 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| explicit | 8 | structured_required_fact_coverage_at_5 | 0.2812 | 100.0% |
| vague | 10 | all_required_documents_at_5 | 0.9000 | 100.0% |
| vague | 10 | document_recall_at_5 | 0.9500 | 100.0% |
| vague | 10 | evidence_group_recall_at_5 | 0.2750 | 100.0% |
| vague | 10 | exact_candidate_recall_at_20 | 0.4833 | 100.0% |
| vague | 10 | exact_candidate_recall_at_50 | 1.0000 | 100.0% |
| vague | 10 | exact_candidate_recall_at_actual_depth | 1.0000 | 100.0% |
| vague | 10 | fact_mrr | 0.1625 | 100.0% |
| vague | 10 | mrr | 0.2500 | 100.0% |
| vague | 10 | precision_at_5 | 0.0800 | 100.0% |
| vague | 10 | quote_coverage_at_5 | 0.2495 | 100.0% |
| vague | 10 | recall_at_5 | 0.2167 | 100.0% |
| vague | 10 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| vague | 10 | structured_required_fact_coverage_at_5 | 0.2750 | 100.0% |

## 模糊题相对配对题

这 10 题与原题共享证据，差值不能并进独立正例的 Recall

- 配对数：10
- recall_at_5 平均差值（模糊 - 配对）：-0.0333
- mrr 平均差值（模糊 - 配对）：-0.0500

## 父块还原（全部命中证据块）

- rate：0.7188
- hits / restored / not_replaced / missing / no_parent：32 / 23 / 9 / 0 / 0

## 全量均值（含模糊配对，不作为检索主指标）

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.7857 | 100.0% |
| document_recall_at_5 | 0.8929 | 100.0% |
| evidence_group_recall_at_5 | 0.2857 | 100.0% |
| exact_candidate_recall_at_20 | 0.5274 | 100.0% |
| exact_candidate_recall_at_50 | 0.8577 | 100.0% |
| exact_candidate_recall_at_actual_depth | 0.9768 | 100.0% |
| fact_mrr | 0.2009 | 100.0% |
| mrr | 0.3125 | 100.0% |
| precision_at_5 | 0.1214 | 100.0% |
| quote_coverage_at_5 | 0.2691 | 100.0% |
| recall_at_5 | 0.2554 | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.2619 | 100.0% |

## 检索侧客观指标（Recall@5 / MRR 等）

块级 Recall/MRR 按 chunk id 精确匹配，文档 Recall 按 source 匹配；
quote/required-fact coverage 按归一化后的块内文字与数值边界匹配。
与上面的 RAGAS judge 指标分开读。

| 指标 | 均值 |
|---|---:|
| all_required_documents_at_5 | 0.7857 |
| document_recall_at_5 | 0.8929 |
| evidence_group_recall_at_5 | 0.2857 |
| exact_candidate_recall_at_20 | 0.5274 |
| exact_candidate_recall_at_50 | 0.8577 |
| exact_candidate_recall_at_actual_depth | 0.9768 |
| fact_mrr | 0.2009 |
| mrr | 0.3125 |
| precision_at_5 | 0.1214 |
| quote_coverage_at_5 | 0.2691 |
| recall_at_5 | 0.2554 |
| required_fact_coverage_at_5 | 0.0000 |
| structured_required_fact_coverage_at_5 | 0.2619 |

## 检索阶段诊断

| 阶段字段 | 均值 |
|---|---:|
| coarse_candidate_count | 150.0000 |
| coarse_source_count | 8.0714 |
| coarse_duplicate_source_slots | 141.9286 |
| rerank_candidate_count | N/A |
| rerank_source_count | N/A |
| rerank_duplicate_source_slots | N/A |
| final_candidate_count | 5.0000 |
| final_source_count | 1.2500 |
| final_duplicate_source_slots | 3.7500 |
- 全部 resolved fusion modes：{"weighted": 28}；cliff enabled：0
- 全部 dense evidence survival recall：N/A；覆盖率：0.0%
- 全部 fusion evidence survival recall：0.8970；覆盖率：100.0%
- 全部 rerank evidence survival recall：N/A；覆盖率：0.0%
- 全部 final_top5 evidence survival recall：0.2464；覆盖率：100.0%
- 独立正例 resolved fusion modes：{"weighted": 18}；cliff enabled：0
- 独立正例 dense evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 fusion evidence survival recall：0.8815；覆盖率：100.0%
- 独立正例 rerank evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 final_top5 evidence survival recall：0.2676；覆盖率：100.0%
- cross_doc resolved fusion modes：{"weighted": 10}；cliff enabled：0
- cross_doc dense evidence survival recall：N/A；覆盖率：0.0%
- cross_doc fusion evidence survival recall：0.8617；覆盖率：100.0%
- cross_doc rerank evidence survival recall：N/A；覆盖率：0.0%
- cross_doc final_top5 evidence survival recall：0.2983；覆盖率：100.0%
- explicit resolved fusion modes：{"weighted": 8}；cliff enabled：0
- explicit dense evidence survival recall：N/A；覆盖率：0.0%
- explicit fusion evidence survival recall：0.9062；覆盖率：100.0%
- explicit rerank evidence survival recall：N/A；覆盖率：0.0%
- explicit final_top5 evidence survival recall：0.2292；覆盖率：100.0%
- vague resolved fusion modes：{"weighted": 10}；cliff enabled：0
- vague dense evidence survival recall：N/A；覆盖率：0.0%
- vague fusion evidence survival recall：0.9250；覆盖率：100.0%
- vague rerank evidence survival recall：N/A；覆盖率：0.0%
- vague final_top5 evidence survival recall：0.2083；覆盖率：100.0%

- sparse survival：unavailable（Milvus hybrid API 未返回内部独立榜单；使用 L0/L1 跨层对比）。
- coarse/rerank chunk IDs 与逐条 evidence survival 见 records.jsonl / per_query.csv。

## 运行可复现配置

- dataset_fingerprint：f9729fbbf7c789150f21a9a82ef178ea40f8ae5fcea129f9d68e4c9cb996dc7c
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
