# 评测报告：L0

- 模式：retrieval
- 样本数：28（含模糊配对；检索主指标看下面的独立正例）
- 完成/失败：28/0

## 独立正例（explicit + cross_doc）

不含 vague 配对题，也不含负样本

- 样本数：18

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.8333 | 100.0% |
| document_recall_at_5 | 0.9167 | 100.0% |
| evidence_group_recall_at_5 | 0.3194 | 100.0% |
| exact_candidate_recall_at_20 | 0.5519 | 100.0% |
| exact_candidate_recall_at_50 | 0.7787 | 100.0% |
| exact_candidate_recall_at_actual_depth | 0.9639 | 100.0% |
| fact_mrr | 0.2500 | 100.0% |
| mrr | 0.4028 | 100.0% |
| precision_at_5 | 0.1667 | 100.0% |
| quote_coverage_at_5 | 0.3133 | 100.0% |
| recall_at_5 | 0.3139 | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.2824 | 100.0% |

- 父块还原 parent_restore_rate：0.7407 （命中 27 块，还原 20，未替换 7）

## 分层

| 层 | n | 指标 | 均值 | 覆盖率 |
|---|---:|---|---:|---:|
| cross_doc | 10 | all_required_documents_at_5 | 0.7000 | 100.0% |
| cross_doc | 10 | document_recall_at_5 | 0.8500 | 100.0% |
| cross_doc | 10 | evidence_group_recall_at_5 | 0.3500 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_20 | 0.7267 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_50 | 0.8017 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_actual_depth | 0.9350 | 100.0% |
| cross_doc | 10 | fact_mrr | 0.2875 | 100.0% |
| cross_doc | 10 | mrr | 0.5250 | 100.0% |
| cross_doc | 10 | precision_at_5 | 0.2200 | 100.0% |
| cross_doc | 10 | quote_coverage_at_5 | 0.3811 | 100.0% |
| cross_doc | 10 | recall_at_5 | 0.3650 | 100.0% |
| cross_doc | 10 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| cross_doc | 10 | structured_required_fact_coverage_at_5 | 0.2833 | 100.0% |
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
| vague | 10 | evidence_group_recall_at_5 | 0.3250 | 100.0% |
| vague | 10 | exact_candidate_recall_at_20 | 0.4833 | 100.0% |
| vague | 10 | exact_candidate_recall_at_50 | 1.0000 | 100.0% |
| vague | 10 | exact_candidate_recall_at_actual_depth | 1.0000 | 100.0% |
| vague | 10 | fact_mrr | 0.2125 | 100.0% |
| vague | 10 | mrr | 0.3500 | 100.0% |
| vague | 10 | precision_at_5 | 0.1200 | 100.0% |
| vague | 10 | quote_coverage_at_5 | 0.3095 | 100.0% |
| vague | 10 | recall_at_5 | 0.2833 | 100.0% |
| vague | 10 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| vague | 10 | structured_required_fact_coverage_at_5 | 0.3250 | 100.0% |

## 模糊题相对配对题

这 10 题与原题共享证据，差值不能并进独立正例的 Recall

- 配对数：10
- recall_at_5 平均差值（模糊 - 配对）：-0.0333
- mrr 平均差值（模糊 - 配对）：-0.0500

## 父块还原（全部命中证据块）

- rate：0.7632
- hits / restored / not_replaced / missing / no_parent：38 / 29 / 9 / 0 / 0

## 全量均值（含模糊配对，不作为检索主指标）

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.8571 | 100.0% |
| document_recall_at_5 | 0.9286 | 100.0% |
| evidence_group_recall_at_5 | 0.3214 | 100.0% |
| exact_candidate_recall_at_20 | 0.5274 | 100.0% |
| exact_candidate_recall_at_50 | 0.8577 | 100.0% |
| exact_candidate_recall_at_actual_depth | 0.9768 | 100.0% |
| fact_mrr | 0.2366 | 100.0% |
| mrr | 0.3839 | 100.0% |
| precision_at_5 | 0.1500 | 100.0% |
| quote_coverage_at_5 | 0.3120 | 100.0% |
| recall_at_5 | 0.3030 | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.2976 | 100.0% |

## 检索侧客观指标（Recall@5 / MRR 等）

块级 Recall/MRR 按 chunk id 精确匹配，文档 Recall 按 source 匹配；
quote/required-fact coverage 按归一化后的块内文字与数值边界匹配。
与上面的 RAGAS judge 指标分开读。

| 指标 | 均值 |
|---|---:|
| all_required_documents_at_5 | 0.8571 |
| document_recall_at_5 | 0.9286 |
| evidence_group_recall_at_5 | 0.3214 |
| exact_candidate_recall_at_20 | 0.5274 |
| exact_candidate_recall_at_50 | 0.8577 |
| exact_candidate_recall_at_actual_depth | 0.9768 |
| fact_mrr | 0.2366 |
| mrr | 0.3839 |
| precision_at_5 | 0.1500 |
| quote_coverage_at_5 | 0.3120 |
| recall_at_5 | 0.3030 |
| required_fact_coverage_at_5 | 0.0000 |
| structured_required_fact_coverage_at_5 | 0.2976 |

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
| final_source_count | 1.3214 |
| final_duplicate_source_slots | 3.6786 |
- 全部 resolved fusion modes：{"dense": 28}；cliff enabled：0
- 全部 dense evidence survival recall：0.8970；覆盖率：100.0%
- 全部 fusion evidence survival recall：N/A；覆盖率：0.0%
- 全部 rerank evidence survival recall：N/A；覆盖率：0.0%
- 全部 final_top5 evidence survival recall：0.2940；覆盖率：100.0%
- 独立正例 resolved fusion modes：{"dense": 18}；cliff enabled：0
- 独立正例 dense evidence survival recall：0.8815；覆盖率：100.0%
- 独立正例 fusion evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 rerank evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 final_top5 evidence survival recall：0.3046；覆盖率：100.0%
- cross_doc resolved fusion modes：{"dense": 10}；cliff enabled：0
- cross_doc dense evidence survival recall：0.8617；覆盖率：100.0%
- cross_doc fusion evidence survival recall：N/A；覆盖率：0.0%
- cross_doc rerank evidence survival recall：N/A；覆盖率：0.0%
- cross_doc final_top5 evidence survival recall：0.3650；覆盖率：100.0%
- explicit resolved fusion modes：{"dense": 8}；cliff enabled：0
- explicit dense evidence survival recall：0.9062；覆盖率：100.0%
- explicit fusion evidence survival recall：N/A；覆盖率：0.0%
- explicit rerank evidence survival recall：N/A；覆盖率：0.0%
- explicit final_top5 evidence survival recall：0.2292；覆盖率：100.0%
- vague resolved fusion modes：{"dense": 10}；cliff enabled：0
- vague dense evidence survival recall：0.9250；覆盖率：100.0%
- vague fusion evidence survival recall：N/A；覆盖率：0.0%
- vague rerank evidence survival recall：N/A；覆盖率：0.0%
- vague final_top5 evidence survival recall：0.2750；覆盖率：100.0%

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
- code_worktree_sha256：bd480210e0546adbce6035f1c407391e804e68ea56f54586243735a3726d81a6
- code_diff_sha256：8cd6e3f5dffb8e272304d15308844a4f4632b6986e7d8b999c0de7ea077ba668
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
