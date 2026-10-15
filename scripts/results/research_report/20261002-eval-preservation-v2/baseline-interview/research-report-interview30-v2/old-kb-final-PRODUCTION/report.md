# 评测报告：PRODUCTION

- 模式：retrieval
- 样本数：30（含模糊配对；检索主指标看下面的独立正例）
- 完成/失败：30/0

## 独立正例（explicit + cross_doc）

不含 vague 配对题，也不含负样本

- 样本数：18

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.3889 | 100.0% |
| context_required_fact_coverage | 0.3796 | 100.0% |
| document_recall_at_5 | 0.5000 | 100.0% |
| evidence_group_recall_at_5 | 0.4074 | 100.0% |
| exact_candidate_recall_at_20 | 0.5318 | 61.1% |
| exact_candidate_recall_at_50 | 0.7625 | 44.4% |
| exact_candidate_recall_at_actual_depth | 0.7136 | 61.1% |
| fact_mrr | 0.3366 | 100.0% |
| mrr | 0.3611 | 100.0% |
| negative_rejection | N/A | 0.0% |
| precision_at_5 | 0.1778 | 100.0% |
| quote_coverage_at_5 | 0.3704 | 100.0% |
| recall_at_5 | 0.3278 | 100.0% |
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
| cross_doc | 10 | mrr | 0.4500 | 100.0% |
| cross_doc | 10 | negative_rejection | N/A | 0.0% |
| cross_doc | 10 | precision_at_5 | 0.2400 | 100.0% |
| cross_doc | 10 | quote_coverage_at_5 | 0.4666 | 100.0% |
| cross_doc | 10 | recall_at_5 | 0.3900 | 100.0% |
| cross_doc | 10 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| cross_doc | 10 | structured_required_fact_coverage_at_5 | 0.4833 | 100.0% |
| explicit | 8 | all_required_documents_at_5 | 0.3750 | 100.0% |
| explicit | 8 | context_required_fact_coverage | 0.2500 | 100.0% |
| explicit | 8 | document_recall_at_5 | 0.3750 | 100.0% |
| explicit | 8 | evidence_group_recall_at_5 | 0.2500 | 100.0% |
| explicit | 8 | exact_candidate_recall_at_20 | 0.3333 | 37.5% |
| explicit | 8 | exact_candidate_recall_at_50 | 1.0000 | 12.5% |
| explicit | 8 | exact_candidate_recall_at_actual_depth | 0.6667 | 37.5% |
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
| vague | 10 | all_required_documents_at_5 | 0.3000 | 100.0% |
| vague | 10 | context_required_fact_coverage | 0.2000 | 100.0% |
| vague | 10 | document_recall_at_5 | 0.3000 | 100.0% |
| vague | 10 | evidence_group_recall_at_5 | 0.2000 | 100.0% |
| vague | 10 | exact_candidate_recall_at_20 | 0.2222 | 30.0% |
| vague | 10 | exact_candidate_recall_at_50 | 1.0000 | 10.0% |
| vague | 10 | exact_candidate_recall_at_actual_depth | 0.6667 | 30.0% |
| vague | 10 | fact_mrr | 0.2000 | 100.0% |
| vague | 10 | mrr | 0.2000 | 100.0% |
| vague | 10 | negative_rejection | N/A | 0.0% |
| vague | 10 | precision_at_5 | 0.0800 | 100.0% |
| vague | 10 | quote_coverage_at_5 | 0.2000 | 100.0% |
| vague | 10 | recall_at_5 | 0.2000 | 100.0% |
| vague | 10 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| vague | 10 | structured_required_fact_coverage_at_5 | 0.2000 | 100.0% |

## 模糊题相对配对题

这 10 题与原题共享证据，差值不能并进独立正例的 Recall

- 配对数：10
- recall_at_5 平均差值（模糊 - 配对）：-0.1000
- mrr 平均差值（模糊 - 配对）：-0.1000

## 全量均值（含模糊配对，不作为检索主指标）

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.3571 | 93.3% |
| context_required_fact_coverage | 0.3155 | 93.3% |
| document_recall_at_5 | 0.4286 | 93.3% |
| evidence_group_recall_at_5 | 0.3333 | 93.3% |
| exact_candidate_recall_at_20 | 0.4655 | 46.7% |
| exact_candidate_recall_at_50 | 0.7889 | 30.0% |
| exact_candidate_recall_at_actual_depth | 0.7036 | 46.7% |
| fact_mrr | 0.2878 | 93.3% |
| mrr | 0.3036 | 93.3% |
| negative_rejection | 0.0000 | 6.7% |
| precision_at_5 | 0.1429 | 93.3% |
| quote_coverage_at_5 | 0.3095 | 93.3% |
| recall_at_5 | 0.2821 | 93.3% |
| required_fact_coverage_at_5 | 0.0000 | 93.3% |
| structured_required_fact_coverage_at_5 | 0.3155 | 93.3% |

## 检索侧客观指标（Recall@5 / MRR 等）

块级 Recall/MRR 按 chunk id 精确匹配，文档 Recall 按 source 匹配；
quote/required-fact coverage 按归一化后的块内文字与数值边界匹配。
与上面的 RAGAS judge 指标分开读。

| 指标 | 均值 |
|---|---:|
| all_required_documents_at_5 | 0.3571 |
| context_required_fact_coverage | 0.3155 |
| document_recall_at_5 | 0.4286 |
| evidence_group_recall_at_5 | 0.3333 |
| exact_candidate_recall_at_20 | 0.4655 |
| exact_candidate_recall_at_50 | 0.7889 |
| exact_candidate_recall_at_actual_depth | 0.7036 |
| fact_mrr | 0.2878 |
| mrr | 0.3036 |
| negative_rejection | 0.0000 |
| precision_at_5 | 0.1429 |
| quote_coverage_at_5 | 0.3095 |
| recall_at_5 | 0.2821 |
| required_fact_coverage_at_5 | 0.0000 |
| structured_required_fact_coverage_at_5 | 0.3155 |

## 检索阶段诊断

| 阶段字段 | 均值 |
|---|---:|
| coarse_candidate_count | 100.8571 |
| coarse_source_count | 1.8571 |
| coarse_duplicate_source_slots | 99.0000 |
| rerank_candidate_count | 42.2857 |
| rerank_source_count | 1.8571 |
| rerank_duplicate_source_slots | 40.4286 |
| final_candidate_count | 2.5000 |
| final_source_count | 0.7500 |
| final_duplicate_source_slots | 1.7500 |
- 全部 resolved fusion modes：{"weighted": 14}；cliff enabled：0
- 全部 dense evidence survival recall：N/A；覆盖率：0.0%
- 全部 fusion evidence survival recall：0.6060；覆盖率：46.7%
- 全部 rerank evidence survival recall：0.5214；覆盖率：46.7%
- 全部 final_top5 evidence survival recall：0.2464；覆盖率：93.3%
- 独立正例 resolved fusion modes：{"weighted": 11}；cliff enabled：0
- 独立正例 dense evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 fusion evidence survival recall：0.6348；覆盖率：61.1%
- 独立正例 rerank evidence survival recall：0.5273；覆盖率：61.1%
- 独立正例 final_top5 evidence survival recall：0.3000；覆盖率：100.0%
- cross_doc resolved fusion modes：{"weighted": 8}；cliff enabled：0
- cross_doc dense evidence survival recall：N/A；覆盖率：0.0%
- cross_doc fusion evidence survival recall：0.6854；覆盖率：80.0%
- cross_doc rerank evidence survival recall：0.5375；覆盖率：80.0%
- cross_doc final_top5 evidence survival recall：0.3900；覆盖率：100.0%
- explicit resolved fusion modes：{"weighted": 3}；cliff enabled：0
- explicit dense evidence survival recall：N/A；覆盖率：0.0%
- explicit fusion evidence survival recall：0.5000；覆盖率：37.5%
- explicit rerank evidence survival recall：0.5000；覆盖率：37.5%
- explicit final_top5 evidence survival recall：0.1875；覆盖率：100.0%
- negative resolved fusion modes：{}；cliff enabled：0
- negative dense evidence survival recall：N/A；覆盖率：0.0%
- negative fusion evidence survival recall：N/A；覆盖率：0.0%
- negative rerank evidence survival recall：N/A；覆盖率：0.0%
- negative final_top5 evidence survival recall：N/A；覆盖率：0.0%
- vague resolved fusion modes：{"weighted": 3}；cliff enabled：0
- vague dense evidence survival recall：N/A；覆盖率：0.0%
- vague fusion evidence survival recall：0.5000；覆盖率：30.0%
- vague rerank evidence survival recall：0.5000；覆盖率：30.0%
- vague final_top5 evidence survival recall：0.1500；覆盖率：100.0%

- sparse survival：unavailable（Milvus hybrid API 未返回内部独立榜单；使用 L0/L1 跨层对比）。
- coarse/rerank chunk IDs 与逐条 evidence survival 见 records.jsonl / per_query.csv。

## 运行可复现配置

- dataset_fingerprint：bf90fe445b81db0f41d4e9c989902708cadeae1ddc21c0ac2e314cd0d7172d47
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
