# 评测报告：PRODUCTION

- 模式：retrieval
- 样本数：30（含模糊配对；检索主指标看下面的独立正例）
- 完成/失败：30/0

## 独立正例（explicit + cross_doc）

不含 vague 配对题，也不含负样本

- 样本数：18

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.3333 | 100.0% |
| context_required_fact_coverage | 0.3241 | 100.0% |
| document_recall_at_5 | 0.4444 | 100.0% |
| evidence_group_recall_at_5 | 0.3518 | 100.0% |
| exact_candidate_recall_at_20 | 0.4517 | 55.6% |
| exact_candidate_recall_at_50 | 0.7286 | 38.9% |
| exact_candidate_recall_at_actual_depth | 0.7850 | 55.6% |
| fact_mrr | 0.3019 | 100.0% |
| mrr | 0.2778 | 100.0% |
| negative_rejection | N/A | 0.0% |
| precision_at_5 | 0.1333 | 100.0% |
| quote_coverage_at_5 | 0.2802 | 100.0% |
| recall_at_5 | 0.2537 | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.3241 | 100.0% |

## 分层

| 层 | n | 指标 | 均值 | 覆盖率 |
|---|---:|---|---:|---:|
| cross_doc | 10 | all_required_documents_at_5 | 0.3000 | 100.0% |
| cross_doc | 10 | context_required_fact_coverage | 0.3833 | 100.0% |
| cross_doc | 10 | document_recall_at_5 | 0.5000 | 100.0% |
| cross_doc | 10 | evidence_group_recall_at_5 | 0.4333 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_20 | 0.5500 | 70.0% |
| cross_doc | 10 | exact_candidate_recall_at_50 | 0.6833 | 60.0% |
| cross_doc | 10 | exact_candidate_recall_at_actual_depth | 0.6929 | 70.0% |
| cross_doc | 10 | fact_mrr | 0.3433 | 100.0% |
| cross_doc | 10 | mrr | 0.3000 | 100.0% |
| cross_doc | 10 | negative_rejection | N/A | 0.0% |
| cross_doc | 10 | precision_at_5 | 0.1600 | 100.0% |
| cross_doc | 10 | quote_coverage_at_5 | 0.3044 | 100.0% |
| cross_doc | 10 | recall_at_5 | 0.2567 | 100.0% |
| cross_doc | 10 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| cross_doc | 10 | structured_required_fact_coverage_at_5 | 0.3833 | 100.0% |
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
| vague | 10 | all_required_documents_at_5 | 0.3000 | 100.0% |
| vague | 10 | context_required_fact_coverage | 0.2000 | 100.0% |
| vague | 10 | document_recall_at_5 | 0.3000 | 100.0% |
| vague | 10 | evidence_group_recall_at_5 | 0.2000 | 100.0% |
| vague | 10 | exact_candidate_recall_at_20 | 0.2222 | 30.0% |
| vague | 10 | exact_candidate_recall_at_50 | 1.0000 | 10.0% |
| vague | 10 | exact_candidate_recall_at_actual_depth | 1.0000 | 30.0% |
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
- recall_at_5 平均差值（模糊 - 配对）：0.0000
- mrr 平均差值（模糊 - 配对）：0.0000

## 全量均值（含模糊配对，不作为检索主指标）

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 0.3214 | 93.3% |
| context_required_fact_coverage | 0.2798 | 93.3% |
| document_recall_at_5 | 0.3929 | 93.3% |
| evidence_group_recall_at_5 | 0.2976 | 93.3% |
| exact_candidate_recall_at_20 | 0.3987 | 43.3% |
| exact_candidate_recall_at_50 | 0.7625 | 26.7% |
| exact_candidate_recall_at_actual_depth | 0.8346 | 43.3% |
| fact_mrr | 0.2655 | 93.3% |
| mrr | 0.2500 | 93.3% |
| negative_rejection | 0.0000 | 6.7% |
| precision_at_5 | 0.1143 | 93.3% |
| quote_coverage_at_5 | 0.2516 | 93.3% |
| recall_at_5 | 0.2345 | 93.3% |
| required_fact_coverage_at_5 | 0.0000 | 93.3% |
| structured_required_fact_coverage_at_5 | 0.2798 | 93.3% |

## 检索侧客观指标（Recall@5 / MRR 等）

块级 Recall/MRR 按 chunk id 精确匹配，文档 Recall 按 source 匹配；
quote/required-fact coverage 按归一化后的块内文字与数值边界匹配。
与上面的 RAGAS judge 指标分开读。

| 指标 | 均值 |
|---|---:|
| all_required_documents_at_5 | 0.3214 |
| context_required_fact_coverage | 0.2798 |
| document_recall_at_5 | 0.3929 |
| evidence_group_recall_at_5 | 0.2976 |
| exact_candidate_recall_at_20 | 0.3987 |
| exact_candidate_recall_at_50 | 0.7625 |
| exact_candidate_recall_at_actual_depth | 0.8346 |
| fact_mrr | 0.2655 |
| mrr | 0.2500 |
| negative_rejection | 0.0000 |
| precision_at_5 | 0.1143 |
| quote_coverage_at_5 | 0.2516 |
| recall_at_5 | 0.2345 |
| required_fact_coverage_at_5 | 0.0000 |
| structured_required_fact_coverage_at_5 | 0.2798 |

## 检索阶段诊断

| 阶段字段 | 均值 |
|---|---:|
| coarse_candidate_count | 97.1538 |
| coarse_source_count | 1.7692 |
| coarse_duplicate_source_slots | 95.3846 |
| rerank_candidate_count | 41.5385 |
| rerank_source_count | 1.7692 |
| rerank_duplicate_source_slots | 39.7692 |
| final_candidate_count | 2.3214 |
| final_source_count | 0.6429 |
| final_duplicate_source_slots | 1.6786 |
- 全部 resolved fusion modes：{"weighted": 13}；cliff enabled：0
- 全部 dense evidence survival recall：N/A；覆盖率：0.0%
- 全部 fusion evidence survival recall：0.7295；覆盖率：43.3%
- 全部 rerank evidence survival recall：0.6385；覆盖率：43.3%
- 全部 final_top5 evidence survival recall：0.1988；覆盖率：93.3%
- 独立正例 resolved fusion modes：{"weighted": 10}；cliff enabled：0
- 独立正例 dense evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 fusion evidence survival recall：0.6983；覆盖率：55.6%
- 独立正例 rerank evidence survival recall：0.5800；覆盖率：55.6%
- 独立正例 final_top5 evidence survival recall：0.2259；覆盖率：100.0%
- cross_doc resolved fusion modes：{"weighted": 7}；cliff enabled：0
- cross_doc dense evidence survival recall：N/A；覆盖率：0.0%
- cross_doc fusion evidence survival recall：0.6405；覆盖率：70.0%
- cross_doc rerank evidence survival recall：0.4714；覆盖率：70.0%
- cross_doc final_top5 evidence survival recall：0.2567；覆盖率：100.0%
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
- vague resolved fusion modes：{"weighted": 3}；cliff enabled：0
- vague dense evidence survival recall：N/A；覆盖率：0.0%
- vague fusion evidence survival recall：0.8333；覆盖率：30.0%
- vague rerank evidence survival recall：0.8333；覆盖率：30.0%
- vague final_top5 evidence survival recall：0.1500；覆盖率：100.0%

- sparse survival：unavailable（Milvus hybrid API 未返回内部独立榜单；使用 L0/L1 跨层对比）。
- coarse/rerank chunk IDs 与逐条 evidence survival 见 records.jsonl / per_query.csv。

## 运行可复现配置

- dataset_fingerprint：f91bc1ae8c8cb742daeed31a19bf9afaad6e55d688c5a794ff01a1da729a0a9f
- kb_fingerprint：e1421cd54105de5e02d69ef234a3b0087a15fc5fef9b301be3230c5282e6e545
- metrics_version：2.0
- annotation_schema_versions：['2.0']
- pdf_set_fingerprint：ddc1e68c63656b46110cd06179ac3e674a0986d37c9f4871a81973104927740b
- chunk_content_fingerprint：ac243d4db9ed2611751f06d22c49d8bdc052ee2d7c08c0333cce31747ff34ed8
- parser_clean_chunk_rules：11184507f70129696792593fcdd360dee8dea7849bb350bc2f1a53fa82bbd406
- code_worktree_sha256：34701a428552175c4137ba39a26a4d32e72b1e1d17f0ce0d6fe408db6de46639
- code_diff_sha256：810d8bd979ad6cd92992099ab7c1a7e84314283f189bdad22678eed3cc18470d
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
