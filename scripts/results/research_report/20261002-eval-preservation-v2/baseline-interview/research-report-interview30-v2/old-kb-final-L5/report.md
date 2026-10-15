# 评测报告：L5

- 模式：retrieval
- 样本数：28（含模糊配对；检索主指标看下面的独立正例）
- 完成/失败：28/0

## 独立正例（explicit + cross_doc）

不含 vague 配对题，也不含负样本

- 样本数：18

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 1.0000 | 100.0% |
| document_recall_at_5 | 1.0000 | 100.0% |
| evidence_group_recall_at_5 | 0.5046 | 100.0% |
| exact_candidate_recall_at_20 | 0.6537 | 100.0% |
| exact_candidate_recall_at_50 | 0.8065 | 100.0% |
| exact_candidate_recall_at_actual_depth | 0.9444 | 100.0% |
| fact_mrr | 0.3935 | 100.0% |
| mrr | 0.6389 | 100.0% |
| precision_at_5 | 0.2000 | 100.0% |
| quote_coverage_at_5 | 0.4175 | 100.0% |
| recall_at_5 | 0.3954 | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.4954 | 100.0% |

- 父块还原 parent_restore_rate：0.8182 （命中 33 块，还原 27，未替换 6）

## 分层

| 层 | n | 指标 | 均值 | 覆盖率 |
|---|---:|---|---:|---:|
| cross_doc | 10 | all_required_documents_at_5 | 1.0000 | 100.0% |
| cross_doc | 10 | document_recall_at_5 | 1.0000 | 100.0% |
| cross_doc | 10 | evidence_group_recall_at_5 | 0.6833 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_20 | 0.6767 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_50 | 0.8517 | 100.0% |
| cross_doc | 10 | exact_candidate_recall_at_actual_depth | 1.0000 | 100.0% |
| cross_doc | 10 | fact_mrr | 0.4833 | 100.0% |
| cross_doc | 10 | mrr | 0.8500 | 100.0% |
| cross_doc | 10 | precision_at_5 | 0.2800 | 100.0% |
| cross_doc | 10 | quote_coverage_at_5 | 0.5630 | 100.0% |
| cross_doc | 10 | recall_at_5 | 0.5117 | 100.0% |
| cross_doc | 10 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| cross_doc | 10 | structured_required_fact_coverage_at_5 | 0.6667 | 100.0% |
| explicit | 8 | all_required_documents_at_5 | 1.0000 | 100.0% |
| explicit | 8 | document_recall_at_5 | 1.0000 | 100.0% |
| explicit | 8 | evidence_group_recall_at_5 | 0.2812 | 100.0% |
| explicit | 8 | exact_candidate_recall_at_20 | 0.6250 | 100.0% |
| explicit | 8 | exact_candidate_recall_at_50 | 0.7500 | 100.0% |
| explicit | 8 | exact_candidate_recall_at_actual_depth | 0.8750 | 100.0% |
| explicit | 8 | fact_mrr | 0.2812 | 100.0% |
| explicit | 8 | mrr | 0.3750 | 100.0% |
| explicit | 8 | precision_at_5 | 0.1000 | 100.0% |
| explicit | 8 | quote_coverage_at_5 | 0.2357 | 100.0% |
| explicit | 8 | recall_at_5 | 0.2500 | 100.0% |
| explicit | 8 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| explicit | 8 | structured_required_fact_coverage_at_5 | 0.2812 | 100.0% |
| vague | 10 | all_required_documents_at_5 | 1.0000 | 100.0% |
| vague | 10 | document_recall_at_5 | 1.0000 | 100.0% |
| vague | 10 | evidence_group_recall_at_5 | 0.4750 | 100.0% |
| vague | 10 | exact_candidate_recall_at_20 | 0.5833 | 100.0% |
| vague | 10 | exact_candidate_recall_at_50 | 0.9000 | 100.0% |
| vague | 10 | exact_candidate_recall_at_actual_depth | 0.9000 | 100.0% |
| vague | 10 | fact_mrr | 0.4167 | 100.0% |
| vague | 10 | mrr | 0.5500 | 100.0% |
| vague | 10 | precision_at_5 | 0.1800 | 100.0% |
| vague | 10 | quote_coverage_at_5 | 0.4752 | 100.0% |
| vague | 10 | recall_at_5 | 0.4500 | 100.0% |
| vague | 10 | required_fact_coverage_at_5 | 0.0000 | 100.0% |
| vague | 10 | structured_required_fact_coverage_at_5 | 0.4750 | 100.0% |

## 模糊题相对配对题

这 10 题与原题共享证据，差值不能并进独立正例的 Recall

- 配对数：10
- recall_at_5 平均差值（模糊 - 配对）：0.1000
- mrr 平均差值（模糊 - 配对）：0.1000

## 父块还原（全部命中证据块）

- rate：0.8400
- hits / restored / not_replaced / missing / no_parent：50 / 42 / 8 / 0 / 0

## 全量均值（含模糊配对，不作为检索主指标）

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| all_required_documents_at_5 | 1.0000 | 100.0% |
| document_recall_at_5 | 1.0000 | 100.0% |
| evidence_group_recall_at_5 | 0.4940 | 100.0% |
| exact_candidate_recall_at_20 | 0.6286 | 100.0% |
| exact_candidate_recall_at_50 | 0.8399 | 100.0% |
| exact_candidate_recall_at_actual_depth | 0.9286 | 100.0% |
| fact_mrr | 0.4018 | 100.0% |
| mrr | 0.6071 | 100.0% |
| precision_at_5 | 0.1929 | 100.0% |
| quote_coverage_at_5 | 0.4381 | 100.0% |
| recall_at_5 | 0.4149 | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 100.0% |
| structured_required_fact_coverage_at_5 | 0.4881 | 100.0% |

## 检索侧客观指标（Recall@5 / MRR 等）

块级 Recall/MRR 按 chunk id 精确匹配，文档 Recall 按 source 匹配；
quote/required-fact coverage 按归一化后的块内文字与数值边界匹配。
与上面的 RAGAS judge 指标分开读。

| 指标 | 均值 |
|---|---:|
| all_required_documents_at_5 | 1.0000 |
| document_recall_at_5 | 1.0000 |
| evidence_group_recall_at_5 | 0.4940 |
| exact_candidate_recall_at_20 | 0.6286 |
| exact_candidate_recall_at_50 | 0.8399 |
| exact_candidate_recall_at_actual_depth | 0.9286 |
| fact_mrr | 0.4018 |
| mrr | 0.6071 |
| precision_at_5 | 0.1929 |
| quote_coverage_at_5 | 0.4381 |
| recall_at_5 | 0.4149 |
| required_fact_coverage_at_5 | 0.0000 |
| structured_required_fact_coverage_at_5 | 0.4881 |

## 检索阶段诊断

| 阶段字段 | 均值 |
|---|---:|
| coarse_candidate_count | 150.0000 |
| coarse_source_count | 8.6786 |
| coarse_duplicate_source_slots | 141.3214 |
| rerank_candidate_count | 50.0000 |
| rerank_source_count | 7.0357 |
| rerank_duplicate_source_slots | 42.9643 |
| final_candidate_count | 5.0000 |
| final_source_count | 3.0714 |
| final_duplicate_source_slots | 1.9286 |
- 全部 resolved fusion modes：{"weighted": 28}；cliff enabled：28
- 全部 dense evidence survival recall：N/A；覆盖率：0.0%
- 全部 fusion evidence survival recall：0.8470；覆盖率：100.0%
- 全部 rerank evidence survival recall：0.5196；覆盖率：100.0%
- 全部 final_top5 evidence survival recall：0.4000；覆盖率：100.0%
- 独立正例 resolved fusion modes：{"weighted": 18}；cliff enabled：18
- 独立正例 dense evidence survival recall：N/A；覆盖率：0.0%
- 独立正例 fusion evidence survival recall：0.8593；覆盖率：100.0%
- 独立正例 rerank evidence survival recall：0.5630；覆盖率：100.0%
- 独立正例 final_top5 evidence survival recall：0.3769；覆盖率：100.0%
- cross_doc resolved fusion modes：{"weighted": 10}；cliff enabled：10
- cross_doc dense evidence survival recall：N/A；覆盖率：0.0%
- cross_doc fusion evidence survival recall：0.9217；覆盖率：100.0%
- cross_doc rerank evidence survival recall：0.6633；覆盖率：100.0%
- cross_doc final_top5 evidence survival recall：0.4867；覆盖率：100.0%
- explicit resolved fusion modes：{"weighted": 8}；cliff enabled：8
- explicit dense evidence survival recall：N/A；覆盖率：0.0%
- explicit fusion evidence survival recall：0.7812；覆盖率：100.0%
- explicit rerank evidence survival recall：0.4375；覆盖率：100.0%
- explicit final_top5 evidence survival recall：0.2396；覆盖率：100.0%
- vague resolved fusion modes：{"weighted": 10}；cliff enabled：10
- vague dense evidence survival recall：N/A；覆盖率：0.0%
- vague fusion evidence survival recall：0.8250；覆盖率：100.0%
- vague rerank evidence survival recall：0.4417；覆盖率：100.0%
- vague final_top5 evidence survival recall：0.4417；覆盖率：100.0%

- sparse survival：unavailable（Milvus hybrid API 未返回内部独立榜单；使用 L0/L1 跨层对比）。
- coarse/rerank chunk IDs 与逐条 evidence survival 见 records.jsonl / per_query.csv。

## 运行可复现配置

- dataset_fingerprint：e40ddab4ca724a8476257287aa424fc5ee99c0112945912626ec875a3881f996
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
