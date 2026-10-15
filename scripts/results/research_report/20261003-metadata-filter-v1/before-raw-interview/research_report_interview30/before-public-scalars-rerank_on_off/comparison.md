# A/B 对比：rerank_on_off

- 判定：no_decision
- 可比性：verified；原因：无
- 主指标：mrr
- 护栏：recall_at_5, precision_at_5
- Bootstrap：10000 次（seed=7）

## Gold（n=0）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|

## Silver（n=28）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|
| all_required_documents_at_5 | 0.8929 | 1.0000 | 0.1071 | [0.0000, 0.2500] | inconclusive | 100.0% |
| document_recall_at_5 | 0.9464 | 1.0000 | 0.0536 | [0.0000, 0.1250] | inconclusive | 100.0% |
| evidence_group_recall_at_5 | 0.3214 | 0.4226 | 0.1012 | [0.0357, 0.1786] | improvement | 100.0% |
| exact_candidate_recall_at_20 | N/A | N/A | N/A | N/A | no_data | 0.0% |
| exact_candidate_recall_at_50 | N/A | N/A | N/A | N/A | no_data | 0.0% |
| exact_candidate_recall_at_actual_depth | 0.4113 | 0.9768 | 0.5655 | [0.4226, 0.7083] | improvement | 100.0% |
| fact_mrr | 0.2143 | 0.3318 | 0.1176 | [0.0402, 0.2009] | improvement | 100.0% |
| mrr | 0.3393 | 0.5179 | 0.1786 | [0.0536, 0.3125] | improvement | 100.0% |
| precision_at_5 | 0.1429 | 0.1857 | 0.0429 | [-0.0071, 0.0929] | inconclusive | 100.0% |
| quote_coverage_at_5 | 0.2977 | 0.3874 | 0.0897 | [0.0040, 0.1844] | improvement | 100.0% |
| recall_at_5 | 0.2911 | 0.3792 | 0.0881 | [0.0048, 0.1774] | improvement | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| structured_required_fact_coverage_at_5 | 0.2976 | 0.4345 | 0.1369 | [0.0595, 0.2202] | improvement | 100.0% |

