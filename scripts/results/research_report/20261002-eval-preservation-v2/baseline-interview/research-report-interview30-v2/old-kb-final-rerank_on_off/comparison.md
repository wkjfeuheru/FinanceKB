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
| all_required_documents_at_5 | 0.8214 | 1.0000 | 0.1786 | [0.0714, 0.3214] | improvement | 100.0% |
| document_recall_at_5 | 0.9167 | 1.0000 | 0.0833 | [0.0179, 0.1548] | improvement | 100.0% |
| evidence_group_recall_at_5 | 0.3333 | 0.4940 | 0.1607 | [0.0179, 0.3036] | improvement | 100.0% |
| exact_candidate_recall_at_20 | N/A | N/A | N/A | N/A | no_data | 0.0% |
| exact_candidate_recall_at_50 | N/A | N/A | N/A | N/A | no_data | 0.0% |
| exact_candidate_recall_at_actual_depth | 0.5571 | 0.9286 | 0.3714 | [0.2375, 0.5149] | improvement | 100.0% |
| fact_mrr | 0.1845 | 0.4018 | 0.2173 | [0.0923, 0.3482] | improvement | 100.0% |
| mrr | 0.2946 | 0.6071 | 0.3125 | [0.1458, 0.4732] | improvement | 100.0% |
| precision_at_5 | 0.1571 | 0.1929 | 0.0357 | [-0.0143, 0.0857] | inconclusive | 100.0% |
| quote_coverage_at_5 | 0.3239 | 0.4381 | 0.1143 | [-0.0291, 0.2592] | inconclusive | 100.0% |
| recall_at_5 | 0.3149 | 0.4149 | 0.1000 | [-0.0369, 0.2429] | inconclusive | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| structured_required_fact_coverage_at_5 | 0.3095 | 0.4881 | 0.1786 | [0.0357, 0.3214] | improvement | 100.0% |

