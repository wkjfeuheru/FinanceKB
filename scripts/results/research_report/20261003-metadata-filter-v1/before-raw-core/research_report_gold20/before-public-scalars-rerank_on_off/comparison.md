# A/B 对比：rerank_on_off

- 判定：no_decision
- 可比性：verified；原因：无
- 主指标：mrr
- 护栏：recall_at_5, precision_at_5
- Bootstrap：10000 次（seed=7）

## Gold（n=0）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|

## Silver（n=18）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|
| all_required_documents_at_5 | 0.8333 | 1.0000 | 0.1667 | [0.0000, 0.3333] | inconclusive | 100.0% |
| document_recall_at_5 | 0.9167 | 1.0000 | 0.0833 | [0.0000, 0.1667] | inconclusive | 100.0% |
| evidence_group_recall_at_5 | 0.3194 | 0.4491 | 0.1296 | [0.0370, 0.2315] | improvement | 100.0% |
| exact_candidate_recall_at_20 | N/A | N/A | N/A | N/A | no_data | 0.0% |
| exact_candidate_recall_at_50 | N/A | N/A | N/A | N/A | no_data | 0.0% |
| exact_candidate_recall_at_actual_depth | 0.4454 | 0.9639 | 0.5185 | [0.3380, 0.6991] | improvement | 100.0% |
| fact_mrr | 0.2292 | 0.3403 | 0.1111 | [0.0208, 0.2083] | improvement | 100.0% |
| mrr | 0.3611 | 0.5556 | 0.1944 | [0.0278, 0.3750] | improvement | 100.0% |
| precision_at_5 | 0.1556 | 0.2000 | 0.0444 | [-0.0222, 0.1222] | inconclusive | 100.0% |
| quote_coverage_at_5 | 0.2911 | 0.4052 | 0.1141 | [-0.0076, 0.2481] | inconclusive | 100.0% |
| recall_at_5 | 0.2954 | 0.3954 | 0.1000 | [-0.0222, 0.2296] | inconclusive | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| structured_required_fact_coverage_at_5 | 0.2824 | 0.4676 | 0.1852 | [0.0833, 0.2963] | improvement | 100.0% |

