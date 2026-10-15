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
| all_required_documents_at_5 | 0.7222 | 1.0000 | 0.2778 | [0.1111, 0.5000] | improvement | 100.0% |
| document_recall_at_5 | 0.8704 | 1.0000 | 0.1296 | [0.0463, 0.2315] | improvement | 100.0% |
| evidence_group_recall_at_5 | 0.3380 | 0.5046 | 0.1667 | [-0.0278, 0.3611] | inconclusive | 100.0% |
| exact_candidate_recall_at_20 | N/A | N/A | N/A | N/A | no_data | 0.0% |
| exact_candidate_recall_at_50 | N/A | N/A | N/A | N/A | no_data | 0.0% |
| exact_candidate_recall_at_actual_depth | 0.5611 | 0.9444 | 0.3833 | [0.2176, 0.5639] | improvement | 100.0% |
| fact_mrr | 0.1898 | 0.3935 | 0.2037 | [0.0509, 0.3634] | improvement | 100.0% |
| mrr | 0.3148 | 0.6389 | 0.3241 | [0.1111, 0.5232] | improvement | 100.0% |
| precision_at_5 | 0.1778 | 0.2000 | 0.0222 | [-0.0444, 0.0889] | inconclusive | 100.0% |
| quote_coverage_at_5 | 0.3318 | 0.4175 | 0.0857 | [-0.1032, 0.2762] | inconclusive | 100.0% |
| recall_at_5 | 0.3324 | 0.3954 | 0.0630 | [-0.1241, 0.2500] | inconclusive | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| structured_required_fact_coverage_at_5 | 0.3009 | 0.4954 | 0.1944 | [0.0000, 0.3889] | inconclusive | 100.0% |

