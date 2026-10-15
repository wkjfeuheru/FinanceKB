# A/B 对比：retrieval_only_ab

- 判定：no_decision
- 可比性：mismatch；原因：kb_fingerprint
- 主指标：recall_at_5
- 护栏：precision_at_5, chunk_mrr
- Bootstrap：10000 次（seed=7）

## Gold（n=18）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|
| document_recall_at_5 | N/A | N/A | N/A | N/A | no_data | 0.0% |
| exact_candidate_recall_at_20 | N/A | N/A | N/A | N/A | no_data | 0.0% |
| exact_candidate_recall_at_50 | N/A | N/A | N/A | N/A | no_data | 0.0% |
| mrr | 0.2407 | 0.1944 | -0.0463 | [-0.1944, 0.0741] | inconclusive | 100.0% |
| precision_at_5 | 0.0667 | 0.0444 | -0.0222 | [-0.0556, 0.0000] | inconclusive | 100.0% |
| quote_coverage_at_5 | N/A | N/A | N/A | N/A | no_data | 0.0% |
| recall_at_5 | 0.2407 | 0.1574 | -0.0833 | [-0.2222, 0.0000] | inconclusive | 100.0% |
| required_fact_coverage_at_5 | N/A | N/A | N/A | N/A | no_data | 0.0% |

## Silver（n=0）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|

