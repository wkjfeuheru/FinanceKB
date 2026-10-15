# A/B 对比：hybrid_vs_dense

- 判定：no_decision
- 可比性：verified；原因：无
- 主指标：recall_at_5
- 护栏：precision_at_5, mrr
- Bootstrap：10000 次（seed=7）

## Gold（n=0）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|

## Silver（n=18）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|
| all_required_documents_at_5 | 0.8333 | 0.7222 | -0.1111 | [-0.2778, 0.0000] | inconclusive | 100.0% |
| document_recall_at_5 | 0.9167 | 0.8611 | -0.0556 | [-0.1389, 0.0000] | inconclusive | 100.0% |
| evidence_group_recall_at_5 | 0.3194 | 0.2917 | -0.0278 | [-0.0833, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_20 | 0.5519 | 0.5519 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_50 | 0.7787 | 0.7787 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_actual_depth | 0.9639 | 0.9639 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| fact_mrr | 0.2500 | 0.2222 | -0.0278 | [-0.0833, 0.0000] | inconclusive | 100.0% |
| mrr | 0.4028 | 0.3472 | -0.0556 | [-0.1667, 0.0000] | inconclusive | 100.0% |
| precision_at_5 | 0.1667 | 0.1444 | -0.0222 | [-0.0667, 0.0000] | inconclusive | 100.0% |
| quote_coverage_at_5 | 0.3133 | 0.2800 | -0.0333 | [-0.1000, 0.0000] | inconclusive | 100.0% |
| recall_at_5 | 0.3139 | 0.2769 | -0.0370 | [-0.1111, 0.0000] | inconclusive | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| structured_required_fact_coverage_at_5 | 0.2824 | 0.2546 | -0.0278 | [-0.0833, 0.0000] | inconclusive | 100.0% |

