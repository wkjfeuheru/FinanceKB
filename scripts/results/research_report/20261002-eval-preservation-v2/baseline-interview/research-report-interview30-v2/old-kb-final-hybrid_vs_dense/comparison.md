# A/B 对比：hybrid_vs_dense

- 判定：no_decision
- 可比性：verified；原因：无
- 主指标：recall_at_5
- 护栏：precision_at_5, mrr
- Bootstrap：10000 次（seed=7）

## Gold（n=0）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|

## Silver（n=28）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|
| all_required_documents_at_5 | 0.7857 | 0.8214 | 0.0357 | [0.0000, 0.1071] | inconclusive | 100.0% |
| document_recall_at_5 | 0.8988 | 0.9167 | 0.0179 | [0.0000, 0.0536] | inconclusive | 100.0% |
| evidence_group_recall_at_5 | 0.3333 | 0.2976 | -0.0357 | [-0.0893, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_20 | 0.6286 | 0.6286 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_50 | 0.8399 | 0.8399 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_actual_depth | 0.9286 | 0.9286 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| fact_mrr | 0.2098 | 0.1741 | -0.0357 | [-0.0893, 0.0000] | inconclusive | 100.0% |
| mrr | 0.3452 | 0.2738 | -0.0714 | [-0.1786, 0.0000] | inconclusive | 100.0% |
| precision_at_5 | 0.1643 | 0.1357 | -0.0286 | [-0.0714, 0.0000] | inconclusive | 100.0% |
| quote_coverage_at_5 | 0.3382 | 0.2953 | -0.0429 | [-0.1071, 0.0000] | inconclusive | 100.0% |
| recall_at_5 | 0.3268 | 0.2792 | -0.0476 | [-0.1191, 0.0000] | inconclusive | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| structured_required_fact_coverage_at_5 | 0.3095 | 0.2738 | -0.0357 | [-0.0893, 0.0000] | inconclusive | 100.0% |

