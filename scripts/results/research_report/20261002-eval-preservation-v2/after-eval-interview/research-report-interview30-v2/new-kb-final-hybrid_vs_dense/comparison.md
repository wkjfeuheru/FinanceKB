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
| all_required_documents_at_5 | 0.8571 | 0.7857 | -0.0714 | [-0.1786, 0.0000] | inconclusive | 100.0% |
| document_recall_at_5 | 0.9286 | 0.8929 | -0.0357 | [-0.0893, 0.0000] | inconclusive | 100.0% |
| evidence_group_recall_at_5 | 0.3214 | 0.2857 | -0.0357 | [-0.0893, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_20 | 0.5274 | 0.5274 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_50 | 0.8577 | 0.8577 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_actual_depth | 0.9768 | 0.9768 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| fact_mrr | 0.2366 | 0.2009 | -0.0357 | [-0.0893, 0.0000] | inconclusive | 100.0% |
| mrr | 0.3839 | 0.3125 | -0.0714 | [-0.1786, 0.0000] | inconclusive | 100.0% |
| precision_at_5 | 0.1500 | 0.1214 | -0.0286 | [-0.0714, 0.0000] | inconclusive | 100.0% |
| quote_coverage_at_5 | 0.3120 | 0.2691 | -0.0429 | [-0.1071, 0.0000] | inconclusive | 100.0% |
| recall_at_5 | 0.3030 | 0.2554 | -0.0476 | [-0.1191, 0.0000] | inconclusive | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| structured_required_fact_coverage_at_5 | 0.2976 | 0.2619 | -0.0357 | [-0.0893, 0.0000] | inconclusive | 100.0% |

