# A/B 对比：hybrid_vs_dense

- 判定：no_decision
- 可比性：verified；原因：无
- 主指标：recall_at_5
- 护栏：precision_at_5, mrr
- Bootstrap：10000 次（seed=7）

## Gold（n=0）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|

## Silver（n=4）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|
| all_required_documents_at_5 | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| document_recall_at_5 | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| evidence_group_recall_at_5 | 0.5000 | 0.5000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_20 | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 50.0% |
| exact_candidate_recall_at_50 | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 50.0% |
| exact_candidate_recall_at_actual_depth | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 50.0% |
| fact_mrr | 0.1458 | 0.1458 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| mrr | 0.2916 | 0.2916 | 0.0000 | [0.0000, 0.0000] | inconclusive | 50.0% |
| precision_at_5 | 0.2000 | 0.2000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 50.0% |
| quote_coverage_at_5 | 0.5000 | 0.5000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| recall_at_5 | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 50.0% |
| required_fact_coverage_at_5 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| structured_required_fact_coverage_at_5 | 0.5000 | 0.5000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |

