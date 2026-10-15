# A/B 对比：cliff_on_off

- 判定：no_decision
- 可比性：verified；原因：无
- 主指标：precision_at_5
- 护栏：recall_at_5, mrr
- Bootstrap：10000 次（seed=7）

## Gold（n=0）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|

## Silver（n=18）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|
| all_required_documents_at_5 | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| document_recall_at_5 | 1.0000 | 1.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| evidence_group_recall_at_5 | 0.5046 | 0.5046 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_20 | 0.6537 | 0.6537 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_50 | 0.8065 | 0.8065 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_actual_depth | 0.9444 | 0.9444 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| fact_mrr | 0.3935 | 0.3935 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| mrr | 0.6389 | 0.6389 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| precision_at_5 | 0.2000 | 0.2000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| quote_coverage_at_5 | 0.4175 | 0.4175 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| recall_at_5 | 0.3954 | 0.3954 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| required_fact_coverage_at_5 | 0.0000 | 0.0000 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| structured_required_fact_coverage_at_5 | 0.4954 | 0.4954 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |

