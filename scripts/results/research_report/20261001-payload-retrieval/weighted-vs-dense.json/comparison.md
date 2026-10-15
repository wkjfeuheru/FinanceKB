# A/B 对比：hybrid_vs_dense

- 判定：inconclusive
- 可比性：verified；原因：无
- 主指标：recall_at_5
- 护栏：precision_at_5, mrr
- Bootstrap：10000 次（seed=7）

## Gold（n=18）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|
| document_recall_at_5 | 0.8704 | 0.8704 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_20 | 0.4630 | 0.3518 | -0.1111 | [-0.2500, 0.0000] | inconclusive | 100.0% |
| exact_candidate_recall_at_50 | 0.5741 | 0.5741 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| mrr | 0.1667 | 0.1944 | 0.0278 | [0.0000, 0.0833] | inconclusive | 100.0% |
| precision_at_5 | 0.0333 | 0.0444 | 0.0111 | [0.0000, 0.0333] | inconclusive | 100.0% |
| quote_coverage_at_5 | 0.0463 | 0.0463 | 0.0000 | [0.0000, 0.0000] | inconclusive | 100.0% |
| recall_at_5 | 0.1018 | 0.1574 | 0.0556 | [0.0000, 0.1667] | inconclusive | 100.0% |
| required_fact_coverage_at_5 | 0.3796 | 0.4352 | 0.0556 | [0.0000, 0.1667] | inconclusive | 100.0% |

## Silver（n=0）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|

