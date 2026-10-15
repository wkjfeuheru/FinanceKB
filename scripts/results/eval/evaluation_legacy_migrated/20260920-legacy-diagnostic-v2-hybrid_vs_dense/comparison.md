# A/B 对比：hybrid_vs_dense

- 判定：inconclusive
- 主指标：recall_at_5
- 护栏：precision_at_5, mrr
- Bootstrap：10000 次（seed=7）

## Gold（n=28）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|
| mrr | 0.5845 | 0.5905 | 0.0060 | [-0.0643, 0.0786] | inconclusive | 100.0% |
| precision_at_5 | 0.1571 | 0.1714 | 0.0143 | [-0.0143, 0.0500] | inconclusive | 100.0% |
| recall_at_5 | 0.6786 | 0.7321 | 0.0536 | [-0.0357, 0.1607] | inconclusive | 100.0% |

## Silver（n=0）

| 指标 | 基线 | 候选 | 差值 | 95% CI | 判定 | 覆盖率 |
|---|---:|---:|---:|---|---|---:|

