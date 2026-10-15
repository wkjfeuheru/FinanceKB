# 评测报告：L5

- 模式：retrieval
- 样本数：18（含模糊配对；检索主指标看下面的独立正例）
- 完成/失败：18/0

## 独立正例（explicit + cross_doc）

不含 vague 配对题，也不含负样本

- 样本数：18

| 指标 | 均值 |
|---|---:|
| mrr | 0.2593 |
| precision_at_5 | 0.0889 |
| recall_at_5 | 0.2963 |

- 父块还原 parent_restore_rate：0.8750 （命中 8 块，还原 7，未替换 1）

## 分层

| 层 | n | recall_at_5 | mrr |
|---|---:|---:|---:|
| cross_doc | 10 | 0.2333 | 0.2833 |
| explicit | 8 | 0.3750 | 0.2292 |

## 父块还原（全部命中证据块）

- rate：0.8750
- hits / restored / not_replaced / missing / no_parent：8 / 7 / 1 / 0 / 0

## 全量均值（含模糊配对，不作为检索主指标）

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| mrr | 0.2593 | 100.0% |
| precision_at_5 | 0.0889 | 100.0% |
| recall_at_5 | 0.2963 | 100.0% |
