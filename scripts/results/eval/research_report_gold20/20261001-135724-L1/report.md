# 评测报告：L1

- 模式：retrieval
- 样本数：18（含模糊配对；检索主指标看下面的独立正例）
- 完成/失败：18/0

## 独立正例（explicit + cross_doc）

不含 vague 配对题，也不含负样本

- 样本数：18

| 指标 | 均值 |
|---|---:|
| mrr | 0.1250 |
| precision_at_5 | 0.0444 |
| recall_at_5 | 0.1574 |

- 父块还原 parent_restore_rate：0.7500 （命中 4 块，还原 3，未替换 1）

## 分层

| 层 | n | recall_at_5 | mrr |
|---|---:|---:|---:|
| cross_doc | 10 | 0.0833 | 0.1500 |
| explicit | 8 | 0.2500 | 0.0938 |

## 父块还原（全部命中证据块）

- rate：0.7500
- hits / restored / not_replaced / missing / no_parent：4 / 3 / 1 / 0 / 0

## 全量均值（含模糊配对，不作为检索主指标）

| 指标 | 均值 | 覆盖率 |
|---|---:|---:|
| mrr | 0.1250 | 100.0% |
| precision_at_5 | 0.0444 | 100.0% |
| recall_at_5 | 0.1574 | 100.0% |

## 检索侧客观指标（Recall@5 / MRR 等）

这些指标按标注证据的 chunk id 精确匹配计算，是检索能力的地面真值，
与上面的 RAGAS judge 指标分开读。

| 指标 | 均值 |
|---|---:|
| mrr | 0.1250 |
| precision_at_5 | 0.0444 |
| recall_at_5 | 0.1574 |
