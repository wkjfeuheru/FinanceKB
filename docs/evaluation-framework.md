# 金融研报 RAG 评测体系方案

> 版本 2.0 ｜ 适用范围：FinanceKB 金融研报问答系统 ｜ 面向读者：工程与算法团队
>
> 本文档是评测体系的**总体设计**：评测集构建、指标选取、判定标准与评测流水线。
> 文中「实现落点」均指向仓库现有代码；标注 **[待补]** 的是需要新建的部分。

---

## 0. 总览

系统要被证明的三件事：

1. **找得全**——该召回的证据块都进了上下文（检索链路）；
2. **答得准**——答案与参考答案一致、忠于上下文、数字正确、引用可溯源（生成链路）；
3. **不乱答**——库里没有或前提错误时拒答，而不是编造（拒答与安全）。

体系一图流：

```
评测集（四层）                 运行器                指标                  判定                  回流
┌──────────────┐        ┌─────────────┐      ┌──────────────┐      ┌──────────────┐      ┌─────────────┐
│ gold 冻结集    │        │ A/B 双臂执行  │      │ 检索 8 项      │      │ 门禁 smoke/   │      │ bad case    │
│ silver 扩展池  │──────▶ │ (ab_runner)  │────▶ │ 生成 9 项      │────▶ │ release 策略  │────▶ │ 入册回归集   │
│ 锚点样本       │        │ 子进程隔离    │      │ 系统 3 项      │      │ 人工校准      │      │ 数据集迭代   │
│ 回归集        │        └─────────────┘      └──────────────┘      └──────────────┘      └─────────────┘
└──────────────┘
```

四层评测集的分工：

| 层 | 规模 | 用途 | 审核要求 |
|----|------|------|----------|
| **gold 冻结集** | 60–90 题 | 门禁判定、发版结论、跨版本对比的唯一事实源 | 100% 人工标注 + 双人审核 |
| **silver 扩展池** | 150–250 题 | 日常回归、细粒度诊断、A/B 提升统计功效 | 自动生成 + 10% 抽检 |
| **锚点样本** | 12–20 条 | 监控 LLM judge 尺度漂移，每批必掺 | 人工构造，期望分数为区间 |
| **回归集** | 持续增长（当前 4 条起步） | 已修复 bad case 的防复发 | 修复验证后入册 |

---

## 1. 评测集构建

### 1.1 规模建议

**总量建议 300 ± 100 题**（不含锚点），按上表分层。规模依据：

- 主指标 Recall@5 上，100 题约能可靠检出 **5pp** 的差异，200 题约 **3.5pp**，300 题约 **2.5pp**；
  低于 50 题时，A/B 差异基本淹没在噪声里（`src/eval/statistics.py` 的配对 Bootstrap 会给出置信区间，`noise_floor` 自动抑制无意义提升）；
- gold 冻结集不需要大：它决定「能不能合、能不能发」，60–90 题已足够检出回归，且每一题都要人工审核，成本高；
- silver 池承担统计功效：数字越大，日常实验的置信区间越窄。

分阶段落地：

| 阶段 | 目标规模 | 说明 |
|------|----------|------|
| P0（当前已完成） | gold 20 题（`research_report_gold20.jsonl`）+ interview 30 题 | 跑通门禁与报告链路 |
| P1 | gold 60 题 + silver 90 题（对齐 `dataset.py::DEFAULT_QUOTAS`：gold 30/15/15，silver 50/25/15） | 补齐拒答、时效、计算题型 |
| P2 | gold 90 题 + silver 200 题 + 锚点 16 条 + 回归集 ≥20 条 | 每类报告类型 × 题型覆盖矩阵达标 |

### 1.2 题型分类与占比

现有 schema 的 `question_type` 枚举为 `single_hop / multi_hop / negative`（`src/eval/dataset.py`），用
`provenance.stratum`（`explicit / cross_doc / vague / negative`）和 `answer_type`（`factual / list / comparison / procedural`）
做业务侧细分。业务题型按下方占比扩展，**新增题型通过 `question_type` 扩枚举实现**（`test_set.py`、校验逻辑同步改）：

| # | 业务题型 | 映射到 schema | 占比 | 300 题示例 | 考察能力 |
|---|----------|---------------|------|-----------|----------|
| 1 | 单文档事实查询 | single_hop / factual | 30% | 90 | 基础定位与抽取 |
| 2 | 数值密集读取（多指标罗列） | single_hop / list | 15% | 45 | 表格与财务摘要读取 |
| 3 | 数值计算（同比、占比、差值） | single_hop / procedural | 8% | 24 | 读取 + 算术 |
| 4 | 多文档对比 / 多跳 | multi_hop / comparison | 15% | 45 | 跨文档综合 |
| 5 | 汇总列表（如「列出某行业所有覆盖标的」） | single_hop / list | 10% | 30 | 多块聚合 |
| 6 | 时效性问答（「最新一期」「2026 年以来」） | single_hop / factual + `stratum=timeliness` | 7% | 21 | 日期元数据过滤、版本选择 |
| 7 | 元数据过滤问答（指定券商/类别/日期） | 任意类型 + `filters` 标注 | 5% | 15 | 过滤策略正确性 |
| 8 | 拒答类（前提错误 / 超出知识库 / 时间超出范围） | negative | 10% | 30 | 不编造 |

要点：

- **拒答类占 10% 是金融场景的底线要求**。预期拒答的三种子类至少各占 1/3；
  每条必须写 `expected_behavior`（`dataset.py` 已强制校验负样本必须有它、且不得带 evidence）。
- 数值计算题的查询里**不要出现答案**，标注里必须含计算公式与容差（见 1.4）。
- 时效性题必须同时标注「期望命中的文档日期」，用于核对系统是否真的取了最新版本（系统侧依赖 `date` 元数据与 `is_current` 版本过滤）。

### 1.3 题目来源

| 来源 | 方式 | 定位 | 建议占比（长期） |
|------|------|------|------------------|
| 真实用户日志 | 从线上问答日志脱敏采样；无答案/低赞 query 优先 | 最贴近真实分布；gold 层时效/过滤题主要来源 | 30–40%，逐月滚动新增 |
| 研报原文改写 | 标注员从证据块出发**反向出题**（先选 evidence 再写 query） | 保证证据可标注、可定位；silver 池主力 | 40% |
| 模型批量生成 + 人工审 | `src/eval/testset_generator.py`（单跳:多跳:负样本 ≈ 7:2:1）打底 | 快速扩量；只进 silver 层 | 20% |
| 专家构造 | 投研/合规同事手工出题 | 拒答类、计算类、时效类、边界陷阱题；gold 层主力 | 按题型缺口补充 |

来源纪律：

- 同一篇研报贡献的**证据占比不得超过 20%**（`dataset.freeze(max_source_share=0.20)` 已强制）；
- 生成题必须经人工审核才可入库（`review_status: pending → approved`），审核时重点核对：问题是否可答、证据是否充分、负样本是否真的无答案。

### 1.4 标注规范

每条评测样本是一个 `EvalCase`（JSONL，schema 1.0/2.0，字段校验见 `dataset.py::validate`）：

| 字段 | 规范 |
|------|------|
| `id` | 全局唯一，建议 `<来源缩写>-<题型>-<序号>`，如 `log-numeric-017` |
| `query` | 用户原话风格；不得包含答案；同义改写题需注明 |
| `question_type` / `difficulty` / `answer_type` | 见 1.2 与 1.5 |
| `reference_answer` | 标准答案。事实题：一句话可直接核对；列表/对比题：要点清单（每条要点独立可判对错） |
| `required_facts` | 答案必须覆盖的事实要点（完整性指标的分子来源） |
| `evidence[]` | 证据引用：`source`（文件名）+ `document_version` + `chunk_id` + `quote`（原文引文，带 `quote_start/end` 偏移与 `document_sha256`，schema 2.0 强制）+ `location`（页码等）。**引文禁止截断数字/英文 token/HTML 标签**（`_safe_quote_boundaries` 已校验） |
| `facts[]`（schema 2.0） | 结构化事实要求：`aliases`（同义表述组，任一命中即算）与 `acceptable_evidence_sets`（多证据 AND 组合，须同源文档内成立）。数值计算题在此声明公式与**容差**（如相对误差 ≤0.5%）**[待补：计算题容差字段]** |
| `expected_behavior` | 仅负样本：`refuse_premise_error` / `refuse_out_of_scope` / `refuse_time_out_of_range` |
| `tier` / `review_status` / `reviewer` / `review_note` | gold 须 `approved`，silver 须 `auto_pass` 才能进冻结集 |

评分细则（人工标注与 LLM judge 共用同一口径，0–1 连续或三档离散）：

- 事实题：答案数值/实体与标准答案一致 = 1；方向或口径对但数值错 = 0.5；编造 = 0；
- 列表/对比题：按要点覆盖率给分（命中要点数 / 应有要点数），漏要点不扣「忠实度」只扣「完整性」；
- 计算题：结果在容差内 = 1；方法对结果错 = 0.5；方法错 = 0；
- 拒答题：正确拒答并说明原因 = 1；拒答但理由错误 = 0.5；编造答案 = 0。

审核工作流：`export_review()` 导出 CSV → 人工逐条批注 → `import_review()` 回填。gold 层双人独立审核，分歧由第三人仲裁。

冻结与版本：每次构建评测集用 `dataset.freeze()` 产出冻结 JSONL + manifest（含 `dataset_version`、`dataset_sha256`、`kb_fingerprint`、各层配额、来源分布）。**知识库指纹不一致时拒绝评测**，防止「指标变化其实是库变了」的假信号。

### 1.5 覆盖度与难度分层

覆盖矩阵（冻结前用脚本校验，**[待补]**：在 `dataset.py` 冻结校验中增加矩阵配额检查）：

| 维度 | 要求 |
|------|------|
| 报告类型 | 个股 / 行业 / 宏观 / 策略 四类都要有题（投研类系统的主轴） |
| 知识库类别 | 投研为主（≥70%），合规风控 / 业务运营 / 管理类合计 ≥20% |
| 券商来源 | 单一券商证据占比 ≤20%（与 `max_source_share` 一致） |
| 时间分布 | 近 3 个月 / 半年 / 一年以上各 ≥15%，保证时效题有意义 |
| 难度 | easy 40% / medium 40% / hard 20% |

难度定义：

| 难度 | 判定标准 |
|------|----------|
| easy | 单块证据、原文字面可答、无元数据过滤 |
| medium | 需要同义改写匹配 / 跨段聚合 / 带元数据过滤 / 数值密集 |
| hard | 多跳跨文档 / 数值计算 / 时效性版本选择 / 存在近似干扰文档（同公司多期报告、同行业多标的同时在库） |

「近似干扰项」是金融研报库最重要的难度来源：同一公司连续多期财报、同一行业多家公司的相似段落，是检索混淆和数值张冠李戴的主要根因，hard 题必须包含这类干扰。

---

## 2. 评测指标选取

### 2.1 检索环节

生产链路：混合召回（bge-m3 稠密 + BM25 稀疏，RRF）粗排 50 → BGE 重排取 Top-5。主 K 取 **5**（与线上上下文条数一致），诊断辅以 K=3/10。

| 指标 | 定义与计算 | 取值 | 实现落点 |
|------|-----------|------|----------|
| **Recall@K（块级）** | Top-K 检索块命中标注证据块的比例：`|hits∩evidence| / |evidence|`，同块去重 | 0–1 | `metrics.py::compute_chunk_retrieval_metrics` |
| **evidence_group_recall@5** | 按标注证据组合计分：一条 fact 的 AND 证据组必须**在同一份同源正文内全部命中**才算完成 | 0–1 | `metrics.py::compute_structured_fact_metrics` |
| **Precision@K** | 固定分母 K 的命中占比，衡量 Top-K 里证据的密度（防「靠堆 K 提召回」） | 0–1 | 同 `compute_chunk_retrieval_metrics` |
| **MRR（fact_mrr）** | 每条 fact 首个完成其证据组的排名倒数均值；衡量「证据排多前」 | 0–1 | `metrics.py::fact_completion_ranks` |
| **nDCG@K** **[待补]** | 证据按重要性（主要证据=2、辅助=1）加权的折损累计增益；对排序质量比 MRR 更敏感 | 0–1 | 新增实现，`chunk_id` 命中作 relevance |
| **context_precision**（RAGAS） | 检索噪音：上下文中真正对答题有用的块占比（LLM 逐块判定有用性后按排名加权） | 0–1 | `ragas_eval.py` |
| **context_recall**（RAGAS） | 知识盲区：标准答案中的要点能在检索上下文中找到声明的比例 | 0–1 | `ragas_eval.py` |
| **all_required_documents@5** | 多跳题要求的所有来源文档是否都进了 Top-5（多跳生成的硬前提） | 0/1 | `metrics.py::compute_structured_fact_metrics` |
| **document_recall@K** | 按唯一 source 计的文档级召回，诊断「块没中但文档中了」的粒度问题 | 0–1 | `metrics.py::compute_document_recall` |
| **parent_restore_rate** | 子块命中后，PostgreSQL 父块能否无损还原引文（`restored / (restored+missing)`，未替换不算失败） | 0–1 | `metrics.py::assess_parent_restore` |
| **filtered_recall@5** **[待补]** | 仅统计带 `filters` 标注的题：过滤条件下证据召回率，专防「过滤过严导致证据被滤掉」 | 0–1 | 复用 `compute_chunk_retrieval_metrics`，按 stratum 切片 |

选择说明：**Recall@5 是检索主指标**（门禁用它），context_recall 补充「标注遗漏」视角（RAGAS 从答案要点反推，能发现标注没标到的证据）；MRR 与 context_precision 管排序与噪音；parent_restore_rate 守护父子块架构的完整性。

### 2.2 生成环节

| 指标 | 定义与计算 | 取值 | 实现落点 |
|------|-----------|------|----------|
| **答案正确性 correctness** | 答案与参考答案的一致性（含数值、实体、方向）。确定性部分（数值、引文覆盖）程序判，语义部分 LLM judge 按 1.4 评分细则给 0–1 | 0–1 | LLM judge + `compute_text_coverage` |
| **忠实度 faithfulness** | 答案中每条声明能否被检索上下文支撑：`被支撑声明数 / 总声明数`。不核对与参考答案是否一致，只核对「说的每句话有没有出处」 | 0–1 | `ragas_eval.py` + 锚点校准 |
| **答案相关性 answer_relevancy** | 答案是否切题（防跑偏、防答非所问）。RAGAS 实现为「从答案反推问题与原问题的相似度」 | 0–1 | `ragas_eval.py` |
| **答案完整性** | `required_facts` 要点覆盖率：每个要点在其 `aliases` 任一表述、且来源文档合规时计 1 | 0–1 | `metrics.py::compute_structured_fact_metrics`（structured_required_fact_coverage@5） |
| **引用准确性 citation_precision / recall** **[待补]** | precision：答案中带引用编号的声明，其引用指向的文档确实支撑该声明；recall：应引用的声明中实际带引用的比例。基于引用账本（声明→引用编号→source）程序核对 | 0–1 | `answer/citations.py` 账本 + judge 抽检 |
| **数值精度 numeric_accuracy** **[待补]** | 答案中所有可抽取数值与参考答案比对：绝对值命中、单位与量级一致、同比方向（+/-）一致。计算题按 1.4 容差判 | 0–1 | `answer/faithfulness.py`（数字/指标/期间/方向核验）扩展成对答案集的比对 |
| **拒答准确率** | 两个方向分开算：① 期望拒答题的正确拒答率（recall_refusal）；② 正常题的误拒率（false_refusal_rate = 1 − 正常题正常作答率） | 0–1 | runner 判定输出格式 + **[待补]** 入指标聚合 |
| **格式合规** **[待补，可选]** | 引用编号规范、无 prompt 泄露、语言与提问一致 | 0/1 | 规则校验 |
| **端到端总体分 overall** | 人工 1–5 分的总体质量分（不进门禁，用于与自动指标对齐） | 1–5 | 人工标注 |

指标间关系（排障时按此归因）：

- correctness 低 + context_recall 高 → 检索没问题，**生成/理解有问题**；
- correctness 低 + context_recall 低 → **检索先背锅**，先修检索再评生成；
- faithfulness 低 → 幻觉，优先级最高（金融场景红线）；
- 完整性低 + faithfulness 高 → 答案没编但漏要点 → 查上下文截断与生成长度；
- 数值精度低但文本覆盖高 → 数字被截断/重排/换算出错 → 查 chunk 边界与表格解析。

### 2.3 系统级指标

| 指标 | 定义 | 用途 |
|------|------|------|
| 延迟 p50/p95 | retrieval 与 generation 分段计时（ms） | 门禁预算：p95 不得超过基线 **1.25×**（`gate.py` release 策略） |
| 单题成本 | LLM token 消耗 / 题 | 成本回归，防止 prompt 膨胀 |
| 异常率 | 运行失败 / 超时 / 空答案占比 | 稳定性监控，>2% 先修稳定性再谈质量 |

---

## 3. 判断标准与判定方法

### 3.1 指标分级与阈值

三档口径：**优秀**（可作为卖点对外宣传）/ **合格**（门禁线，低于即阻断）/ **不合格**（阻断并记 P0 问题）。
初次建线方法：用当前系统跑全量基线，把「业务可接受的最低水平」与「基线 +1 个噪声带宽」对齐；此后每季度回顾一次，评分细则变更时阈值同步重标。

| 指标 | 优秀 | 合格（门禁线） | 不合格 |
|------|------|----------------|--------|
| Recall@5 | ≥0.90 | ≥0.80 | <0.70 |
| evidence_group_recall@5 | ≥0.85 | ≥0.75 | <0.65 |
| context_precision | ≥0.85 | ≥0.75 | <0.65 |
| context_recall | ≥0.90 | ≥0.80 | <0.70 |
| MRR (fact_mrr) | ≥0.75 | ≥0.65 | <0.50 |
| all_required_documents@5（多跳题） | 1.0 | ≥0.90 | <0.80 |
| parent_restore_rate | ≥0.95 | ≥0.90 | <0.80 |
| correctness | ≥0.85 | ≥0.75 | <0.65 |
| faithfulness | ≥0.92 | **≥0.85（红线）** | <0.80 |
| answer_relevancy | ≥0.90 | ≥0.80 | <0.70 |
| 答案完整性 | ≥0.85 | ≥0.75 | <0.65 |
| 引用准确率 precision | ≥0.95 | ≥0.90 | <0.85 |
| 数值精度 | ≥0.99 | ≥0.95 | <0.90 |
| 拒答 recall（期望拒答题） | ≥0.95 | ≥0.90 | <0.80 |
| 误拒率（正常题） | ≤0.05 | ≤0.10 | >0.15 |
| 延迟 p95 | ≤1.1× 基线 | ≤1.25× 基线 | >1.5× 基线 |

金融场景的两条**一票否决红线**（违反任何一条，其余分数再好也阻断发布）：

1. faithfulness < 0.85（编造比漏答严重）；
2. 期望拒答题出现「编造答案」率 > 10%（`expected_behavior` 被违反）。

除绝对阈值外，门禁同时执行**相对判定**（`gate.py`）：主指标回退超过 `max_regression` 或配对 Bootstrap 判定 regression 即阻断；提升未超噪声下限（`noise_suppressed`）不予采信；可比性（`comparability_status=verified`）未验证不得下结论。

### 3.2 评估方式选择：确定性优先，LLM 兜底，人工定标

| 方式 | 适用指标 | 原则 |
|------|----------|------|
| 确定性程序 | 块级 Recall/Precision/MRR、evidence_group、text_coverage、数值核验、引用定位、parent_restore、拒答判定、延迟 | **凡能程序判的绝不走 LLM**：便宜、可复现、无 judge 漂移 |
| LLM-as-Judge | correctness 语义部分、faithfulness、answer_relevancy、context_precision / context_recall | 只用于「语义等价性」这类程序判不了的问题 |
| 人工 | gold 标注审核、judge 抽检复评、overall 打分、争议终审 | 人只做两件事：定标准（标注）与盯标准（校准） |

### 3.3 LLM-as-Judge 设计要点

**绝对打分 prompt 的十项要点**（适用 faithfulness / correctness / relevancy / context_precision）：

1. **角色与任务分离**：system 只写「你是严格的事实核查评估员」，任务、上下文、答案放 user，避免角色与数据混注入；
2. **逐声明拆解**：先让 judge 把答案拆成原子声明列表，再逐条判 `supported / unsupported / not_enough_information`，最后聚合成分数——直接要总分不稳定；
3. **评分细则内嵌**：把 1.4 的 0 / 0.5 / 1 定义原文写进 prompt，并给 2–3 个**带期望分数的 few-shot 样例**（可直接从锚点样本取，few-shot 与锚点同步更新）;
4. **明确「仅以提供的上下文为准」**：禁止 judge 使用自身金融知识补判（它知道的可能是错的或过时的）；
5. **明确「简洁准确优于详细冗长」**：压制长度偏差（`judge_bias.py` 会监控分数~长度秩相关）;
6. **输出 JSON schema**（`{"claims": [...], "score": 0.87}`），`temperature=0`，解析失败重试一次后记 null 不进分母;
7. **打分与解释分离**：先输出结构化判定，解释放最后字段，防止解释措辞干扰评分聚合;
8. **数值声明交回程序**：prompt 要求 judge 对数字类声明只标「数字类」，具体核验走 `faithfulness.py` 的确定性比对;
9. **裁判模型与生成模型不同**（当前生成用 DeepSeek，裁判建议用另一家更强的模型），并**固定裁判模型版本**记录在报告里——换裁判 = 换尺子，历史分数不可直接对比;
10. **分项可追溯**：每题落盘逐声明判定明细（`records.jsonl` 已有此设计），bad case 不需要重跑即可复盘。

**成对裁判（A/B 场景）**：绝对分在两次运行间的漂移可能大于被测差异，「哪臂更好」必须同屏比较（`pairwise_judge.py`）：

- 顺序随机：按 `Random(seed:case_id)` 决定左右，双臂机会均等；
- 换序复评：同一题交换顺序再判一次，两次结论不一致的题标记为 `flip`，只统计一致题或按 0.5 计分；
- 输出 `order_flip_rate` 与 `first_position_win_rate` 进 judge_bias 报告。

**裁判质量的三道防线**：

| 防线 | 机制 | 判定 |
|------|------|------|
| 锚点样本（每批） | 12–20 条期望分数已知的人工构造样本掺入每批评估（`anchors.py`），期望是区间（如 faithfulness ∈ [0.9,1.0] / [0,0.3]） | 任一锚点越界 → 本批全部作废，先修 judge 再重跑 |
| 人工校准（每周） | 分层抽 50–100 条已评估样本人工按同一口径打分（`human_calibration.py`），算 MAE / 偏置 / 秩相关 | MAE ≤0.1、偏置 ≤0.05、秩相关 ≥0.7；超线则调整 few-shot / 重写细则，必要时给 judge 分加固定偏移 |
| 偏差审计（每周） | 位置 / 长度 / 自我偏好三类偏差量化（`judge_bias.py`） | `order_flip_rate` > 0.2 或长度秩相关 > 0.3 时修 prompt |

### 3.4 人工抽检比例

| 对象 | 比例 | 说明 |
|------|------|------|
| gold 层标注 | 100% 双人审核 | 分歧第三人仲裁；负样本审核重点：确认库里真的没有答案 |
| silver 层标注 | `auto_pass` 后随机抽 10% | 抽检不合格率 >10% 时该批次全部降级回 pending 重审 |
| LLM judge 自动分 | 每批按指标 × 分数段分层抽 5–10% | 低分段（<0.6）与临界段（0.75–0.9）必须覆盖——门禁判定发生在这两段 |
| 拒答题人工复看 | 100% | 拒答判定是程序规则，规则改动后全量复看一次 |
| 一致性门槛 | Cohen's κ ≥0.7（分档）/ MAE ≤0.1（连续） | 低于门槛先对齐标注口径再继续评测 |

### 3.5 评测频率与回归测试机制

触发矩阵：

| 触发条件 | 跑什么 | 判定 | 频率 |
|----------|--------|------|------|
| 改动检索 / 重排 / 规划 / 生成 prompt（PR 合入前） | **smoke 门禁**：gold 冻结集全量 + 锚点 + 可比性校验（`scripts/ab_rag.py` + `scripts/evaluate.py gate --policy smoke`） | 阻断式：主指标不回退、无锚点越界 | 每次 PR |
| 每日定时 | gold + silver 全量 + judge 全量指标 + 延迟/成本 | 非阻断，报告归档 + 指标趋势看板；越阈值告警 | 每晚 |
| 每周 | 人工校准抽样（50–100 条）+ judge 偏差审计 | 非阻断，超线修 judge | 每周 |
| 发版前 | **release 门禁**：gold 证据级（`evidence_grade=gold`，silver 结果不作数）+ 锚点校验 + 延迟预算 1.25× + 主指标回退检查（`gate --policy release`） | 阻断式 | 每次发版 |
| 知识库结构变更（换 embedding / reranker / 重建索引） | 全量评测 + `kb_fingerprint` 一致性校验；证据 chunk_id 全量重新映射与校验（`dataset.validate`） | 阻断式（指纹不匹配直接拒评） | 每次变更 |
| 线上 bad case 流入 | 自动入册候选池 → 周评审 → 确认复现后修复 → 修复验证通过后固化进回归集 | 回归集每次发版必跑 | 持续 |

回归测试机制的具体约定：

- **回归集入册三条标准**：① 能稳定复现的线上/评测 bad case；② 已有修复且验证通过；③ 与既有回归题不重复。三条同时满足才入册（`research_report_regression4.jsonl` 是雏形，目标季度末 ≥20 条）;
- **评测集不可变原则**：冻结集一旦发布不改题——改题 = 换尺子。要扩容走新一轮 freeze，manifest 记录 `dataset_version` 与 `dataset_sha256`；历史报告永远能对回当时的数据集版本;
- **结果归档**：每次运行落盘 `per_query.csv + records.jsonl + summary.json + report.md`（现有目录 `scripts/results/eval/<数据集>/<时间戳>/`），报告含：分层指标、与上一版的 delta、噪声下限、锚点状态、延迟分段；
- **门禁退出码语义**（`gate.py`）：`0` 通过 / `1` 阻断 / `2` 结论不足（inconclusive，如可比性未验证、提升在噪声内）——CI 中 `2` 告警不阻断，人工判断后必须落一个明确结论;
- **告警升级**：连续 2 次 smoke 阻断 → 升级为 P1；红线指标（faithfulness / 拒答编造率）任一次越线 → 直接 P0 回滚评估。

---

## 4. 评测流水线落地

### 4.1 组件与代码映射

| 流水线环节 | 现有实现 | 缺口 |
|-----------|----------|------|
| 数据集定义 / 校验 / 冻结 | `src/eval/dataset.py`（schema、审核工作流、freeze+manifest） | 难度×题型覆盖矩阵校验 **[待补]** |
| 题目生成 | `src/eval/testset_generator.py`（7:2:1） | 数值计算、时效性题型模板 **[待补]** |
| 检索指标 | `src/eval/metrics.py`（Recall/Precision/MRR/evidence_group/parent_restore） | nDCG、filtered_recall **[待补]** |
| 生成指标 | `src/eval/ragas_eval.py`（四指标）+ `answer/faithfulness.py`（数值核验） | 引用准确性、数值精度聚合、拒答指标 **[待补]** |
| A/B 执行 | `src/eval/ab_runner.py`（子进程隔离双臂）+ `ab_catalog.py`（实验目录） | — |
| 统计判定 | `src/eval/statistics.py`（配对 Bootstrap、噪声下限、护栏） | — |
| judge 质量防线 | `anchors.py` + `human_calibration.py` + `pairwise_judge.py` + `judge_bias.py` | — |
| 门禁 | `src/eval/gate.py`（smoke/release）+ `scripts/evaluate.py` | 红线指标检查项 **[待补]** |
| 回归集 | `research_report_regression4.jsonl` | 入册流程与发版必跑编排 **[待补]** |

### 4.2 常用入口

```powershell
python scripts/ab_rag.py --list                          # 查看可复现的 A/B 实验
python scripts/ab_rag.py --experiment hyde_on_off --fast --sample 5
python scripts/evaluate.py gate --report <report.json> --policy release  # 门禁判定（退出码 0/1/2）
python scripts/eval_recall.py --testset src/eval/data/research_report_gold20.jsonl --top-k 5 8 10 --only L5  # 召回-K 曲线（诊断）
```

### 4.3 分阶段实施计划

| 阶段 | 周期 | 交付物 | 验收标准 |
|------|------|--------|----------|
| P0（已完成） | — | gold20 + interview30 + 门禁链路 | `gate --policy smoke` 可用 |
| P1 | 2 周 | 题型扩枚举（计算/时效/过滤）、拒答与引用指标、锚点扩到 16 条、gold 扩到 60 题 | release 门禁全项可判；judge 校准首次达标（MAE ≤0.1） |
| P2 | 4 周 | silver 200 题、覆盖矩阵校验、每日定时任务 + 趋势看板、回归集入册流程 | 300±100 题全量跑通；bad case 周评审机制运转 |
| P3（持续） | 季度 | 阈值季度回顾、用户日志滚动补充、评测集 v2 冻结 | 主指标趋势稳定，回归集 ≥20 条 |

### 4.4 角色分工

| 角色 | 职责 |
|------|------|
| 算法/工程 | 跑评测、修回归、维护指标实现与门禁 |
| 标注员（1–2 人） | 出题、标注证据、审核 CSV 回填、每周人工校准打分 |
| 投研/合规专家 | 拒答与计算类题目终审、红线阈值业务确认、争议仲裁 |
| Tech Lead | 门禁结论裁定、阈值修订、评测集版本发布 |
