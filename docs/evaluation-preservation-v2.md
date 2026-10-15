# 研报评测与内容保留：第一批

本批只修订标注和确定性内容损失，不调整模型、候选路由、融合权重、切块参数或生成提示词。BGE-M3 / CLS / 1024、256/25 和 dense/sparse 0.8/0.2 保持不变。

## 标注与指标

JSONL schema 2 的 `facts` 为完整事实；`allowed_sources` 限制来源，`aliases` 外层任选、内层全部满足。`acceptable_evidence_sets` 同样外层任选、内层全部需要。替代证据不增加事实分母。稳定 `evidence_id` 与 PDF SHA、页内原文偏移不因知识库重建改变；`chunk_id` 和 `location.anchor_id` 仅为当前知识库映射。

脚本及代理核验的样本为 `silver/auto_pass`，不是人工 Gold。保留兼容文件名 `research_report_gold20.jsonl` 不代表人工批准。interview30 中的十条配对题由核心集派生，不可与核心20合并为50个独立样本。四条表格/风险回归题单独统计。

`evidence_group_recall_at_5` 衡量完整事实证据组；`fact_mrr` 为逐事实首次完整组覆盖排名倒数的均值；`structured_required_fact_coverage_at_5` 在允许来源的实际正文内匹配已标注完整等价表述。保留旧 exact chunk Recall/MRR、quote coverage 与旧 `required_fact_coverage_at_5`，不得把不同口径直接混为同一指标。

固定 candidate Recall@20/@50 仅在实际候选数达到对应深度时可用；不足时为 null，并报告实际深度覆盖率。PRODUCTION 调用真实检索入口及整表展开、上下文预算，检索覆盖率和实际模型上下文覆盖率分开报告。负例不进入正例 Recall 分母；空回答且无明确拒答标记不算拒答成功。

## 内容保留

- 清洗保留业务风险、独立数值和重复短行；段落 SimHash 只记录疑似重复，不自动删正文。文档间重复检测不变。明确页码、版权等删除带实际文本和规则审计。
- HTML 表格还原中间/尾部空格、跨度与位置。合并文字只留左上，其他格为空；异常保留原始 HTML，不生成错列表格。
- 文本父块批量读取后，只有长度50～3000且包含所有待合并子块正文才合并；失败保留不同子块。成功合并记录覆盖子块身份。表格、图片走原有独立展开。

## 可复现迁移

先备份完整 dense/scalar、schema、指纹、PDF 清单和 PostgreSQL 父块/整表；生成式 BM25 sparse 无法直接读取，恢复时用保存的 content 重算。校验备份完整，不把仅有文件存在当作备份成功。

```powershell
python scripts/kb_snapshot.py backup --snapshot <旧库备份目录>
python scripts/revise_research_evaluation.py --snapshot <旧库备份目录> --out-dir <旧库修订标注目录>
python scripts/evaluate.py retrieval run --dataset <旧库修订核心JSONL> --strategy L0 L1 L2 L3 L5 PRODUCTION --dataset-version research-report-facts-v2 --run-id old-kb-final --outdir <基线目录>
python scripts/rebuild_research_kb.py prepare --prepared <新鲜解析目录>
python scripts/rebuild_research_kb.py verify --prepared <新鲜解析目录>
```

完整测试通过、六层基线成功并匹配标注/旧库内容指纹后，提升清洗与切块规则版本，再明确指定 `--confirm-collection finance_kb` 执行 ingest。该操作原地删除开发库及派生数据，必须先验证备份，禁止直接照抄路径执行。

确认 embedding 与 reranker 缓存的真实推理烟测通过后，可在迁移/评测进程中设置 `$env:MODEL_OFFLINE='true'`、`$env:HF_HUB_OFFLINE='1'`、`$env:TRANSFORMERS_OFFLINE='1'`，避免缓存加载期间的远程查询故障；这不改变模型或云端 LLM 的调用策略。

```powershell
python scripts/rebuild_research_kb.py ingest --prepared <新鲜解析目录> --snapshot <旧库备份目录> --baseline <基线目录> --dataset <旧库修订核心JSONL> --confirm-collection finance_kb --report <回灌记录JSON>
python scripts/kb_snapshot.py backup --snapshot <新库备份目录>
python scripts/revise_research_evaluation.py --snapshot <新库备份目录> --remap <旧库修订JSONL> --out-dir <新库重映射目录>
```

未映射的原文证据保留，并输出为潜在数据损失；不得删样本提高分数。新旧库分别保存标注、规则/代码哈希和评测快照。不同知识库指纹只作逐题描述性比较，不绕过门禁发布显著性结论，也不将标注修订引起的分数上涨当作检索改进。
