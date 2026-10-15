# 研报可信公共元数据与过滤修复 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复发布日期/发布机构的确定性误抽取和不可信自动硬过滤，安全更新现有18份报告标量，并产生可追溯的真实生产评测。

**Architecture:** 公共元数据解析、查询过滤计划分别进入聚焦模块，原服务只做集成；迁移使用现有集合级锁及快照校验，禁止重算向量。先验证元数据修复，再集成过滤策略；记录不同干预阶段，不修改评测标注。

**Tech Stack:** Python3.13、Pydantic/FastAPI、PyMuPDF、Milvus/SQLAlchemy/PostgreSQL、pytest；全部使用现有依赖。

## Global Constraints

- 用户已确认`docs/superpowers/specs/2026-10-03-report-metadata-filter-design.md`方案B。用户已批准原地工作，不提交、不推送、不回退任何原有改动。
- 继续使用BGE-M3/CLS/1024维/CPU ONNX INT8、256/25切块、0.8/0.2融合、现有候选配额及重排序设备。
- 不改生成提示词、引用校验、表格行级索引、multi-hop策略和人工评测标注。
- 不调用rebuild_collection、不删除向量、不增加Milvus字段；现有`date`固定为报告发布日期，`ingested_at`独立。
- 迁移只能影响当前18份原PDF的公共标量及其审计/展示存储；PDF、chunk ID/content/向量、父块及整表不变。
- API显式约束和已确认的自然语言范围不可被放宽；未确认自动条件不得成为唯一硬排除依据；基础设施故障不得触发扩大范围。
- 核心20、派生interview30、独立回归4保持不变；silver不可升级为人工Gold，配对不并为独立样本。
- 所有测试先RED再GREEN，每项任务独立审查。技能的提交/新worktree/删除审查记录步骤由上述用户约定覆盖：保留工作区及审查记录，用实际工作树前后diff审查。

## 文件与执行边界

Task 1只实现公共元数据与入库集成；Task 2实现过滤策略与API/生产入口；Task 3实现安全标量迁移；Task 4负责真实迁移与生产验证。实现代理按顺序执行，不并行写共享文件。主代理可并行运行只读快照/旧版本生产基线。

### Task 1: 可信公共元数据解析与入库集成

**Files:**
- Create: `finance_rag/src/rag/ingestion/public_metadata.py`
- Modify: `finance_rag/src/services/document_service.py`
- Modify: `finance_rag/src/rag/ingestion/metadata_extractor.py`
- Modify: `finance_rag/src/services/ingestion_pipeline.py`
- Modify: `finance_rag/src/schemas/document.py`（仅日期有效性/公共审计相关）
- Test: `tests/unit/test_public_metadata.py`
- Test: `tests/unit/test_metadata_extractor.py`, `tests/unit/test_ingestion_enrichment.py`

**Interfaces:**
- Produces `resolve_public_metadata(source: str, markdown: str = '', *, pdf_path: Path | None = None, explicit: dict | None = None, as_of: date | None = None) -> PublicMetadataResult`。
- `PublicMetadataResult`有`date: str`, `broker: str`, `needs_review: bool`, `audit: dict`；`as_metadata() -> dict`提供公共字段和`public_metadata_audit`。
- `audit`包括`rules_version='public-metadata-v1'`、`source`、`pdf_sha256`及`fields`；每个`fields['date'/'broker']`含`value`、`status`（known/unknown/conflict/manual）、`evidence`列表和`reasons`列表。证据有原文、页码、位置及来源，不依赖检索排名。
- 日期基础校验可导出`validate_publication_date(value: str, *, allow_empty: bool = True) -> str`供PATCH复用；财务期间与查询范围校验在Task 2。
- 只用原始source/封面/明确发布标签抽日期。不得以临时UUID、mtime、相关研究、表格或报告期填发布日期。日期不明为`''`；自动未来日期和来源冲突需复核，不换用其他兜底猜测。
- 自身页脚/封面发布署名优先，受控机构别名归一化；引用机构不是发布机构。无可靠身份或可靠来源冲突不得任选一个。
- `public_metadata_audit`作为文档级业务审计传递，非Milvus新增字段；canonical builder保持三种类型payload约束并保留合法公共审计。用户输入不得伪造可信审核来源。

- [ ] **Step 1: 先写真实临时名不变性和期间排除回归。** 最小测试使用独立字面期望：

```python
from pathlib import Path
from types import SimpleNamespace
from finance_rag.src.services.document_service import resolve_document_date

def test_publication_date_not_controlled_by_temp_uuid():
    results = []
    for filename in ('.abcdef2053abcdef.uploading.pdf', '.abcdef2006abcdef.uploading.pdf'):
        parsed = SimpleNamespace(path=Path(filename), source='报告.pdf', metadata={},
            chunks=SimpleNamespace(markdown='发布日期：2026年9月27日。'))
        results.append(resolve_document_date(parsed))
    assert results == ['2026-09-27', '2026-09-27']
```

补充：报告期2026-06-30先于发布日期2026-09-24、日期只有年份/年月、无日期、2026-02-31、冲突候选、PDF第一页布局日期在解析3000字符之外、页脚东吴与正文引用中信、公共review状态被后续enrich/PATCH清掉。

- [ ] **Step 2: 运行上述测试验证RED。**

```powershell
.venv/Scripts/python.exe -m pytest tests/unit/test_public_metadata.py -v
```

预期失败为临时名导致2053/2006或新接口缺失；记录具体输出，不能只测试mock。

- [ ] **Step 3: 实现解析器与最小服务集成。** 对原PDF只读读取第一页文本块及几何位置，保留证据；非PDF用明确发布标签/原source日期，读取失败降级为unknown诊断，不伪造日期。`resolve_document_date`兼容入口委托新解析器而不读取临时basename。入库解析在正文清洗后用原PDF补公共证据，`enrich_chunks/apply_metadata`不让LLM broker覆盖已确认字段，不让类型审核覆盖公共审核。显式日期/PATCH用真实日历校验，人工局部改动只解决对应审核项。

核心集成形状：

```python
resolved = resolve_public_metadata(parsed.source, parsed.chunks.markdown or '',
    pdf_path=parsed.path, explicit=metadata)
merged_metadata.update(resolved.as_metadata())
parsed = replace(parsed, metadata=merged_metadata)
```

- [ ] **Step 4: 跑聚焦回归并自检。**

```powershell
.venv/Scripts/python.exe -m pytest tests/unit/test_public_metadata.py tests/unit/test_metadata_extractor.py tests/unit/test_canonical_metadata.py tests/unit/test_ingestion_enrichment.py tests/integration/test_document_metadata_patch.py -v
```

保留红绿输出、修改文件、公共接口及实际PDF抽取结果到Task报告。不要提交，不更新现库，交独立任务审查。

### Task 2: 可信过滤计划、显式API校验与有界兜底

**Files:**
- Create: `finance_rag/src/rag/retrieval/filter_policy.py`
- Modify: `finance_rag/src/services/chat_service.py`
- Modify: `finance_rag/src/schemas/chat.py`
- Modify: `finance_rag/src/api/routes/chat.py`
- Modify: `finance_rag/src/agent/prompts/chat.py`（仅过滤分析提示，禁止修改ANSWER_PROMPT）
- Modify: `finance_rag/src/orchestration/graph.py`（仅必要过滤透传，不能绕过显式条件）
- Modify: `finance_rag/src/agent/state/models.py`（编译图必须保留上述透传状态，含反思计划）
- Modify: `finance_rag/src/rag/retrieval/hybrid_retriever.py`（仅必要的完整ISO发布日期精确集合过滤构造，不改召回/排序/配额）
- Modify: `frontend/src/views/ChatView.vue`（仅必要的精确日期集合chip展示适配，保持既有布局）
- Test: `frontend/tests/chat-filter-contract.test.js`（精确日期集合与旧区间真实chip逻辑）
- Modify: `finance_rag/src/eval/retrieval.py`（仅生产过滤诊断透传）
- Test: `tests/unit/test_filter_policy.py`
- Test: `tests/unit/test_retrieval_tool.py`, `tests/unit/test_metadata_filters.py`
- Test: `tests/integration/test_chat_filter_contract.py`

**Interfaces:**
- Consumes原问句、显式filters和LLM原始候选；保持已有`normalize_metadata_filters`兼容读取，但自动候选和严格API不能共用“静默丢字段”的语义。
- Produces `build_filter_plan(query: str, *, explicit_filters: dict | None = None, inferred_filters: dict | None = None) -> FilterPlan`。
- `FilterPlan`包含`origin: str`, `hard_filters: dict`, `tentative_filters: dict`, `discarded: list[dict]`，`attempts() -> list[dict]`返回有序尝试，`as_dict() -> dict`提供诊断。
- Produces `validate_explicit_filters(filters: dict | None) -> dict | None`，无效日期/逆序范围/未知键/非法值/不合法类型行业组合抛`ValueError`；保留None和{}区别。Pydantic在SSE响应开启之前转为4xx。
- 自动日期只绑定明确报告发布日期语义，不用财务年/半年度。候选机构/实体/类别依据原問句受控别名/类型分支校验；无法明确验证的字段不升级为hard。
- 自动条件有可靠直接依据时为hard；合法但未确认的提示可用于优先分支，但必须另检索hard-only分支，即便优先分支非空也不能让提示永久排除其他必要来源。无依据/非法候选直接discard。
- 先优先分支（如有tentative），再hard-only分支，最后仅在空召回且无基础设施故障时按hard-only dense兜底；最多初始+两次额外尝试。没有tentative不重复同过滤hybrid。所有结果满足hard，使用既有merge排序和配额，不新增融合策略。
- 顶层`filters`为最终硬约束；诊断包括explicit/inferred/hard/tentative/discarded及每次实际有效filters、reason、candidate_count/error。生产前端返回与真正effective一致，未确认提示不得显示为已满足范围。
- 修复现有路由向`chat/chat_stream`传`infer_filters`但服务签名缺参数的问题；直接调用与LangGraph路径也不能丢参数/用户条件，生成行为不改。

- [ ] **Step 1: 写失败测试。**

```python
def test_financial_period_is_not_publication_filter():
    from finance_rag.src.rag.retrieval.filter_policy import build_filter_plan
    plan = build_filter_plan('杰瑞股份2026H1营业收入是多少？',
        inferred_filters={'date': {'gte': '2026-01-01', 'lte': '2026-06-30'}})
    assert 'date' not in plan.hard_filters
    assert 'date' not in plan.tentative_filters

def test_invalid_calendar_api_date_is_rejected():
    import pytest
    from finance_rag.src.rag.retrieval.filter_policy import validate_explicit_filters
    with pytest.raises(ValueError):
        validate_explicit_filters({'date': {'gte': '2026-02-31'}})
```

真实pipeline边界测试补：explicit{}不调用推断，infer=false禁用，明确某日/某机构限制不会删，优先分支非空仍不靠未确认提示排除硬范围候选，两公司/行业跨类型分支保留，基础设施异常不扩大范围，API/SSE在外部调用前4xx，逐次filters诊断、次数上限、最终约束一致。

- [ ] **Step 2: 验证RED并实现最小接口。**

```powershell
.venv/Scripts/python.exe -m pytest tests/unit/test_filter_policy.py tests/integration/test_chat_filter_contract.py -v
```

过滤模块只依赖轻量类型/词表/日期逻辑，不反向依赖chat_service形成循环。服务获取LLM候选后生成计划，用每次attempt显式传filters，不闭包误复用最初filters；保留租户/版本/删除状态默认表达式。

- [ ] **Step 3: 真实调用路径验证GREEN。**

```powershell
.venv/Scripts/python.exe -m pytest tests/unit/test_filter_policy.py tests/unit/test_metadata_filters.py tests/unit/test_retrieval_tool.py tests/unit/test_retrieval_two_stage.py tests/unit/test_retrieval_diagnostics_eval.py tests/integration/test_chat_filter_contract.py -v
```

报告说明实际attempt结构、接口与所有覆写位置；不写现库，交独立审查。

### Task 3: 安全公共标量修正与迁移入口

**Files:**
- Create: `scripts/repair_report_metadata.py`
- Modify: `finance_rag/src/infrastructure/vector_store/milvus_kb.py`
- Modify: `finance_rag/src/infrastructure/vector_store/fingerprint_store.py`（公共/类型审核持久化及精确补偿，保留内容指纹）
- Modify: `finance_rag/src/infrastructure/relational_db/document_manifest.py`
- Modify: `finance_rag/src/services/document_service.py`（标量更新协调及审计来源）
- Modify: `finance_rag/src/rag/ingestion/metadata_extractor.py`（审计合并边界必要处）
- Test: `tests/unit/test_public_metadata_migration.py`
- Test: `tests/integration/test_document_metadata_patch.py`
- Docs: `docs/report-metadata-filter-repair.md`

**Interfaces:**
- ConsumesTask1公共解析器、既有`scripts.kb_snapshot.verify_snapshot`和实际source SHA、Task2无关。
- Extend `KnowledgeBase.update_document_metadata(source: str, metadata: dict, *, update_source: str = 'manual', metadata_audit: dict | None = None) -> dict`；manual为既有API默认，自动迁移为`public_metadata`（meta_source可采用16字节以内的`public`），不得冒充manual/Gold。
- CLI `python scripts/repair_report_metadata.py preflight --snapshot DIR --output DIR`只读解析18原PDF并保存不可覆盖提案；`apply --snapshot DIR --proposal FILE --confirm-collection finance_kb --report FILE`有门禁和逐源结果；`verify --snapshot DIR --report FILE`只读验收。
- 提案绑定完整before scalars及ID/content/dense摘要、PDF SHA、代码/规则版本、审计来源；apply发现source集合、SHA、before标量、身份摘要漂移即拒绝该源，不用旧提案覆盖并发用户修改。
- 数据审核与每源可变状态均锁内读取再检查。人工修正值只有确切原文矛盾才列候选，不自动批量覆盖标记manual；冲突提案进入待审而不是选择某日期。
- 更新前校验全部源内行；只改date/broker/公共review/meta_source，不碰股票名/行业/类型/标题/content/hash/vector/租户/版本/删除字段；不触发embedding。
- 同步现有PG documents日期（若该源有对应tenant/manifest记录才更新，不臆造新documents）、指纹审计（PDF SHA/内容指纹不变）及API展示；PG父块/表完全不动。若多个存储同步失败，执行目标明确的逐源补偿并报告failed；不能误报全局原子事务。
- 恢复/失败证据记录完整；apply和verify有非零失败退出码。目录必须在明确输出路径，不覆盖历史备份。

- [ ] **Step 1: 失败回归覆盖迁移前漂移与向量不变。** 示例断言使用固定手工向量和标量：

```python
def test_public_update_does_not_claim_manual_or_reembed(prepared_kb):
    kb, rows = prepared_kb
    before = list(rows[0]['dense_vector'])
    result = kb.update_document_metadata('报告.pdf', {'date': '2026-09-27', 'broker': '东吴证券'},
        update_source='public_metadata', metadata_audit={'rules_version': 'public-metadata-v1'})
    assert rows[0]['dense_vector'] == before
    assert rows[0]['meta_source'] != 'manual'
    assert result['updated_count'] == 1
```

`prepared_kb`为测试工具提供的外部存储假体，运行真实更新方法、捕获实际写行，不mock被测方法。补proposal before漂移/PDF变更/未知源/部分失败、人工审核保护、未知日期清空但review保留、PG显示一致、补偿失败报告、source内全部行校验。

- [ ] **Step 2: 跑RED再实现入口及存储协调。**

```powershell
.venv/Scripts/python.exe -m pytest tests/unit/test_public_metadata_migration.py tests/integration/test_document_metadata_patch.py -v
```

脚本仅preflight/verify读库；apply须显式confirm及完整快照先校验。文档写出操作顺序、失败补偿、日期未知策略及如何读诊断。schema不新增，内容和embedding不重跑。

- [ ] **Step 3: 聚焦GREEN与独立审查。**

```powershell
.venv/Scripts/python.exe -m pytest tests/unit/test_public_metadata_migration.py tests/unit/test_kb_snapshot.py tests/unit/test_kb_eval_fingerprint.py tests/integration/test_document_metadata_patch.py -v
```

交报告及preflight实测候选清单，不在实现代理里apply真实迁移。

### Task 4: 真实迁移、两轮生产评测与交付

**Files:**
- Artifacts: `scripts/results/research_report/20261003-metadata-filter-v1/`
- Modify: `README.md`（公共日期/查询过滤语义说明）
- Create: `scripts/replay_metadata_baseline.py`（固定旧过滤条件的只读诊断回放，不新增生产策略）
- Modify if needed: `finance_rag/src/eval/retrieval.py`（仅共享现有生产结果/上下文适配，避免复制评分和展开逻辑）
- Test: `tests/unit/test_metadata_replay.py`
- Test: 全部`tests/`

**Interfaces:** 消费三项已审查实现与原始PDF、第一批固定标注及现有evaluate/kb_snapshot CLI；不得改评测题、gold标签、事实组或chunk映射。

**诊断回放的可复现接口补充：** `python scripts/replay_metadata_baseline.py --baseline-run-dir DIR --dataset FILE --outdir DIR --run-id ID`。只读加载同一核心数据集及 before-PRODUCTION records，验证题目身份/原问句对应和原显式条件；调用真实 `retrieve_documents(query, filters=固定旧条件, infer_filters=False, k=50, rerank_top_n=5)`，复用现有生产父块/整表/预算及指标适配，策略名 `METADATA_REPLAY`。保存 baseline records SHA、每题原过滤 SHA、数据集与新库/代码指纹、实际控制条件和失败原因。不得重新推断、改题、改事实或静默修正不合法旧条件；不可回放样本显式失败并非零退出。只回放正例，负例生成行为留给真实 PRODUCTION 两轮。TDD 覆盖实际调用参数、原条件不可变、身份不符/非法条件拒绝、证据与实际上下文都评分、多分支无统一粗排时不可用。controller 在完整代码审查及迁移验收后运行真实回放；实现代理仅测试，不自行 apply 或启动正式 after 评测。

- [ ] **Step 1: 保留第一批后的完整快照和原始生产基线。**

```powershell
.venv/Scripts/python.exe scripts/kb_snapshot.py backup --snapshot scripts/results/research_report/20261003-metadata-filter-v1/before
.venv/Scripts/python.exe scripts/evaluate.py retrieval run --help
```

`evaluate --help`用于核实当前CLI真实参数后执行，不臆造开关。基线运行放在Task2代码变更前并保存原代码摘要；既有第一批after评测为历史参照，不替代新鲜基线。

- [ ] **Step 2: 保存preflight，完整pytest通过且代码审查后才apply。**

```powershell
.venv/Scripts/python.exe -m pytest tests/ -v
.venv/Scripts/python.exe scripts/repair_report_metadata.py preflight --snapshot scripts/results/research_report/20261003-metadata-filter-v1/before --output scripts/results/research_report/20261003-metadata-filter-v1/proposal
```

逐源阅读审计，明确原文日期/机构、未确认/冲突/人工值清单；白皮书缺日期允许unknown，不手填文件名日期。代码质量gate后，按脚本实际proposal路径调用apply显式确认finance_kb。保存apply日志及逐源结果，不能失败后继续宣称成功。

- [ ] **Step 3: 证明不变性，分别记录元数据和过滤干预。**

verify脚本与新快照比较每个chunk ID、content、dense摘要、PDF集合，PG597/105派生数据逐行不变；业务canonical合法，date/broker源内一致。诊断回放使用上一批固定推断条件对照“标量修正后”；记录为REPLAY，不冒充PRODUCTION。

- [ ] **Step 4: 不改标注重跑生产与原始路径。**

真实PRODUCTION对核心20、派生30和回归4至少两轮，记录模型/代码/标量KB指纹、有效过滤、退让和每阶段/上下文coverage；原始L0/L1/L2/L3/L5至少一轮验证内容/向量身份不变后的结果。自动LLM调用不固定时报告波动，按同题描述性比较，不绕过指纹显著性门禁。用完整pytest最后再次验收。

- [ ] **Step 5: 写交付报告并只保留工作树。**

报告包括修复与未修复问题、18源更新/unknown/review/失败清单、标量before/after与原文审核、向量/派生不变性证明、逐题与分阶段真实评测、两轮生产波动、候选深度有效样本数与延迟。交付不预设性能提升，不将标量/过滤修复归因embedding升级。不提交、不推送、不删除审查workspace；下一批B的排序优化需独立设计。

## 计划自检

各接口在相应任务定义；公共元数据、API约束、存储迁移和真实验收分别有具体测试入口。题集/向量/生产生成行为保持边界，真实状态更新等待完整备份及审查；实现代理不得自行apply现库。
