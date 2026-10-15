# 第一批修复：评测有效性与确定性内容损失

用户已批准此计划，允许原地重建 finance_kb。工作在既有脏工作区进行，保留所有用户修改；不提交或推送。

## 全局约束

- 中文交流；BGE-M3 CLS 1024、256/25 切块、融合权重及候选配额不变。
- 不优化路由、多跳、表格行索引、生成提示词或引用校验。
- 原始 PDF 18 份是唯一回灌来源；先完整备份向量/标量/父块/整表/文件指纹再重建。
- 核心20和派生30修订为独立自足问题，按 silver/auto_pass 记录代理审核，不伪造人工 Gold。
- exact 指标向后兼容；增加事实证据组/事实MRR/来源绑定覆盖；历史结果保留，跨KB只描述性比较。

## Task 1: 内容保留

修复 cleaner、chunker HTML转换、HybridRetriever 父块扩展；先写失败回归再实现，记录红绿结果。
保留业务风险提示、独立数字和重复短行，段落SimHash只诊断；删除内容需规则审计。
标准库HTMLParser布局保留空单元和rowspan/colspan，左上文字其余空占位，合并说明；异常保留HTML且记录原因；Markdown转义竖线换行。二维payload接口不变。
父块读完再合并，50..3000且包含全部子块才覆盖；失败/长短/不存在/不匹配保留各子块，只按chunkID去重；附covered_child_ids及诊断。

## Task 2: 评测schema与指标

schema2增加facts与稳定evidence_id/document_sha256/quote_start/quote_end/location；兼容旧JSONL；替代证据OR组合、组内AND。
增加evidence_group_recall_at_5/fact_mrr/structured_required_fact_coverage_at_5/all_required_documents_at_5和各阶段存活；来源绑定，数字单位时期不误匹配；深度不足固定@20/@50为null另报实际深度。
新增PRODUCTION策略复用生产retrieve_documents与整表展开、上下文预算，报告检索和生成上下文覆盖；负例独立现有拒答验证。
完善版本指纹：标注/指标/PDF集合/处理规则/chunk内容摘要/脏树实际代码差异。不得破坏现有指纹比较门禁。

## Task 3: 标注修订

修核心20/派生30：完整项目名和金额、明确报告范围、完整必需事实、多份有效替代证据、完整引文及原文位置；同事实需要跨块时全部片段。
新增4个真实回归题（2表格2风险），单独集。脚本不根据检索排名挑证据。
保存旧文件，发布版本、逐题变更、审核材料；PDF SHA证据跨KB重映射；未映射不静默删样本。

## Task 4: 备份、基线、回灌与验收

先保存finance_kb所有行（含向量）、schema、fingerprints及SQL两表，哈希/行数校验，提供恢复脚本。
修订标注旧库基线，完整pytest通过后规则版本提升，rebuild_collection并ingest_reports全量18PDF。
审计18文档canonical、1024归一化向量、派生引用；重映射证据，新旧两套和回归集评测及PRODUCTION上下文/负例；报告无法映射及各阶段损失，跨指纹不推断显著提升。
