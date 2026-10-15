# 公共元数据修正操作

本迁移仅针对 `finance_kb` 快照中的原 PDF，以及配置 `TENANT_ID` 对应的行。
修改字段限于 `date`（报告发布日期）、`broker`、`meta_source=public` 和 `needs_review`。
不重嵌入、不改 schema、正文、向量、ID、股票名、行业、类型、标题、版本、删除标记、父块和整表。

## 准备与逐源审查

先完成完整快照并运行 `scripts.kb_snapshot.verify_snapshot`。保留原快照及历史日志。
在维护窗口停止其他进程的上传、删除、PATCH 和标量写入；KB 写锁仅协调同一进程，不能排斥其他进程或直接数据库写入。
所有输出使用新的显式路径。提案目录和报告文件已存在时拒绝覆盖。

```powershell
.venv/Scripts/python.exe scripts/repair_report_metadata.py preflight --snapshot scripts/results/research_report/20261003-metadata-filter-v1/before --output <新的提案目录>
```

`preflight` 只读原 PDF、Milvus、现有 PG 表和指纹文件，输出 `proposal.json`。
逐源检查 `audit.fields` 的证据页码/位置、`field_status`、`patch`、`before` 和 `manifest_before`。
提案绑定全部 before 标量、每行 ID/content/dense 摘要、完整指纹、原 PDF SHA、代码文件摘要、规则版本和解析日期。
全部 before 行必须与当前库一致；新增/缺失 source、PDF 变更或租户不符会拒绝。

`unknown` 的日期或机构设为空串并保留 review；不得从文件 mtime、入库时间或 `H3_AP` 平台标识猜测。
`conflict_review` 不选取某一个冲突日期；任何人工审核字段或整文 `meta_source=manual` 都进入 `manual_review`。
仅可靠原文明确矛盾时列出人工值候选，自动 apply 始终不覆盖人工值。
南华/开源自身署名的暂缓事项必须交最终审查，不手工把旧机构升级为已验证。

## 确认后执行

本阶段实现者只运行 preflight/verify。控制者完成完整测试、独立代码审查和逐源提案审查后，使用重新生成且代码摘要一致的最终提案。

```powershell
.venv/Scripts/python.exe scripts/repair_report_metadata.py apply --snapshot <before目录> --proposal <最终目录>/proposal.json --confirm-collection finance_kb --report <新的apply报告.json>
.venv/Scripts/python.exe scripts/repair_report_metadata.py verify --snapshot <before目录> --report <新的verify报告.json>
```

apply 在每源集合锁内再次读取全部来源、源内每行和可信指纹；拒绝漂移源，保留失败记录并继续其他源。
PG `documents` 仅更新既有 tenant/source 记录的日期，不建立任何 manifest 表或新文档；SQL 行锁内再次检查提案 before。
缺表报告 `skipped_missing_table`，缺对应文档报告 `skipped_missing_document`，其他集合显示 `not_applicable`；连接失败是失败，不能伪装为缺表。
正常入库与 PATCH 的公共/类型审计都持久化在指纹存储；SHA、SimHash 等原始指纹不变。API 默认 PATCH 仍标为 `manual`。

## 失败与验收

这是跨存储逐源补偿，不是全局原子事务。SQL 事务覆盖日期更新与外部写入阶段；异常后恢复该源完整 Milvus before 行、指纹记录和精确 PG 日期。
报告保存每源写前恢复数据、异常、补偿结果及补偿失败原因。SQL 补偿发现新的第三方日期时拒绝覆盖。
`failed`、`pending_review` 和补偿失败均令 apply 非零退出。不要直接重试旧提案：先检查失败源实际状态，保留原报告，再重新 preflight。
进程终止可能留下 `running` 源；利用报告中的 `before_rows`/`before_fingerprint`/`manifest_before` 人工核对恢复，脚本不会声称崩溃后自动恢复。

verify 只读检查 PDF SHA、全部非公共字段/正文/向量、指纹和类型审计不变，核对已迁移字段与原 PDF 审计及 PG 展示日期，并核对 finance_kb 范围的父块/整表原值。
`unchanged_before` 表示仍是原快照，`public_updated` 表示公共字段已更新；验收成功不等于所有来源都已迁移，须同时核对 apply 报告。
验收失败返回非零退出码。其他集合（包括 `kb_probe`）不属于迁移范围。
