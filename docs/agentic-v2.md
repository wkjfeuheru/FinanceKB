# Agentic RAG v2 实施与验收

本版只处理已确认的八项高优先级问题，保留原检索算法及存储结构。

## 修改对应关系

| 问题 | 主要实现 | 回归重点 |
|---|---|---|
| H1 重试语义 | `services/retrieval_service.py`、`orchestration/graph.py` | 初检不计补检；最后一批执行；重复及基础设施错误出口 |
| H2 证据契约 | `agent/tools/retrieval.py`、`schemas/chat.py`、`chat_service.build_context` | 集合与子块身份、父块覆盖关系、可空版本身份、输入不变 |
| H3 充分性门禁 | `orchestration/nodes/grading.py` | 实际裁剪上下文逐事项覆盖；非法JSON、未知ID不能放行 |
| H4 共享路径 | `services/retrieval_service.py`、`services/answer_service.py`及评测适配 | 线上与评测过滤、上下文及输出策略一致 |
| H5 工具协议 | `orchestration/nodes/decision.py`及编译图 | 参数限制、调用ID关联、强制检索、请求状态隔离 |
| H6 按需规划 | `agent/planning.py`及共享执行器 | 简单不拆、超限澄清、组合不遗漏、定向补检及累计证据 |
| H7 评测修正 | `eval/manifest.py`、`statistics.py`、`runner.py`及说明文档 | 去重分组、审核等级、按组统计、冻结指纹及质量护栏 |
| H8 引用出口 | `citation_validator.py`、`answer_service.py`、API及前端 | 逐声明校验、部分回答限定事项、验证前无事实输出、未知状态 |

上述Python路径均相对于`finance_rag/src`。工作保留在`codex/agentic-v2`，未自动提交；既有工作区修改保持原状。

## 已实现的行为

- 简单问题不分解；历史指代只替换为用户历史明确的实体。复杂问题最多3个必要事项，超限澄清。
- 金融事实必须检索；Agent使用带调用ID的检索工具协议，模型不能改写请求过滤范围。
- 初检后最多2批补检；只补缺失事项，重复查询停止。证据池跨轮保留，最新补检证据参与实际上下文覆盖检查。
- 覆盖检查针对实际预算后的上下文，检查实体、指标、期间和单位；相关分数不能代替充分性判断。
- 标准与Agent、HTTP与SSE共用答案出口。生成后逐声明校验，失败最多重生成一次；无法校验不发布事实。
- 有据部分回答列出缺口；来源响应保留子块、父块覆盖ID和已有元数据，全文留在服务内部。
- 线上和A/B共用检索及答案执行器；算法层实验与生产流程实验分别标记。
- 独立样本按精确重复及显式配对归组。人工Gold和自动Silver分别统计；协议、裁判、数据或知识库身份不完整时不能宣称提升。

## 配置与接口

```ini
ENABLE_LANGGRAPH=false
ENABLE_CITATION_VALIDATION=true
LANGGRAPH_MAX_REFLECT_ROUNDS=2
AGENT_MAX_SUBPROBLEMS=3
```

`LANGGRAPH_MAX_REFLECT_ROUNDS`保留旧名称，实际表示初检后的补检批次数，范围0～2。`ENABLE_LANGGRAPH=false`仍走标准链路，标准链路也执行验证门禁。

当前工作区`.env`已将`ENABLE_LANGGRAPH`从`true`调整为`false`，保留`ENABLE_CITATION_VALIDATION=true`。其他部署需自行同步配置并重启进程。

若已有环境显式配置`ENABLE_CITATION_VALIDATION=false`，金融事实会返回无法验证的说明；该开关不能恢复未核验事实输出。

HTTP响应和SSE完成事件保留原字段，新增`answer_status`与`missing_items`。状态为`complete/partial/refused/unverified/clarification`；`citation_validation.status`为`verified/failed/unknown/not_required`，unknown的score为空。

引用校验使用实际内容窗口，而非展示预览。跨来源声明可使用`supporting_quotes`，逐来源验证原文。原文缺失、实体数值交换、期间或单位错误不会通过确定性守卫；自然语言支持判断仍依赖模型，不构成数学证明。

## 可复现检查

```powershell
.\.venv\Scripts\python.exe scripts/build_eval_manifest.py
.\.venv\Scripts\python.exe scripts/check_agentic_model.py
.\.venv\Scripts\python.exe scripts/check_agentic_model.py --live
G:\node.js\node.exe --test frontend/tests/*.test.js
```

模型检查默认不联网，`--live`调用已配置端点，使用合成资料，不执行数据库检索，会产生少量API费用。2026-10-13实际运行工具调用与声明JSON协议检查通过，耗时约6.6秒；这不是准确率或效果评测。

2026-10-13验证结果：完整非慢速Python回归988项通过、24项慢速测试未执行；其中新增及受影响的定向回归358项通过。前端8项测试及生产构建通过，本次涉及的Python模块和脚本Ruff检查通过。

全量回归期间发现旧向量写入测试遗漏表格存储替身，尝试连接真实PostgreSQL而阻塞；已补齐父块及表格存储边界的离线替身，入库模块22项测试和后续完整回归均通过。两个旧检索/评测测试已迁移至共享执行器契约。

评测清单位于`scripts/results/agentic-v2/manifest.json`：146条原始评测记录、94个潜在独立组、12个已审Gold独立组、70条待复核记录；6个锚点和4个回归题另列。潜在组不能视为已审核独立样本。

历史14.12%相对提升来自2026-08-25的5题实验，同次faithfulness下降0.1000。原始历史产物不改写，不能用其宣称v2或100题效果。

## 仍需验收的项目

- 当前语料下的人工逐声明核验、错误引用及跨文档准确率。
- 固定知识库和裁判后，改写、路由、反思和完整Agent的真实A/B与成本测量；本次未运行批量付费评测。
- 全部线上证据的版本SHA绑定。当前仅保留已知字段，无法确定时为空，不从文件名或当前文件冒充历史版本身份。

新图启用前必须完成真实资料质量验收。回退新图使用`ENABLE_LANGGRAPH=false`，保留标准答案门禁。
