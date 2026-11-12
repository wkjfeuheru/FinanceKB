# MCP 整改与验收记录

日期：2026-11-10。范围：FinanceKB 本地 HTTPS 注册的五套 MCP、14 个工具，以及 Research-Copilot 的 FinanceKB schema 指纹。

**结论：工具契约、严格校验、动态发现、统一错误分类及有界重试已部署并通过相应验收；Infoway 的 Retry-After 响应头处理仍有一项未闭环。** 本次真实调用覆盖 14 个工具，故障重试使用独立注入测试验证，两者不混作数据源长期稳定性证明。初次审计保留在 [原检查报告](mcp-comprehensive-audit.md)。

## 整改项与验收结论

| 原报告项目 | 本次实现 | 验收结论 |
| --- | --- | --- |
| P1 知识域严格参数校验 | 两工具接入 StrictToolsMCP；先按公开 JSON Schema 校验，再执行业务校验；顶层及 filters 禁止未知字段 | 通过。字符串/布尔页码、缺必填项、未知参数、日期及分页错误均拒绝 |
| P1 自动重试 | 所有上游 HTTP 请求接入统一同步/异步策略；总期限、取消传播、错误及降级日志 | 主要通过。次数耗尽、恢复、永久错误、取消、总期限均有故障注入验证；SDK Retry-After 例外见后文 |
| P2 新闻说明 | keyword 改为任意合法公司/主题词；说明源首批最多 10 条及日期过滤；默认 limit 改为 10 | 通过。保留 limit 1–100 的输入兼容性，但不承诺超过 10 条 |
| P2 知识输出契约 | 返回 SearchResponse，自动发布 outputSchema、readOnlyHint；文本内容包含完整 JSON 结果 | 通过。研报非空、政策空结果均通过公开输出 schema 校验 |
| P2 日期/财报/available 说明 | 说明单边补日期及示例、years 的不含终点窗口、2026 NBS 日程限制、缺报价整体不可用、空结果语义 | 通过。最终 tools/list 中包含这些说明 |
| P2 机器可读错误 | 统一五类错误及 retryable，保留 MCP isError 与协议错误区分 | 通过。65 项真实非法调用均返回 invalid_argument；五域未知工具均返回 -32602 |

知识检索的查询/分页/过滤字段说明、报告类型 enum 和日期 pattern 已公开。真实日期、前后顺序和 page × page_size ≤ 200 仍由业务校验实现，schema 描述中明确此约束。

核心实现：[严格入口](<F:/python project/FinanceKB/src/mcp/common/server.py:36>)、[统一请求策略](<F:/python project/FinanceKB/src/queries/source_requests.py:126>)、[知识工具](<F:/python project/FinanceKB/src/mcp/knowledge/server.py:88>)、[知识 HTTP](<F:/python project/FinanceKB/src/mcp/knowledge/client.py:53>)、[过滤字段](<F:/python project/FinanceKB/src/api/schemas/search.py:40>)。

## 逐工具真实 HTTPS 验收

全部 14 个工具均可经 tools/list 发现，带输入/输出 schema、参数说明和只读标记。输入 schema 校验通过，成功结果同时通过 outputSchema 校验；文本 JSON 与 structuredContent 一致。

| MCP server | tool | 本次实际返回 | 结论 |
| --- | --- | --- | --- |
| market | resolve_symbol | 000001 返回上证指数及平安银行两个候选 | 通过 |
| market | get_ohlcv | 五粮液 11 月 9–10 日日线 2 条；另验 11 月 10 日分钟线 241 条、沪深300日线 2 条，均来自东财 | 通过 |
| market | get_realtime_quote | 茅台 1 条 Infoway；另验五粮液、平安银行、沪深300，返回 3 条新浪快照 | 通过；深市 Infoway 514 已降级，盘中时效另验 |
| market（2026-11-11 直连改造后复测） | resolve_symbol / get_realtime_quote | 东财搜索解析 000001、平安银行、沪深300；新浪快照返回平安银行、茅台、沪深300、上证指数 4 条 | 通过 |
| market（2026-11-11 直连改造后复测） | get_ohlcv | 东财 K 线直连复测期间出口 IP 被东财临时限流，解封后补验日线与分钟线 | 待复测 |
| knowledge（2026-11-11 GPU 复测） | search_research | RTX 4060 + CUDA 12.6 镜像生效（torch 2.14.1+cu126，cuda available=True，RERANKER_DEVICE=cuda）；两次真实检索 12.2s（含预热）/3.7s，CPU 时代为 25–70s | 通过；延迟下降约一个数量级 |
| fundamentals | get_financial_statements | 茅台最近完整年度三张报表，东财 | 通过 |
| fundamentals | get_valuation_metrics | 五粮液 11 月 10 日 PE、PB、PS 三项，东财 | 通过 |
| fundamentals | get_company_profile | 五粮液公司概况及高管/股东结构，1 个聚合对象，Infoway + 东财 | 通过 |
| fundamentals | get_dividend_history | 茅台 2 条分红事件，东财 | 通过 |
| macro | get_macro_indicator | 2026 年 GDP 三期观察值，东财 | 通过 |
| macro | get_economic_calendar | 11 月国家统计局计划日程 11 条 | 通过 |
| news | search_news | 半导体主题新闻 10 条，东财 | 通过 |
| news | get_filings | 五粮液公告 2 条，东财 | 通过 |
| news | get_earnings_calendar | 10 月 28 日命中日程 830 条，东财 | 通过 |
| knowledge | search_research | 半导体行业景气度检索 1 条研报片段 | 通过 |
| knowledge | search_policy | 合法空 items，与当前空政策集合一致 | 通过空结果契约；政策内容召回待有数据后验证 |

五个端点未携带认证时均返回 HTTP 401。按 server 分组的非法参数真实验收为 market 17、fundamentals 17、macro 7、news 12、knowledge 12，共 65 项，无接受非法请求的情况。另有知识两工具各自拒绝字符串页码、布尔页码及未知字段的单元回归。

完整证据：[发现与错误调用](mcp-remediation-acceptance-evidence.json)、[成功调用与返回数据](mcp-remediation-live-evidence.json)、[调用方验收](mcp-remediation-consumer-evidence.json)。这三个文件记录实际返回，不含认证信息。

## 重试策略与错误契约

“重试次数”指首次请求之后的额外次数，针对单个上游请求。分页和多报表查询会发出多次不同请求，但共享同一个工具总期限。

| 项目 | 运行行为 |
| --- | --- |
| 次数 | 默认最多额外 2 次，即最多 3 次尝试；MCP_SOURCE_MAX_RETRIES 可配置为 0–5 的整数 |
| 可重试 | 瞬时网络/远端协议错误、请求超时、HTTP 429、502、503、504；Infoway 对应暂时性错误 |
| 不重试 | HTTP 400/422、401/403、Infoway 514、SDK 日配额业务码 502、响应格式/业务校验错误 |
| 等待 | 默认第 1 次 1 秒、第 2 次 2 秒，各加 0–0.25 秒抖动；增加次数时继续指数退避，受总期限约束 |
| Retry-After | 直接 HTTP 路径解析秒数或 HTTP-date；与本地退避取更大值，剩余时间不足则结束，不提前再次请求；Infoway SDK 例外见下节 |
| 请求超时 | 行情/基本面及 Infoway 默认 8 秒，宏观/资讯 10 秒；知识 API 默认 120 秒，均收紧到剩余工具预算 |
| 总期限 | MCP_CALL_TIMEOUT_SECONDS 默认 120 秒，范围 1–600；含等待、分页和降级时间，超时返回 upstream_timeout |
| 取消 | 异步请求传播取消；同步工作线程的等待被唤醒，不再启动重试/后续请求。已经进行中的同步 HTTP 受请求超时约束，不保证立即中断 socket |
| 次数耗尽 | 抛出脱敏 SourceError；已有切源路径继续降级，否则返回 isError=true；预算耗尽后不再发起降级请求 |
| 日志 | source、attempt_count、code、retryable；切源记录 target、fallback=true，不记录认证头和上游响应体 |

Infoway SDK 的 max_retries=1 表示仅首次尝试；外层统一策略拥有重试权，避免双层重试。514 权限错误实测日志显示 attempt_count=1，随后 source_fallback 到新浪/东财。SDK 业务码 502 是日配额耗尽，与 HTTP 502 不同，不作瞬时故障重试。[Infoway 适配器](<F:/python project/FinanceKB/src/queries/infoway_source.py:30>)

工具执行失败返回 MCP CallToolResult 的 isError=true；content 中的文本为可解析 JSON，例如：

```json
{"error":{"code":"rate_limited","message":"上游请求失败","source":"eastmoney","retryable":true,"attempt_count":3}}
```

错误 code 为 invalid_argument、upstream_timeout、rate_limited、permission_denied、upstream_unavailable。source/attempt_count 在来源错误中提供；入口 schema 校验的 attempt_count=0。业务校验和入口总期限错误没有对应单一来源，不保证 source/attempt_count 字段。未知工具是 JSON-RPC -32602，认证失败是 HTTP 401。错误结果不按成功的 outputSchema 校验。retryable=true 表示错误性质允许以后再试，不代表服务端还会继续尝试。

成功返回中，结构化工具 available=false 表示此次未取得所需数据并附 reason；available=true 且 items=[] 表示查询完成但无匹配。知识工具沿用 SearchResponse，无 available 字段，以 items 和分页字段表达检索结果。

## 尚未闭环及使用边界

1. **Infoway 的 Retry-After 响应头：接受为已知边界，维持现状（2026-11-10 决议）。** 当前锁定 infoway-sdk 0.4.1，SDK 抛出的异常不携带响应头，也没有公开的 HTTP client 注入参数。429 按 1/2 秒加抖动的有界策略处理，不声称遵从源端指定等待时间。重开闭环的条件：Infoway 发布暴露响应头/transport 的 SDK 版本，或决定对涉及的 5 个端点改用官方 REST 适配器（复用现有 retry_after 解析）；届时补 429 注入用例。在此之前不改 SDK 私有传输。另注：Infoway 真实 429 是否携带 Retry-After 需盘中触发限流才能确认，当前实现在上游不回传时自动回退本地退避，功能层面不受影响。
2. 本次是收盘后验证。新浪快照的盘中时效与上游长期可用率未据此验收通过。
3. **行情链路 2026-11-11 起改为东财 K 线 + 新浪快照直连，不再经 Infoway**（决议记录见实施方案）：深市 514 授权问题对行情工具不再存在；Infoway 仅保留在基本面（公司概况、交易日历），其 Free 套餐 600 次/日配额与 2026-11-16 到期仍待用户申请升级或授权。已知边界：东财分钟级历史约最近 7 周（1 分钟约 5 个交易日）；东财对突发高频访问按出口 IP 临时限流。
4. 新闻源仍只返回首批最多 10 条；NBS 默认仅配置 2026 年；这些覆盖边界已在工具说明公开。
5. Research-Copilot 当前工具时限为 90 秒，可能先于服务器的 120 秒取消请求。服务端有取消控制；如业务要使用完整服务端预算，需要调用方另行调整时限。此次保留现有调用方上限。

## 测试、部署和兼容性

- FinanceKB：tests/unit/mcp、tests/integration/mcp、tests/unit/api/test_search_contract.py 合计 **107 passed**；包含故障注入，不调用真实付费模型。
- Research-Copilot：tests/unit/test_financekb_sources.py **3 passed**。
- 静态检查：本次 src/mcp、src/queries、SearchFilters 及整改测试的 Ruff 检查通过。测试有一项 Starlette/httpx 弃用提示，无失败。
- 重试注入覆盖：503 后恢复、三次耗尽、429 Retry-After 秒数及日期解析、知识 HTTP 异步恢复/耗尽、401/403/422 不重试、Infoway 514/日配额 502 不重试、SDK 暂时限流/503 重试、取消和整体期限。
- 五个本地 MCP 容器已更新且健康。四个结构化 MCP 没有 DATABASE_URL / POSTGRES_* 环境变量。
- RAG app 和 PostgreSQL 容器 ID 未变化（b7742c21025f、fb489fc8a5f8）；只读核验父块 733、整表块 138，研报检索成功。
- 已审阅真实 schema 差异并更新 Research-Copilot 的 12 项 APPROVED_SCHEMAS。结构化结果字段未变；主要是参数/结果描述及新闻默认 limit，知识域新增公开输出契约。消费者实测发现全部 12 项，并完成深市报价、估值、新闻三次真实调用。[调用方指纹](<F:/python project/Research-Copilot/src/research_copilot/sources/financekb.py:13>)
- 已绑定主机的配置更新仍使用 Research-Copilot 原有绑定流程。此次没有变更绑定记录或运行报告任务。

本地配置新增 MCP_SOURCE_MAX_RETRIES=2 和 MCP_CALL_TIMEOUT_SECONDS=120，示例见 .env.example，Compose 已注入五套 MCP。测试用源码、既有数据源适配器和部署配置变更均保留在工作区，未创建提交。

