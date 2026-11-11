# MCP 服务综合检查报告

> **2026-11-11 综合复检结论：四项检查全部通过，发现的一项低危问题已当日修复并实测验证。** 五套服务 14 个工具均可经 `tools/list` 动态发现（名称、描述、inputSchema/outputSchema、readOnlyHint 完整）；入参具备 schema 级（required/type/pattern/enum/min/max/uniqueItems/additionalProperties=false）与业务级双重校验，失败统一返回 `isError=true` 结构化错误（`code=invalid_argument`，不发起上游调用）；上游调用统一接入有界重试（瞬时错误/429/502/503/504 可重试，默认额外 2 次、指数退避+抖动、120 秒总预算，耗尽返回脱敏 typed 错误，实测 `attempt_count=3`）；未知工具返回 JSON-RPC -32602，未鉴权返回 401。`search_policy` 过滤器已改为政策专用的日期过滤（`PolicyFilters`），不再暴露无意义的券商/报告类型字段。详见下文"2026-11-11 综合复检"节。
>
> 本文保留整改前审计结论。2026-11-10 已执行整改并部署，最新结果见 [MCP 整改与验收记录](mcp-remediation-acceptance.md)：14 个工具真实调用通过、65 项非法参数均被拒绝，Infoway SDK 的 Retry-After 响应头处理经 2026-11-10 决议接受为已知边界（本地退避兜底，详见整改与验收记录）。

## 2026-11-11 综合复检

检查方式：对运行中的五套服务走完整 MCP 协议（initialize → `tools/list` → 注入非法参数/未知工具/未鉴权调用），快照存于 `.local-data/audit-20261111/tools-list.json`；并复核 `src/mcp/*/server.py`、`src/queries/source_requests.py` 校验与重试实现。

| 维度 | 结果 | 证据 |
| --- | --- | --- |
| 1. 工具描述 | 通过（1 项低危措辞问题） | 14 个工具名称域内唯一，描述均说明用途、参数含义、边界与返回值；全部 `readOnlyHint=true`。`search_policy` 的 filters 描述沿用了研报的"券商、报告类型"，政策库无语义 |
| 2. 参数校验 | 通过 | schema 级：所有工具 `additionalProperties=false`、必填数组完整、日期 pattern、enum、min/max、symbols 1–20 uniqueItems、query 1–500 字符等；业务级：日期先后、指数仅 1d/1w/1M、page×page_size≤200、K 线 5000 上限、symbol 可解析。失败均返回 `isError=true` + `{"error":{"code":"invalid_argument",...}}`，`attempt_count=0` 表明未触碰上游 |
| 3. 重试机制 | 通过 | `source_requests` 统一策略：可重试=超时/网络/远端协议错误/429（解析 Retry-After）/502/503/504；不可重试=400/422/401/403/Infoway 514/日配额/格式校验错误；次数=1+MCP_SOURCE_MAX_RETRIES（默认 2，范围 0–5）；间隔=2^(n-1) 秒 + 0–0.25s 抖动，与 Retry-After 取大；总预算 MCP_CALL_TIMEOUT_SECONDS（默认 120s，含等待）；耗尽返回脱敏 typed 错误。实测东财故障返回 `attempt_count=3, retryable=true`；知识域异步客户端有独立有界重试（故障注入覆盖恢复/耗尽） |
| 4. 动态发现 | 通过 | 五套服务 `tools/list` 各返回 3/4/2/3/2 个工具，含名称、描述、inputSchema（含 `$defs`）、outputSchema、annotations；未知工具 -32602；未鉴权 401 |

不符合项与整改建议：

1. ~~（低）`search_policy` 复用 `SearchFilters`（broker/report_type/date）~~ **已修复（2026-11-11 当日）**：`search_policy` 改用政策专用的 `PolicyFilters`（仅 date_from/date_to，wire 格式仍映射为 `SearchFilters` 空列表 + 日期），实测 `tools/list` schema 只含日期字段、传 broker 被拒于入口校验、日期过滤真实调用通过。新增 `test_search_policy_maps_date_filters_to_search_filters` 与 `test_search_policy_rejects_research_only_filter_fields` 两个用例。
2. （提示）行情工具自 2026-11-11 起直连东财/新浪（无 Infoway 备份源）。东财对出口 IP 的突发限流会使 `get_ohlcv` 返回 `upstream_unavailable(retryable=true)`，属已文档化边界；如需更高可用性可后续申请 Infoway 深市授权作为备份。
3. （提示）`tests/unit/mcp/test_market_contract.py` 的东财改写用例本机 venv 缺 `mcp` 包无法执行，需在 CI 或带依赖容器跑一轮确认。

检查日期为 2026-11-10，范围为本地 FinanceKB 经 Caddy 注册的五套 MCP 服务，共 14 个工具。已核验运行中服务的 `initialize`、`tools/list` 和参数错误调用，并审阅服务注册、查询实现、来源适配器及 Research-Copilot 调用方。报告以本次实际返回的契约为准。

**综合结论为部分通过。** 五套服务全部可动态发现，14 个名称合法且域内唯一。四套结构化服务的严格入参校验已实现。知识检索的两个工具存在类型宽松转换、多余参数被忽略、参数说明及输出契约不足的问题。14 个工具均缺少可用的自动请求重试策略，已有降级切源可以恢复部分 Infoway 错误。

## 检查依据与实际打开的文件

| 类别 | 文件 |
| --- | --- |
| 注册和 HTTP 入口 | `deploy/Caddyfile`、`src/mcp/common/server.py`、五个域的 `src/mcp/<domain>/server.py` |
| 知识检索 | `src/mcp/knowledge/client.py`、`src/api/schemas/search.py` |
| 查询与参数模型 | `src/queries/market.py`、`fundamentals.py`、`macro.py`、`news.py`、`content.py`、`base.py`、`market_types.py`、`fundamentals_types.py`、`content_types.py` |
| 来源调用 | `src/queries/infoway_source.py`、`content_http.py`、`market_sources.py`、`fundamentals_sources.py`、`macro_sources.py`、`news_sources.py` |
| 调用方 | `F:/python project/Research-Copilot/src/research_copilot/sources/mcp_client.py` |
| 依赖实现 | 运行容器内 Infoway SDK 的 `infoway/_http.py`，核实 `max_retries` 的实际含义 |

协议参考 MCP Tools 2025-11-25。自动重试属于本次用户提出的运行要求。协议允许省略 `outputSchema` 和工具 annotations，因此报告将知识检索的这两项归为契约完整性缺口。协议将未知工具列为协议错误，将参数校验和执行失败列为工具错误。[MCP Tools 规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools)

本次新增报告和证据文件。业务代码整改列在建议中。完整发现结果、输入输出 schema、检查参数及错误文本已保存在 [HTTPS 检查原始证据](<F:/python project/FinanceKB/docs/mcp-comprehensive-audit-evidence.json>)。

## 动态发现

| 服务 | 路径 | 实测工具数 | 名称、描述、inputSchema | outputSchema | 只读 annotations | 结论 |
| --- | --- | ---: | --- | --- | --- | --- |
| mcp-market | `/mcp/market` | 3 | 全部存在，schema 语法有效 | 3/3 | 3/3 | 通过 |
| mcp-fundamentals | `/mcp/fundamentals` | 4 | 全部存在，schema 语法有效 | 4/4 | 4/4 | 通过 |
| mcp-macro | `/mcp/macro` | 2 | 全部存在，schema 语法有效 | 2/2 | 2/2 | 通过 |
| mcp-news | `/mcp/news` | 3 | 全部存在，schema 语法有效 | 3/3 | 3/3 | 通过 |
| mcp-knowledge | `/mcp/knowledge` | 2 | 全部存在，参数说明不完整 | 0/2 | 0/2 | 发现通过，契约部分实现 |

每套服务一次 `tools/list` 即返回完整列表，`nextCursor=null`。五套均声明 tools capability，`listChanged=false`。调用方能够主动刷新列表，服务当前采用固定注册列表。Research-Copilot 已实现分页读取、游标循环检查及最多 50 页保护。工具列表变化后的主动推送当前未启用。[服务注册](<F:/python project/FinanceKB/src/mcp/common/server.py:85>) · [调用方发现实现](<F:/python project/Research-Copilot/src/research_copilot/sources/mcp_client.py:269>)

## 逐工具检查

以下“通过”表示对应维度的实现与本次检查结果符合要求。“部分”表示已有实现，但存在列明的说明或校验缺口。所有工具的自动重试均为“缺失”，具体调用次数和失败处理见后面的重试表。

### mcp-market

| 工具 | 用途和返回值 | 参数及校验 | 描述检查与整改建议 | 动态发现 |
| --- | --- | --- | --- | --- |
| `resolve_symbol` | 按简称、裸代码或标准代码匹配沪深股票和三个宽基指数，返回代码、名称、交易所、类型、来源。歧义返回候选 | 必填 `keyword`，字符串长度 1–64，归一化后为空则拒绝。额外字段拒绝 | 部分。用途清楚，但“首批宽基指数”未列出范围，未说明当前为精确匹配及股票目录有最长约一小时的进程内缓存。建议列出三个指数，明确匹配规则 | 通过 |
| `get_ohlcv` | 返回 OHLC、成交量及单位、成交额、来源和时间。股票不复权，指数以点计价 | 必填 `symbol/start_date/end_date`。标准代码格式和目录校验，日期须真实且起始不晚于结束。默认 `interval=1d`，股票支持 1m/5m/15m/30m/60m/1d/1w/1M，指数限 1d/1w/1M。最多 5000 根，超限报错 | 部分。描述未显式列出 OHLCV 返回字段和 5000 根上限，“近期”未量化。建议说明东财 1 分钟线最多近期 5 个交易日，历史覆盖不足会报错，明确成交量单位随类型返回 | 通过 |
| `get_realtime_quote` | 返回股票价格或指数点位、来源时间、来源及可用的涨跌幅 | 必填 `symbols`，1–20 个标准代码，禁止重复，验证目录收录。额外字段拒绝 | 部分。主源成交时间与新浪快照时间已经区分。尚未说明任一标的缺报价时整体返回 `available=false, items=[]`，以及收盘后时间可以停留在交易时段。建议补充这两项语义 | 通过 |

依据为 [行情工具](<F:/python project/FinanceKB/src/queries/market.py:32>)、[类型与日期校验](<F:/python project/FinanceKB/src/queries/market_types.py:9>)、[目录匹配](<F:/python project/FinanceKB/src/queries/market_sources.py:140>) 和 [K 线来源](<F:/python project/FinanceKB/src/queries/market_sources.py:183>)。

### mcp-fundamentals

四个工具的 `symbol` 均必填，须为六位代码加 `.SH/.SZ`，并验证为收录股票。输入使用严格 JSON Schema 校验。

| 工具 | 用途和返回值 | 其他参数及校验 | 描述检查与整改建议 | 动态发现 |
| --- | --- | --- | --- | --- |
| `get_financial_statements` | 返回利润表、资产负债表、现金流量表，含报告期、公告日、期间口径、科目、数值、单位及来源 | `period=annual`，仅 annual/quarterly。`years=3`，整数 1–10，字符串整数被拒绝 | 部分。“从当前日期向前查询”与实际整年/整季边界不够一致。annual 截止本年 1 月 1 日，quarterly 截止本季度首日，起点为该边界向前 years 年。建议精确说明，补充季报累计口径及科目 Decimal 的序列化形式 | 通过 |
| `get_valuation_metrics` | 返回最近已收盘交易日的 PE TTM、PB MRQ、PS TTM，含交易日、定义和来源 | 只有 `symbol`。交易日历参与目标日期计算，拒绝用其他日期估值替代 | 通过，附改进建议。用途与实际日期选择一致。建议说明使用北京时间 15 点作为当日收盘判断，主源当日指标未更新时返回不可用 | 通过 |
| `get_company_profile` | 返回公司概况、董监高、十大股东、十大流通股东、实际控制人及来源，股东附报告期和持股比例分母 | 只有 `symbol` | 通过，附改进建议。主体和数据分组清晰。建议解释缺失字段为 null/空列表，以及各组报告期可能不同，便于外部系统使用 | 通过 |
| `get_dividend_history` | 返回分红方案、每股税前现金红利、送转、报告期和可用事件日期，按公告日倒序 | `limit=20`，整数 1–100 | 通过，附改进建议。每股金额单位已在输出 schema 说明。建议在工具描述中补充返回包含方案状态，便于调用方区分预案与实施 | 通过 |

依据为 [基本面工具](<F:/python project/FinanceKB/src/queries/fundamentals.py:44>) 和 [基本面输出模型](<F:/python project/FinanceKB/src/queries/fundamentals_types.py:12>)。

### mcp-macro

两个工具的可选日期均默认为空字符串，格式为 YYYY-MM-DD，包含端点。代码验证真实日期及先后顺序。两端均省略时使用工具描述的默认区间，只提供起点时向后展开默认窗口，只提供终点时向前展开。

| 工具 | 用途和返回值 | 参数及校验 | 描述检查与整改建议 | 动态发现 |
| --- | --- | --- | --- | --- |
| `get_macro_indicator` | 返回 CPI/GDP 历史观察期、指标值、口径、单位、来源及可用发布日期 | 必填 `indicator`，仅 CPI/GDP。日期默认为近 12 个月 | 部分。指标及累计口径明确，单边日期的默认补齐规则未说明。建议补充，并明确观察期日期与实际发布日期的区别 | 通过 |
| `get_economic_calendar` | 返回统计局计划发布日期、事件名称、原始日程页面 URL、来源 | 全部可选，默认今天起 30 天。请求涉及未配置年份时返回不可用 | 部分。当前只配置 2026 年，但描述未公布覆盖年份。“日程原文”实际返回页面链接。建议表述为“原始日程链接”，并列明年份与单边日期规则 | 通过 |

依据为 [宏观工具](<F:/python project/FinanceKB/src/queries/macro.py:12>)、[默认日期窗口](<F:/python project/FinanceKB/src/queries/content.py:31>) 和 [已配置年度来源](<F:/python project/FinanceKB/src/queries/macro_sources.py:14>)。

### mcp-news

| 工具 | 用途和返回值 | 参数及校验 | 描述检查与整改建议 | 动态发现 |
| --- | --- | --- | --- | --- |
| `search_news` | 对东财关键词搜索首批结果做日期过滤，按发布时间倒序返回标题、时间、媒体、URL、来源 | 必填 `keyword`，字符串 1–128，去空白后须为可打印非空文本。日期可选，默认近 7 天。`limit=20`，整数 1–100 | 不符合。参数说明仍写“已同步的公司或主题关键词”，与按需查询冲突。另 limit 默认 20、上限 100，但来源最多 10 条。建议修正参数文案，统一 limit 与覆盖范围，说明按来源首批结果筛选及单边日期规则 | 通过 |
| `get_filings` | 返回单股近期公告的标题、类型、披露日期、可识别财报期和详情链接 | 必填股票 `symbol`，东财目录验证。`limit=10`，整数 1–100 | 通过。工具用途、排序、返回值清楚，参数说明可独立理解 | 通过 |
| `get_earnings_calendar` | 按当前预约日或实际披露日匹配，返回首次预约、变更、当前预约、实际披露日与命中日期类型 | 日期均可选，默认今天起 30 天。校验真实日期及顺序，上游单种日期查询最多 20 页 | 部分。描述未明确返回字段、两种日期匹配为“或”、单边日期规则和查询容量。建议补充简短说明，保留完整输出 schema | 通过 |

依据为 [资讯工具](<F:/python project/FinanceKB/src/queries/news.py:25>)、[关键词的过期说明](<F:/python project/FinanceKB/src/queries/content_types.py:15>) 和 [财报预约分页上限](<F:/python project/FinanceKB/src/queries/news_sources.py:163>)。

### mcp-knowledge

| 工具 | 用途和返回值 | 已实现校验 | 不符合项与整改建议 | 动态发现 |
| --- | --- | --- | --- | --- |
| `search_research` | 检索 finance_report，返回相关片段、来源、页码、分数、分页状态 | query 必填，去空白后 1–500 字符。page 默认 1 且 ≥1，page_size 默认 8 且 1–50，并限制 page×page_size≤200。filters 校验券商列表、报告类型、日期与顺序 | 类型可转换，顶层额外参数被忽略。四个顶层参数均缺 description，报告类型允许值未进 enum，日期格式未进 schema，分页组合上限未说明。缺 outputSchema 和只读标记。建议统一严格入口、补全参数与返回契约 | 通过，契约部分实现 |
| `search_policy` | 检索 finance_policy，当前返回正常空 items，响应结构与研报一致 | 与 search_research 共用校验；filters 内部已有 extra=forbid | 存在相同缺口。HTTPS 实测 page="1"、page=true 和多余字段均被接受。建议用“成功时返回片段及分页信息，无匹配时 items 为空”描述返回语义，避免依赖调用方理解“200 语义” | 通过，契约部分实现 |

filters 当前支持 broker（最多 20 个，每个 1–64 字符）、report_type（最多 4 个，运行时检查允许值）、date_from/date_to（可空，真实 YYYY-MM-DD 且顺序正确）。这些验证已经存在，发现接口对取值含义的表达仍不足。[知识工具注册](<F:/python project/FinanceKB/src/mcp/knowledge/server.py:90>) · [请求与过滤模型](<F:/python project/FinanceKB/src/api/schemas/search.py:40>)

## 参数错误与协议错误

本次通过 HTTPS 执行了 65 个参数检查，其中 62 个被拒绝，3 个被接受。四个结构化域合计 53 个检查全部拒绝，覆盖每个工具的必填项或类型、枚举、长度、范围、日期、重复值和未知字段中的适用项。知识域 12 个检查中 9 个拒绝，3 个接受，详见原始证据。

| 情况 | 实际行为 | 判断 |
| --- | --- | --- |
| 四个结构化域的 schema 校验失败 | `CallToolResult.isError=true`，文本类似 `years 未通过 type 校验，请参见 tools/list` | 已实现 |
| 真实日期无效、日期倒置、空白关键词等业务校验失败 | 工具错误，文本说明具体原因 | 已实现 |
| 知识域缺必填 query、范围越界、无效过滤值 | 工具错误，包含 Pydantic 校验文本或自定义检索参数错误 | 已实现，但格式与四个结构化域不同 |
| 知识域 page="1"/page=true | 实际转换为整数后成功 | 严格类型校验不符合 |
| 知识域顶层未知字段 | 实际忽略后成功，schema 也未声明顶层 additionalProperties=false | 与仓库严格入参目标不一致 |
| 四个结构化域调用不存在工具 | JSON-RPC `-32602` | 已实现 |
| 知识域调用不存在工具 | `isError=true` 的工具错误 | 与未知工具的协议错误处理不一致 |

为验证两个知识工具的共性，又在独立 Python 进程中注入假的知识客户端，分别调用注册后的工具。两者均接受字符串页码、布尔页码和未知顶层字段。该检查验证本地参数入口行为，实际 RAG 查询质量属于其他验收范围。[严格入口](<F:/python project/FinanceKB/src/mcp/common/server.py:31>) · [知识域使用默认 FastMCP](<F:/python project/FinanceKB/src/mcp/knowledge/server.py:127>)

结构化输出的公共 `available` 说明也有歧义。当前文案“false 表示本次结果不是真实数据”与“缺指定交易日估值”“未覆盖年份”等返回情况不匹配。建议改成“本次请求是否取得所需数据，false 时查看 reason”，并明确成功空列表、业务不可用和工具执行错误三种结果。[公共响应模型](<F:/python project/FinanceKB/src/queries/base.py:22>)

## 重试机制

本节将“重试次数”定义为首次请求之后的额外尝试次数。下面均指单个上游请求，分页和多张财报查询可能产生多次不同请求。

| 调用链 | 实际触发行为 | 最大自动重试次数 | 间隔 | 最终处理 |
| --- | --- | ---: | --- | --- |
| Infoway SDK | 配置 max_retries=1；SDK 使用 range(max_retries)，因此仅首次请求。超时、限流也没有第二次尝试 | 0 | 无重试间隔。1.05 秒是进程内请求限频 | 转为脱敏 DataSourceError，由适配器按条件切换东财或新浪 |
| 行情东财、股票搜索、基本面东财 | 单次 httpx.get，timeout=8 秒 | 0 | 无 | 请求/格式异常转为 DataSourceError，通常以工具错误返回 |
| 新浪行情 | 单次 httpx.get，timeout=8 秒 | 0 | 无 | 失败报工具错误，返回缺报价时由查询层标记不可用 |
| 宏观东财、NBS、资讯东财 | 单次 client.get，timeout=10 秒 | 0 | 无 | 转为 ContentSourceError；分页总数变化、超上限也报错 |
| 知识 MCP → knowledge-api | 单次 AsyncClient.post，默认 timeout=120 秒，可由环境变量覆盖 | 0 | 无 | 超时映射 504，HTTP 错误分类，再返回脱敏 ToolError |
| Research-Copilot → MCP | 每次工具调用只入队一次，isError 原样交给上层 | 0 次工具调用自动重放 | 无 | 请求超时或连接异常失败；会话重建服务后续调用 |

Infoway SDK 自身具备限流/超时/HTTP 异常的重试分支和 `min(2**attempt, 8)` 秒退避代码，但当前配置使退避分支无法执行。独立进程注入 ReadTimeout 后，实际计数为 Infoway 1 次、宏观/资讯 HTTP 1 次、知识 API HTTP 1 次，均无自动重试。[Infoway 调用配置](<F:/python project/FinanceKB/src/queries/infoway_source.py:22>) · [公共内容 HTTP](<F:/python project/FinanceKB/src/queries/content_http.py:18>) · [知识 HTTP 客户端](<F:/python project/FinanceKB/src/mcp/knowledge/client.py:35>) · [调用方单次请求](<F:/python project/Research-Copilot/src/research_copilot/sources/mcp_client.py:516>)

| 工具 | 降级与失败处理补充 | 重试要求结论 |
| --- | --- | --- |
| resolve_symbol | Infoway 股票目录失败后查东财，指数使用固定映射 | 缺失 |
| get_ohlcv | 股票 Infoway 失败后查东财，K 线数量超限直接报错；指数直接查东财 | 缺失 |
| get_realtime_quote | Infoway 失败或缺少股票报价时查新浪；指数直接查新浪 | 缺失 |
| get_financial_statements | 目录有降级，财报读取本身是东财单源 | 缺失 |
| get_valuation_metrics | 目录和交易日历有降级，估值读取本身是东财单源 | 缺失 |
| get_company_profile | Infoway 公司概况失败后查东财，高管和股东仍为东财 | 缺失 |
| get_dividend_history | 目录有降级，分红读取本身是东财单源 | 缺失 |
| get_macro_indicator | 东财失败报错 | 缺失 |
| get_economic_calendar | NBS 失败报错，未配置年份返回不可用 | 缺失 |
| search_news | 东财失败报错 | 缺失 |
| get_filings | 东财目录或公告失败报错 | 缺失 |
| get_earnings_calendar | 东财失败或分页不一致报错 | 缺失 |
| search_research | 知识 API 失败转换为工具错误 | 缺失 |
| search_policy | 同上 | 缺失 |

## 整改建议及验收条件

| 优先级 | 问题 | 建议 | 验收条件 |
| --- | --- | --- | --- |
| P1 | 知识域两工具的严格参数校验和未知工具处理不一致 | 接入共用严格入口。同步限制顶层未知字段，保持 filters 的业务校验。为查询、分页、过滤字段补齐说明、enum 和日期格式 | 两工具均拒绝字符串/布尔页码及未知字段，未知工具返回协议错误，原有效检索请求继续通过 |
| P1 | 所有工具缺少自动请求重试 | 在来源 HTTP 层设置统一可配置策略。建议瞬时网络异常、超时、429、502/503/504 最多额外重试 2 次，退避 1 秒、2 秒并加入抖动，429 尊重 Retry-After。Infoway 514、鉴权失败和用户参数错误直接进入既定失败/降级逻辑。设置总体调用期限，避免上下游同时重试放大请求 | 用故障注入核验瞬时失败后恢复、次数耗尽、永久错误、取消与总体期限。记录 attempt_count、source、错误类型及是否降级，日志脱敏 |
| P2 | 新闻 keyword 参数仍要求“已同步” | 修正文案为任意公司或主题关键词，明确首批最多 10 条的覆盖及 limit 作用 | tools/list 文案与按需查询一致，任意合法关键词可进入上游查询 |
| P2 | 知识工具缺输出 schema、只读标记，描述依赖 HTTP API | 使用 SearchResponse 定义公开输出契约，添加 readOnlyHint，直接说明返回片段、来源页码及分页 | tools/list 返回对应契约，结构化响应通过 schema 校验，文本兼容返回包含足够结果信息 |
| P2 | 默认日期规则、财报 years 边界、NBS 年份和 available 含义不完整 | 按本报告逐项补充参数及结果说明，给单边日期补齐规则加示例 | 外部调用方仅根据 tools/list 即能构造正确参数并区分成功空结果、业务不可用和执行错误 |
| P2 | 重试决策缺少稳定机器可读错误分类 | 统一定义 invalid_argument、upstream_timeout、rate_limited、permission_denied、upstream_unavailable，并携带 retryable；沿用 MCP isError 标识执行错误 | 客户端可根据错误类别决定重试，且错误正文不包含凭据、堆栈或原始上游响应 |

以上为初次审计时的整改建议和验收条件。实施进展及剩余项以 [整改验收记录](mcp-remediation-acceptance.md) 为准，原审计证据保留用于前后对照。
