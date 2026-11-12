# FinanceKB 结构化 MCP 数据源接入实施方案

## 当前扩展（配置接线，未部署验收）

- `get_segment_revenue` 位于现有 `mcp-fundamentals`，基本面共 5 个工具；新增独立 `mcp-futures`，仅提供只读工具 `get_commodity_price(symbol)`，路径为 `/mcp/futures`。扩展后为结构化 5 套 + 知识 1 套，合计 **6 servers、16 tools**（行情 3、基本面 5、宏观 2、资讯 3、期货 1、知识 2）。
- 期货直连东财，返回最新期货报价快照，**不是现货价、结算价或日收盘价**；主连不是可交割的具体月份合约，不推算其对应主力月份。标识示例：`225.lcm`（碳酸锂主连）、`225.lc2705`（碳酸锂月份合约，可用性以当前目录为准）、`113.cum`（沪铜主连）。支持上期所 `113/SHFE`、大商所 `114/DCE`、郑商所 `115/CZCE`、上期能源 `142/INE`、广期所 `225/GFEX` 的主连及具体月份合约。
- 币种和报价单位未经可靠规格核实时，`currency`、`price_unit` 返回 `null`，不能默认元/吨。`source_time` 为源更新时间（Asia/Shanghai），不等于交易日或最后成交时间，缺失为 `null`；`as_of` 为响应生成的 UTC 时间。报价可能延迟或在非交易时段保持旧值，无实时性 SLA。
- 仅查询目标市场与品种目录：两次目录请求加一次报价请求，正常路径共三次 GET；目录与报价复用共享重试和总调用预算，无降级源。Compose 继承公共配置及本地结构化镜像，Caddy 原样转发 `/mcp/futures` 与 `/mcp/futures/*`，并等待期货服务健康；沿用现有鉴权，无新增密钥、宿主端口或权限。

本节说明当前扩展范围与配置接线。2026-11-12 离线验证：期货契约测试（123 项单测）与六套注册/集成测试通过，离线全量范围 349 通过、4 失败均为既有行情用例（`httpx.Response` 未带 request 实例，与本次改动无关）；主营构成含超 28 位精度与全局 Decimal context 不变回归，并完成"最新报告期 max→min"变异自查（9 个用例捕获后恢复）。同日免费低频冒烟：目录链路（futsse-static）主机与容器均通，`113.aum`（小数精度品种）经完整 MCP 链路端到端成功；其后 push2 报价端点因出口 IP 突发限流临时不可用，`225.lcm`、`225.lc2705`、`113.cum` 当次未能复测（三者的源响应形态已在方案批准前的只读核实中取到真实样本），限流恢复后可按[已批准期货方案](futures-mcp-plan.md)复测。部署与长期可用率未验收；下文既有运行记录与历史验收数字保留当时口径。

## 当前运行方式

2026-11-10 MCP 契约整改已部署：五套服务、14 个工具统一严格校验、输出 schema 和只读标记；上游瞬时故障默认最多额外重试 2 次，总调用时限 120 秒。知识检索同步纳入。14 个工具真实 HTTPS 调用通过，Research-Copilot 的 12 项审核指纹已更新。Infoway SDK 未暴露 Retry-After 响应头，该项经 2026-11-10 决议接受为已知边界、维持现状（详见整改与验收记录）。详见 [整改与验收记录](mcp-remediation-acceptance.md)。

2026-11-11 行情链路调整（决议）：**K 线统一直连东财、实时快照统一直连新浪，移除 Infoway 主源与降级链路**。原因是 Free 套餐无深市授权（514）且 600 次/日配额、2026-11-16 到期，降级路径已被验收为可用，改为事前直连可省去注定失败的调用。行情工具不再依赖 `INFOWAY_API_KEY`，也不再 import Infoway SDK。已知边界：东财分钟级历史有限（1 分钟约 5 个交易日，5/15/30/60 分钟约最近 7 周），日线可回溯多年；东财对突发高频访问会按出口 IP 临时限流，有界重试与工具总期限内可吸收短时封禁。

行情、基本面、宏观和资讯的 12 个工具在调用时查询上游。四套服务移除了 PostgreSQL 连接配置和原同步命令。股票代码解析经东财搜索接口实时完成，首批三个指数的代码映射随服务发布。研报 RAG 继续用 PostgreSQL 保存父块和整表索引，文档清单仍沿用原有写入流程。

| 数据 | 来源 |
| --- | --- |
| 沪深股票代码和简称 | 东财证券搜索 |
| 沪深股票最新价格 | 新浪行情快照，响应标注新浪与快照时间 |
| 沪深股票与指数 K 线 | 东财不复权 K 线；日线多年，1 分钟约 5 个交易日，5/15/30/60 分钟约最近 7 周 |
| 公司概况 | Infoway 主源，失败时东财 F10 公司资料，董监高和股东仍取东财 |
| 估值 | 东财最近已收盘交易日的 PE TTM、PB MRQ、PS TTM；确定估值交易日时，Infoway 交易日历失败则使用东财交易日历 |

基本面保留 Infoway 自动切换，覆盖其业务错误（含权限码 `514`）、请求失败和无效响应；行情已无双源切换。资讯仍使用东财，新闻接受任意关键词，日期筛选覆盖东财首批最多 10 条搜索结果。财报日历按当前预约日或实际披露日向东财取数并合并去重。宏观指标仍使用东财，计划发布日程使用已配置的国家统计局年度页面。

2026-11-10 收盘后更新本地 Compose。四个结构化容器均无 `DATABASE_URL` 或 `POSTGRES_*` 环境变量。通过本地 HTTPS 实测 12 个工具全部返回非空结果，非法参数和未知工具均被拒绝。2026-11-11 行情改直连后重新实测：resolve_symbol（东财搜索）、get_realtime_quote（新浪快照，含深市与沪市股票及两个指数）通过；get_ohlcv 东财日线/分钟线验证期间出口 IP 被东财临时限流，恢复后复测（见整改与验收记录）。

同次验收从 RAG 的 PostgreSQL 读到 733 条父块、138 条整表块，研报检索通过本地 HTTPS 返回了结果。旧的结构化快照表作为历史数据保留。Research-Copilot 更新了报价与公司概况的两项 Schema 指纹，客户端实测发现 12 个工具并调用了这两项。已绑定主机的配置更新沿用 Research-Copilot 现有绑定流程。

以下内容保留首期方案与分批验收历史。其中的同步命令、快照依赖和仅主源约束适用于当时的版本。

## 首期方案与分批验收

方案已确认，按批验收。首期覆盖行情 3 个、基本面 4 个、宏观 2 个、资讯 3 个，共 12 个 MCP 工具。行情包含沪深 A 股和首批宽基指数。基本面包含公司资料与分红。宏观首期覆盖中国 CPI、GDP 和国家统计局发布日程。

每类数据只接表中指定主源。同一个工具处理不同标的或数据分组时，按表中的固定来源取数。补源、自动切换、跨源补字段和复杂边界策略留待后续。必要的参数校验、请求上限、超时和错误返回纳入首期。研报 RAG、入库和附件解析各自独立。Research-Copilot 是首个接入方，其他系统可通过同一 MCP 契约发现并调用工具。[数据平台部署方案](data-platform-deploy.md)

## 工具描述与主源

“工具描述”列可作为 `tools/list` 中 `description` 的文案。参数级描述同步说明格式、枚举、默认值和返回口径。

| MCP | 工具 | 工具描述 | 首期主源 |
| --- | --- | --- | --- |
| `/mcp/market` | `resolve_symbol` | 按简称或代码查找沪深股票和首批宽基指数，返回标的代码、名称和类型。存在多个匹配时返回候选。 | 股票用 Infoway `basic.get_symbols`。指数用东财 `ulist.np/get` 读取首批标的的目录字段。 |
| `/mcp/market` | `get_ohlcv` | 按标的、日期区间和周期查询历史 K 线，返回开、高、低、收和可用的成交量。股票价格为不复权价格，指数价格字段以点为单位。 | 股票用 Infoway `stock.get_kline`。指数用东财 `index_zh_a_hist`，首期提供日、周、月线。 |
| `/mcp/market` | `get_realtime_quote` | 查询一个或多个标的的最新行情快照。股票返回最新成交价和成交时间，指数返回最新点位、涨跌幅和行情时间。 | 股票用 Infoway `stock.get_trade`，指数用新浪 `hq.sinajs.cn`。 |
| `/mcp/fundamentals` | `get_financial_statements` | 查询单家公司指定年数的利润表、资产负债表和现金流量表，返回报告期、公告日、科目原值与可确认的币种和单位。季报利润和现金流按年初至期末累计，资产负债表按期末时点。 | 东财 F10 `NewFinanceAnalysis`。 |
| `/mcp/fundamentals` | `get_valuation_metrics` | 查询单家公司最近已收盘交易日的 PE TTM、PB MRQ 和 PS TTM，返回估值日期和指标定义。 | 东财数据中心 `RPT_VALUEANALYSIS_DET`。交易日历用 Infoway。 |
| `/mcp/fundamentals` | `get_company_profile` | 查询单家公司的概况、公开披露的董监高与主要股东。返回行业、主营业务、任职信息、十大股东和十大流通股东，以及对应披露期。 | 公司概况用 Infoway `stock_info.get_company`，董监高用东财 F10 `CompanyManagement/PageAjax`，股东用东财 F10 `ShareholderResearch`。 |
| `/mcp/fundamentals` | `get_dividend_history` | 查询单只股票的分红事件，返回每股税前现金红利、送股转增、方案进度、报告期及可用的公告、登记、除息和发放日期。 | 东财 `stock_fhps_detail_em`。 |
| `/mcp/macro` | `get_macro_indicator` | 查询中国 CPI 或 GDP 的历史观测值。CPI 含全国、城市和农村的月度指数、同比、环比和累计指数。GDP 含全国及三次产业年初至季末累计值和累计同比。 | 东财数据中心 `RPT_ECONOMY_CPI`、`RPT_ECONOMY_GDP`，对应 AKShare 的 `macro_china_cpi`、`macro_china_gdp`。 |
| `/mcp/macro` | `get_economic_calendar` | 查询中国统计数据的计划发布日期，返回发布事项、计划日期和来源。 | 国家统计局年度统计信息发布日程。 |
| `/mcp/news` | `search_news` | 按已同步的公司或主题关键词查询本地新闻，返回标题、北京时间、媒体和来源链接。 | 东财资讯搜索 `cmsArticleWebOld`，对应 AKShare 的 `stock_news_em`。每次采集每个关键词最多 10 条。 |
| `/mcp/news` | `get_filings` | 查询单家公司最近的公告，返回标题、类型、披露日期、可从标题识别的财报期和详情链接。 | 东财公告 `api/security/ann`，对应 AKShare 的 `stock_individual_notice_report`。 |
| `/mcp/news` | `get_earnings_calendar` | 查询已同步报告期的沪深公司财报披露日程，返回首次预约、变更、当前预约和实际披露日期。 | 东财数据中心 `RPT_PUBLIC_BS_APPOIN`，对应 AKShare 的 `stock_yysj_em`。 |

首批指数为上证指数 `000001.SH`、深证成指 `399001.SZ` 和沪深 300 `000300.CSI`。行情结果标注 `instrument_type` 为 `stock` 或 `index`。沪深 300 的上游代码分别映射为东财 `1.000300` 和新浪 `sh000300`。裸代码 `000001` 可匹配上证指数和平安银行，由 `resolve_symbol` 返回候选。报价保留源时间，指数和股票的成交量按各自来源与单位解释。[东财指数行情](https://quote.eastmoney.com/center/hszs.html) · [AKShare 指数接口](https://akshare.akfamily.xyz/data/index/index.html) · [新浪沪深 300 行情](https://finance.sina.com.cn/realstock/company/sh000300/nc.shtml) · [中证指数代码示例](https://oss-ch.csindex.com.cn/static/html/csindex/public/uploads/indices/detail/files/zh_CN/930868factsheet.pdf)

## Tool calling 契约

以 MCP Tools 2025-11-25 为设计基线，沿用现有四个 Streamable HTTP 路径和 Bearer 鉴权。基本面用 `get_company_profile` 替换占位工具 `compare_companies`，新增 `get_dividend_history`。联调前更新调用方工具白名单。仓库声明 `mcp>=1.29,<2`、Pydantic 2，实施时以实际部署版本跑协议验证。[MCP Tools 规范](https://modelcontextprotocol.io/specification/2025-11-25/server/tools) · [现有服务装配](../src/mcp/common/server.py) · [依赖声明](../requirements.txt)

- `tools/list` 发布工具名、描述、对象型 `inputSchema` 和 `outputSchema`。输入 Schema 声明字段描述、必填项、类型、枚举、默认值与上下限，并用 `additionalProperties: false` 拒绝未知参数。按只读查询声明 `readOnlyHint`。消费方以工具发现实际返回的 JSON Schema 作为接入契约。
- `tools/call` 先校验参数对象、必填字段、未知字段和类型，再校验跨字段规则。拒绝隐式类型转换，如把字符串 `"10"` 作为整数。有效日期还需做日历解析，`start_date` 不得晚于 `end_date`。参数无效时说明字段与允许的值，不发起上游调用。用公开注册接口生成 Schema，并验证未知字段实际被拒绝。[FastMCP 工具定义](https://github.com/modelcontextprotocol/python-sdk/blob/v1.29.0/src/mcp/server/fastmcp/server.py)
- 正常结果在 `structuredContent` 返回符合 `outputSchema` 的 JSON 对象，并在 `content` 附同一结果的 JSON 文本供旧客户端读取。沿用 `QueryEnvelope` 的 `available`、`domain`、`reason`、`as_of`、`items`，为各工具的 `items` 声明具体字段类型。上游单位、来源、行情时间、报告期或观察期按工具记录。服务响应的 `as_of` 继续用 UTC，业务日期按北京时间解释。[响应契约](../src/queries/base.py) · [MCP 结构化结果](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#structured-content)
- 查询成功且确认无记录时返回 `available=true` 和空 `items`。数据范围不可用时返回 `available=false` 及可诊断原因。参数错误、主源鉴权失败、请求超时或上游响应异常用 `isError=true`。四套结构化服务在 SDK 执行工具前检查已注册工具名，未知工具返回 JSON-RPC `-32602`，畸形请求由 SDK 的协议解析层处理。错误信息使用脱敏的简短描述。[MCP 错误处理](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#error-handling) · [SDK 响应处理](https://github.com/modelcontextprotocol/python-sdk/blob/v1.30.0/src/mcp/server/lowlevel/server.py)

### 工具参数与基本校验

日期参数为 `start_date`、`end_date`，格式为 `YYYY-MM-DD`，区间含首尾。日期可选时沿用省略或空字符串表示缺省。只传一个日期时，以该日期为锚点，按该工具默认窗口的长度补另一端。业务查询使用 `resolve_symbol` 返回的六位数字加大写后缀代码，首期后缀为 `.SH`、`.SZ` 和 `.CSI`。服务端按证券字典核对标的类型，基本面和公司公告只接受沪深股票。下表中的默认窗口和上限为首期拟定值。

| 工具 | 参数与默认值 | 服务端规则 |
| --- | --- | --- |
| `resolve_symbol` | `keyword` 必填字符串 | 去除首尾空白后长度 1–64，可输入公司简称、指数名称、裸代码或标准代码。 |
| `get_ohlcv` | `symbol`、`start_date`、`end_date` 必填，`interval` 默认 `1d` | 股票周期为 `1m/5m/15m/30m/60m/1d/1w/1M`，指数为 `1d/1w/1M`。按标的类型校验周期，单次最多 5,000 根，真实可用范围以主源验收为准。 |
| `get_realtime_quote` | `symbols` 必填字符串数组 | 1–20 个标准代码，同一代码只传一次。股票和指数分别路由到各自的主源。 |
| `get_financial_statements` | `symbol` 必填，`period=annual`，`years=3` | `annual` 仅查年报，`quarterly` 查各季末（含年末），`years` 为 1–10 的整数。 |
| `get_valuation_metrics` | `symbol` 必填 | 仅返回目标交易日的指标。东财提供交易日期，`source_time` 留空。 |
| `get_company_profile` | `symbol` 必填 | 管理层任职时间和股东报告期分别记录。股东持股比例注明分母为总股本或流通股本。 |
| `get_dividend_history` | `symbol` 必填，`limit=20` | `limit` 为 1–100 的整数。按公告日倒序返回，区分预案与实施。 |
| `get_macro_indicator` | `indicator` 必填，日期缺省为截至当天的最近 12 个月 | `indicator` 枚举为 `CPI/GDP`。按月末或季末观察期筛选。东财未给出实际发布日期时，`publication_date` 为空。 |
| `get_economic_calendar` | 日期缺省为从当天起 30 天 | 按计划发布日期筛选，保留事件名称与日程来源。 |
| `search_news` | `keyword` 必填，`limit=20`，日期缺省为截至当天的最近 7 天 | 关键词去除首尾空白后长度 1–128，`limit` 为 1–100 的整数。按发布时间倒序，最多返回 `limit` 条已采集记录。 |
| `get_filings` | `symbol` 必填，`limit=10` | `limit` 为 1–100 的整数。按实际披露日期倒序返回。 |
| `get_earnings_calendar` | 日期缺省为从当天起 30 天 | 按当前预约日或实际披露日匹配，标明命中的日期类型。当前预约取最后一次有效变更，未变更时取首次预约。 |

下面是一组查询沪深 300 日 K 的 `tools/call` 参数。

```json
{
  "name": "get_ohlcv",
  "arguments": {
    "symbol": "000300.CSI",
    "start_date": "2026-11-02",
    "end_date": "2026-11-06",
    "interval": "1d"
  }
}
```

## 主源验证与验收

- **行情。** 东财已实测上证指数、深证成指、沪深 300 各 5 条日 K。新浪已实测三只指数的收盘后行情与源时间，盘中时效需在目标部署环境验证。Infoway 个股样本中茅台可取 K 线与最新成交，五粮液返回 `514`，当前免费套餐样本限频每秒 1 次。验收按证券和周期记录实际权限、历史覆盖与更新时间。[Infoway K 线](https://docs.infoway.io/rest-api/http-endpoints/get-candles) · [Infoway 免费套餐](https://docs.infoway.io/getting-started/api-limitation/free-plan)
- **基本面。** 茅台和五粮液的东财三表已核对报告期、关键科目与累计口径，估值已核对目标交易日，董监高已核对董事长及任职日期。股东及分红复验见第二批记录。公司概况来自 Infoway。[Infoway 公司概览](https://docs.infoway.io/rest-api/basic-info/get-stock-fundamental) · [东财股东研究](https://emweb.eastmoney.com/PC_HSF10/ShareholderResearch/Index?code=SH600519&color=w&type=soft) · [东财分红接口](https://github.com/akfamily/akshare/blob/main/akshare/stock/issue/stock_fhps_em.py)
- **宏观。** 已核对 2026 年 10 月 CPI、前三季度 GDP 及国家统计局年度日程。第三批验收记录列出对账值和同步范围。[东财宏观接口参考](https://github.com/akfamily/akshare/blob/main/akshare/macro/macro_china.py) · [国家统计局发布日程](https://www.stats.gov.cn/xw/tjxw/tzgg/202512/t20251224_1962137.html)
- **资讯。** 样本已覆盖茅台、五粮液和半导体新闻，两股公告及沪深 2026Q3 财报预约和实际披露日期。按来源、证券或主题、公告日和预约字段核对返回。新闻按本地已同步范围提供查询。公告用公开详情链接作为证据。[AKShare 股票资讯与业绩预约](https://akshare.akfamily.xyz/data/stock/stock_research.html) · [东财公告样例](https://data.eastmoney.com/notices/detail/600519/AN202610301831245500.html)

## 实施批次

复用部署方案中的 PostgreSQL 保存结构化记录，采集端校验后按自然键幂等写入，工具读取已同步数据。实时行情使用指定主源，带请求超时和限频。Infoway 密钥从运行环境变量 `INFOWAY_API_KEY` 读取。主源权限或覆盖未达到确认目标时，记录样本和差距，暂停对应能力验收。[数据平台部署方案](data-platform-deploy.md)

| 批次 | 范围 | 验收 |
| --- | --- | --- |
| 1. 基础与行情 | 运行配置、主源适配与表迁移，`src/queries/market.py` 及行情 MCP | 三个工具覆盖沪深股票和首批指数，验证代码消歧、历史 K 线、真实行情快照与主源权限。 |
| 2. 基本面 | 三表、估值、公司资料和分红，`src/queries/fundamentals.py` 及基本面 MCP | 四个工具返回类型化结果，两股财报与分红按公告对账，公司资料、高管和股东的来源与日期可追溯。 |
| 3. 宏观与资讯 | 指标和日历、新闻、公告及业绩预约，`src/queries/macro.py`、`src/queries/news.py` 及对应 MCP | 五个工具各有主源成功样本，日期、单位、来源与披露状态准确。 |
| 4. 协议及消费方联调 | 四个 MCP 与 Research-Copilot，另用独立 MCP 客户端复核 | 发现恰好 12 个工具。核验鉴权、描述、输入与输出 Schema、结构化结果及错误分层。每个工具验证一次正常调用和至少一个无效参数案例，抽验未知字段、非法日期、主源错误、空结果及旧客户端文本兼容。离线契约和部署环境实测分别记录。 |

各批按已确认的范围验收。行情股票池、历史覆盖和盘中时效以实际主源授权及运行数据为准。

## 第一批验收记录

**基础与行情已实现，已验证范围通过。** 2026-11-09 使用独立 PostgreSQL 16 和 MCP Python SDK 1.30.0 执行验收。MCP 单测 40 项通过。独立 MCP 客户端通过真实 Streamable HTTP 验证 Bearer 鉴权、三个工具的发现与 Schema、额外参数和非法日期拒绝、结构化结果与文本兼容、正常空结果和主源错误。SDK 1.30.0 实测将未知工具返回为工具错误，和协议中的 JSON-RPC 示例存在差异，纳入第四批协议验收。[MCP 错误处理](https://modelcontextprotocol.io/specification/2025-11-25/server/tools#error-handling)

Infoway 返回 5,263 条沪深股票目录。茅台的收盘后快照可用，8 个 K 线周期在隔离 Docker 环境取到样本。K 线成交量按每手 100 股归一化，并与成交额对账。隔离 Docker 中东财返回三只指数共 15 根日 K，重复写入仍为 15 条。三只指数的东财周 K 和月 K 也已取到样本，新浪返回三只指数的收盘后快照。五粮液的 Infoway 日 K 和快照均返回 `514`，股票目录覆盖与行情授权分别验收。按确认范围，深市个股行情授权与盘中时效列为待补验，基本面批次继续执行。[Infoway 免费套餐](https://docs.infoway.io/getting-started/api-limitation/free-plan)

在已构建的行情容器内设置 `DATABASE_URL` 和 `INFOWAY_API_KEY` 后，可用以下命令初始化行情表、同步目录并装载一段历史 K 线。

Infoway 请求按进程限频。使用每秒 1 次的套餐时，目录和 K 线同步安排在报价调用较少的时段。

```bash
python -m src.mcp.market.sync --catalog all
python -m src.mcp.market.sync --symbols 600519.SH 000300.CSI --start-date 2026-11-02 --end-date 2026-11-06
```

## 第二批验收记录

**切换东财前的首次验收。** 基本面四个工具已实现，分红和公司资料的已验证部分通过，财报、估值和高管当时待补验。验收使用隔离 PostgreSQL 16、MCP Python SDK 1.30.0、Infoway SDK 0.4.1 和独立 Streamable HTTP 客户端。MCP 单测 42 项通过。实际验证四个工具的鉴权、发现、只读声明、输入与输出 Schema、参数错误和结构化结果。分红重复同步后，数据库仍为两股各五个数据集加交易日历，共 11 个快照。

| 工具 | 实测结果 | 数据验收 |
| --- | --- | --- |
| `get_financial_statements` | 茅台、五粮液的三表均能入库并通过 MCP 查询。两家公司 2024 年总资产及归母净利润样本与年报一致。 | 两股 Infoway 最新年报为 2024 年、最新季报为 2025Q3。2024 年 `total_revenue` 茅台为 1476.44 亿元、五粮液为 763.40 亿元，与公告营业收入 1708.99 和 891.75 亿元未能对齐。两股经营现金流科目也有差异。主源科目保留原始名称、数值和报告期，相关口径与覆盖待补验。 |
| `get_valuation_metrics` | 两股均同步了 PE、PB、PS 序列，并按 Infoway 交易日历计算最近已收盘交易日。 | 2026-11-09 查询的目标交易日为当天，源最新原始时间戳落在 11 月 8 日。工具返回该交易日不可用。 |
| `get_company_profile` | Infoway 返回两股公司概况。东财返回两股 2026Q3 的十大股东和十大流通股东，每份 10 条。隔离 Docker 中复验通过。 | 两股 Infoway 公司概况响应均缺董监高清单，工具标记 `executives_reported=false`。 |
| `get_dividend_history` | 东财各返回 28 条记录。2025 年度实施方案中，茅台每股税前 28.02423 元、除权除息日为 2026-06-26。五粮液每股税前 2.5796852 元、除权除息日为 2026-07-16。金额、登记日和除权除息日与公告核对通过，隔离 Docker 中复验通过。 | 本次东财样本缺现金发放日，字段保留为空。 |

[贵州茅台 2024 年报摘要](https://www.moutaichina.com/mtgf/attachDir/2025/04/2025040817544294292.pdf) · [五粮液 2024 年报摘要](https://static.cninfo.com.cn/finalpage/2025-04-26/1223311582.PDF) · [茅台权益分派公告](https://vip.stock.finance.sina.com.cn/corp/view/vCB_AllBulletinDetail.php?CompanyCode=10000351&gather=1&id=12402962) · [五粮液权益分派公告](https://static.cninfo.com.cn/finalpage/2026-07-10/1225417855.PDF)

在行情股票目录已同步、基本面容器已配置 `DATABASE_URL` 和 `INFOWAY_API_KEY` 后，可用以下命令同步基本面。按数据集验收时可通过 `--datasets` 选择 `annual`、`quarterly`、`valuation`、`profile` 或 `dividend`。

```bash
python -m src.mcp.fundamentals.sync --symbols 600519.SH 000858.SZ
```

### 2026-11-09 Infoway 主源复验

复验直接读取 Infoway 原始 REST 响应，并用上市公司报告核对。基本面契约回归用例 5 项通过。下面三项仍待补验。

- **三表覆盖与口径。** 对茅台和五粮液分别调用利润表、资产负债表和现金流量表，使用 `type=STOCK_CN` 并省略 `period_type`。六次请求均成功，三表的最新年报仍为 2024 年，最新季报仍为 2025Q3。Infoway 财报发布状态接口分别返回茅台 2026Q2 和五粮液 2026Q3，而两家公司已于 10 月披露 2026Q3。收入明细只到 2024 年，财务指标统计的报期与三表一致。发布状态和收入明细的覆盖与三表独立。[Infoway 财务接口](https://docs.infoway.io/rest-api/basic-info/get-stock-financial-data) · [茅台 2026Q3 上交所公告](https://www.sse.com.cn/assortment/stock/list/info/announcement/index.shtml?productId=600519) · [五粮液 2026Q3 报告](https://static.cninfo.com.cn/finalpage/2026-10-28/1225630063.PDF)
- **财报数值对账。** 两股 2024 年度的总资产和归母净利润样本仍一致。以下是主源原值与公司年报披露值的对照。两个利润表请求的 `traceId` 分别为 `fbe61740-845f-4382-9182-13d5f9e0b84b` 和 `2091f2f4-ac55-4e3b-aff4-9ceec5fe45c4`。[茅台 2024 年报摘要](https://www.moutaichina.com/mtgf/attachDir/2025/04/2025040817544294292.pdf) · [五粮液 2024 年报摘要](https://static.cninfo.com.cn/finalpage/2025-04-26/1223311582.PDF)

| 股票 | Infoway 科目 | Infoway 原值 | 年报对应披露值（元） |
| --- | --- | ---: | ---: |
| 贵州茅台 | `total_revenue` | 147,644,119,745.66 | 170,899,152,276.34 |
| 五粮液 | `total_revenue` | 76,339,858,222.81 | 89,175,178,322.70 |
| 贵州茅台 | `cash_f_operating_activities` | 92,556,074,320.09 | 92,463,692,168.43 |
| 五粮液 | `cash_f_operating_activities` | 33,953,707,832.78 | 33,939,755,192.78 |

- **估值交易日。** 同源交易日历确认 11 月 6 日和 9 日为交易日。两股 PE、PB、PS 在 11 月 7–8 日的数值均与 11 月 6 日相同，原始 `updated_at` 为 11 月 8 日。11 月 9 日收盘后复验仍无当天时点，工具继续返回目标交易日不可用。[Infoway 估值接口](https://docs.infoway.io/rest-api/basic-info/get-stock-fundamental) · [Infoway 交易日历](https://docs.infoway.io/rest-api/basic-info/get-market-trading-days)
- **董监高清单。** 两股公司概览分别以 `zh-CN` 和 `en` 读取，四次请求均成功，原始响应都只有 `basic_info`、`company_profile`、`historical` 和 `related` 四组字段，均未找到高管清单。工具继续标记 `executives_reported=false`。[Infoway 公司概览](https://docs.infoway.io/rest-api/basic-info/get-stock-fundamental)

### 2026-11-09 东财主源切换验收

财报、估值和董监高三项已切换到东财。用隔离 PostgreSQL 16 实时拉取贵州茅台 `600519.SH` 和五粮液 `000858.SZ` 的主源数据，再通过 MCP Python SDK 1.30.0 独立客户端调用四个工具。验证鉴权、工具发现、输入和输出 Schema、错误参数，以及结构化结果和文本结果的一致性。另用空目录隔离库运行 MCP 回归，44 项单测和 1 项协议集成测试通过。

| 项目 | 两股样本结果 |
| --- | --- |
| 财报 | 均取到 2025 年报及 2026 三季报的三张表。两期营业收入、归母净利润、资产总计和经营活动现金流量净额逐项与公告相符。2024 年报此前有差异的营业收入和经营现金流也已对齐。营业收入与营业总收入分别返回。三季报利润表和现金流量表标记为年初至期末累计，资产负债表标记为期末时点。东财响应没有明确的合并范围字段，`reporting_scope` 留空。 |
| 估值 | 两股均返回 2026-11-09 的 PE TTM、PB MRQ、PS TTM。茅台分别为 21.98562565、6.66048613、10.35552752，五粮液为 23.79844005、2.44950014、6.78634926。 |
| 董监高 | 茅台返回 12 条，五粮液返回 14 条。抽验董事长及任职起日，茅台陈华为 2025-11-28，五粮液邓敏为 2026-06-26，与董事会公告相符。 |
| 股东及分红 | 两股均返回 2026 三季报的两份十大股东清单。2025 年度分红分别为每股税前 28.02423 元和 2.5796852 元。 |

财报、估值和公司资料分别写入新的数据集键，查询在新快照写入前返回未同步。隔离库预置的 8 个旧快照保持原样，重复同步两股估值后仍为 19 个快照。本次为两股样本验收，目标环境部署、全市场覆盖和持续更新稳定性待验证。

[茅台 2025 年报](https://big5.sse.com.cn/site/cht/www.sse.com.cn/disclosure/listedinfo/announcement/c/new/2026-04-17/600519_20260417_9QS4.pdf) · [茅台 2026 三季报](https://static.cninfo.com.cn/finalpage/2026-10-31/1225649896.PDF) · [五粮液 2025 年报摘要](https://static.cninfo.com.cn/finalpage/2026-04-30/1225273090.PDF) · [五粮液 2026 三季报](https://static.cninfo.com.cn/finalpage/2026-10-28/1225630063.PDF)

[茅台估值](https://data.eastmoney.com/gzfx/detail/600519.html) · [五粮液估值](https://data.eastmoney.com/gzfx/detail/000858.html) · [茅台董事会公告](https://static.cninfo.com.cn/finalpage/2025-11-29/1224836337.PDF) · [五粮液董事会公告](https://disc.static.szse.cn/disc/disk03/finalpage/2026-06-27/11ac4fbb-13e6-45fc-83cf-36522e62803f.PDF)

## 第三批验收记录

**宏观与资讯五个工具通过主源样本验收。** 2026-11-09 在隔离 PostgreSQL 16 中同步真实主源数据，使用 MCP Python SDK 1.30.0 独立客户端调用 `/mcp/macro` 和 `/mcp/news`。两个服务均验证 Bearer 鉴权、工具发现、只读声明、输入和输出 Schema、无效参数拒绝，以及结构化结果与文本结果一致。61 项 MCP 单测和 1 项协议集成测试通过。

| 工具 | 真实数据与核对结果 |
| --- | --- |
| `get_macro_indicator` | 入库 226 个 CPI 和 83 个 GDP 观察期。2026 年 10 月全国 CPI 同比 +0.7%、环比 0%。2026 年前三季度全国 GDP 累计 1,071,071.2 亿元、累计同比 +4.6%，与国家统计局公布值一致。东财提供观察期，实际发布日期字段留空。[CPI 发布](https://www.stats.gov.cn/sj/zxfb/202611/t20261109_1965795.html) · [GDP 发布](https://www.stats.gov.cn/english/PressRelease/202610/t20261019_1965619.html) |
| `get_economic_calendar` | 从国家统计局 2026 年日程解析 142 个事项。11 月 9 日的 CPI 日程和 3 月 4 日、31 日的两项 PMI 日程与原表一致。未同步的年度返回不可用原因。[年度日程原表](https://www.stats.gov.cn/xw/tjxw/tzgg/202512/t20251224_1962137.html) |
| `search_news` | 贵州茅台、五粮液、半导体三个关键词各同步 10 条。验证标题、媒体、链接、北京时间和按发布时间倒序。已同步关键词在指定日期无匹配时返回可用空结果，未同步关键词返回不可用。[东财新闻接口参考](https://github.com/akfamily/akshare/blob/main/akshare/stock/research/stock_news_stock.py) |
| `get_filings` | 茅台和五粮液各同步 100 条公告。2026 三季报的报告期均为 9 月 30 日，两股披露日分别为 10 月 31 日和 10 月 28 日，详情链接可追溯。[茅台公告](https://data.eastmoney.com/notices/detail/600519/AN202610301831245500.html) · [五粮液公告](https://data.eastmoney.com/notices/detail/000858/AN202610271831037454.html) |
| `get_earnings_calendar` | 同步 5,222 条沪深 2026 三季报记录，其中 200 条的当前预约日期和首次预约日期不同。茅台、五粮液的预约与实际披露日均与公告对应，单日查询返回命中的日期类型。[东财财报预约](https://data.eastmoney.com/bbsj/202609/yysj.html) |

重复执行同一批同步后，9 个范围快照的记录数一致。新闻和公告按源记录 ID 合并，已采集旧记录可以继续按日期查询。指标、统计日程和财报预约按范围替换快照，预约变更以最近一次同步的主源为准。生产部署、定时采集和持续更新稳定性留待后续验证。

在配置 `DATABASE_URL` 的环境执行同步。个股公告依赖已有的行情证券目录。统计日程首期已登记 2026 年，新闻按传入关键词同步，财报预约按传入的季末报告期同步。

```bash
python -m src.mcp.macro.sync --indicators CPI GDP --years 2026
python -m src.mcp.news.sync --keywords 贵州茅台 五粮液 半导体 --symbols 600519.SH 000858.SZ --report-dates 2026-09-30
```

## 第四批验收与交接

2026-11-09 使用隔离 Docker 网络启动四个结构化服务、Caddy 和 PostgreSQL 16。通过 Caddy 的 HTTPS 入口验证证书、Bearer 鉴权和 MCP Streamable HTTP。FinanceKB 容器与 Research-Copilot 客户端均使用 MCP Python SDK 1.30.0。行情、基本面、宏观和资讯使用各自已确认的主源，重新同步样本后执行验收。

| 验收项 | 结果 |
| --- | --- |
| 独立 MCP 客户端 | 四套端点分别发现 3、4、2、3 个工具。12 个正常调用和 12 个非法参数用例通过，核对了只读声明、输入与输出 Schema 和结构化结果与旧客户端文本的一致性。 |
| 错误分层 | 四套端点的未知工具均返回 JSON-RPC `-32602`。畸形 `tools/call` 请求也返回协议错误。已知工具的非法参数和上游失败返回 `isError=true`。已同步但无匹配，以及查询范围未同步，各自按信封契约返回。 |
| Research-Copilot | 通过真实主机绑定、HTTPS MCP 会话和 Searcher 工具节点调用 12 个工具，生成 12 条可追溯的证据。逐项核对工具名、Schema 指纹、结构化内容和证据文件。非法参数、不可用结果经原有错误路径返回。测试输入直接构造工具调用消息，模型调用为零。 |

FinanceKB 的 64 项 MCP 回归通过，另用 SDK 1.29.0 复验未知工具协议错误。Research-Copilot 离线回归 851 项通过、6 项跳过，排除 152 项标记为付费模型、数据库或 Docker 的用例。上表的真实 HTTPS、PostgreSQL 与 Searcher 联调单独执行。

Research-Copilot 的接入默认关闭。启用时配置 `RC_FINANCEKB_BASE_URL` 为包含可选端口的 HTTPS 源站，`RC_FINANCEKB_API_TOKEN` 使用服务端 `MCP_API_TOKENS` 中的一个令牌。主机逐域核对只读工具白名单以及输入、输出 Schema 的指纹，发生变化时该域工具不会挂载。通过审核的工具以 `financekb_` 前缀交给 Searcher。已知的新闻发布时间、公告披露日和财报公告日进入现有的研究截止时间检查，原始结果及时间字段保存在证据中。

配置中的 HTTPS 源站和审核后的工具契约参与 Research-Copilot 的主机配置指纹，令牌轮换无需变更指纹。对已绑定的目标主机启用 FinanceKB 时，按 Research-Copilot 的现有运维流程审核配置并显式更新主机绑定。目标使用私有 CA 时，客户端进程需信任对应 CA。

本批验收使用贵州茅台和沪深 300 的收盘后行情。五粮液实时行情复验仍返回 Infoway `514`，深市个股权限与盘中时效继续列为待补验。现有部署实例更新、定时采集和持续更新稳定性属于上线验收。

## 2026-11-10 本地部署验收

四套结构化服务已更新到本地 Compose 的 `financekb-structured-mcp:local` 镜像。四个服务共用镜像和源码，运行 MCP SDK 1.30.0。行情和基本面从忽略提交的 `.env.local` 注入 Infoway 凭据，使用 SDK 0.4.1。原 Caddy HTTPS 入口和 PostgreSQL `finance_rag` 用于本次验收。

通过 `https://localhost/mcp/{domain}` 实测发现 12 个工具。12 个正常调用均返回非空结构化数据，12 个非法参数调用均返回工具错误。四域的未鉴权请求均返回 401，未知工具均返回 JSON-RPC `-32602`。还核验了结构化结果与兼容文本一致，以及可用空结果与未同步结果的区别。

| 数据域 | 入库范围与实测结果 |
| --- | --- |
| 行情 | 5,263 只股票与 3 只指数。贵州茅台及上证指数、深证成指、沪深 300 各同步 2026-11-02 至 11-09 的 6 根日 K。四个标的在 11:13 至 11:14 两次盘中快照中，源时间均向前推进。 |
| 基本面 | 贵州茅台、五粮液的财报、估值、公司资料和分红快照，加交易日历共 11 个快照。两股均返回 2025 年报、2026 三季报、11 月 9 日估值、董监高、十大股东和分红。 |
| 宏观 | 226 个 CPI 观察期、83 个 GDP 观察期和 142 项 2026 年国家统计局日程。 |
| 资讯 | 三个关键词各同步 10 条新闻，两股各同步 100 条公告，2026 三季报日程同步 5,222 条。 |

**待补验项为深市个股行情授权。** 在目标环境单独调用五粮液实时行情并同步其日 K，两项均返回 Infoway `514`。当前沿用已确认的主源范围，实际估值和新闻时效取决于后续同步。[Infoway 免费套餐授权](https://docs.infoway.io/getting-started/api-limitation/free-plan)

在仓库根目录可按以下命令重建这四套服务的共享本地镜像并更新容器。同步入口见各批验收记录。

```powershell
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.local.yml build mcp-market
docker compose --env-file .env.local -f docker-compose.yml -f docker-compose.local.yml up -d --no-deps --no-build --pull never --wait mcp-market mcp-fundamentals mcp-macro mcp-news
```

### 2026-11-10 收盘后补验

本机 Docker Linux 引擎启动后仍不可达，四套本地 HTTPS 入口未响应。本节记录上游直测，与上午已完成的目标环境验收分开。

| 项目 | 补验证据 | 备选或处置 |
| --- | --- | --- |
| Infoway 深市个股行情 | 用已配置的 Key 分别查询五粮液、平安银行的末笔成交，两次均返回 `514`。五粮液日 K 仍返回 `514`，茅台返回当日收盘成交。上午另曾遇到一次批量调用 `429`。 | 需要逐笔或末笔成交口径时，先确认 Infoway 正式 A 股行情授权。按 Key 共享限频，区分 `429` 限频与 `514` 授权错误。[Infoway 免费套餐](https://docs.infoway.io/getting-started/api-limitation/free-plan) · [REST 限频](https://docs.infoway.io/en-docs/getting-started/api-limitation/rest-api-limitation) |
| 个股 K 线候选 | 东财两股均取到 11 月 6 日至 10 日的 3 根不复权日 K 和 144 根 5 分钟 K。五粮液 1 分钟样本覆盖 11 月 4 日至 10 日的 5 个交易日。 | 东财适合先验收日 K 和近期分钟线，历史分钟覆盖、成交量单位和跨日可用性分别验收。[AKShare 东财 K 线实现](https://github.com/akfamily/akshare/blob/main/akshare/stock/quote/stock_hist_em.py) |
| 报价快照候选 | 新浪和东财均返回两股的收盘后快照，两股价格均与东财日 K 收盘价一致。新浪三只宽基指数也均有当日数据。 | 新浪、东财可作为行情快照候选。保留供应商的快照时间口径。逐笔或末笔成交需求单独按授权成交源验收。[新浪五粮液](https://finance.sina.com.cn/realstock/company/sz000858/nc.shtml) · [东财五粮液](https://quote.eastmoney.com/sz000858.html) |
| 估值与新闻更新 | 东财两股均已有 11 月 10 日的三项估值。半导体搜索返回 10 条，样本最新发布时间为 16:52:48，北京时间。新浪五粮液公司资讯页也有当日新闻。 | 估值按最近收盘交易日校验后同步，新闻按关键词续采并按文章 ID 去重。新浪公司资讯可作按股候选。主题历史覆盖需要支持分页的来源或授权新闻接口。[东财五粮液估值](https://data.eastmoney.com/gzfx/detail/000858.html) · [新浪公司资讯](https://finance.sina.com.cn/realstock/company/sz000858/nc.shtml#) |

备选源为样本直测。跨交易日稳定性和下游使用授权仍需验证，当前服务保持已确认的主源配置。

#### Docker 恢复后的本地复验

四套本地 HTTPS 健康检查均返回 200。补同步前，11 月 10 日的样本日 K 未入库，两股估值均返回 `valuation_unavailable_for_trade_date:2026-11-10`。使用现有主源补同步后，通过本地 HTTPS 确认茅台和三只宽基指数各返回 11 月 10 日的 1 根日 K，茅台和五粮液均返回当天的 PE TTM、PB MRQ、PS TTM。

对三个已配置关键词续采后，11 月 10 日的茅台、五粮液、半导体新闻分别由 0、1、7 条变为 10、1、11 条。半导体最新发布时间从 10:50:57 更新到 16:52:48，北京时间。茅台和三只指数均返回当日收盘后快照。五粮液和平安银行通过本地 HTTPS 单独查询实时行情仍返回 `514`，深市授权继续待补验。
