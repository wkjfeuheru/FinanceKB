# 独立期货 MCP：get_commodity_price

## Context

业务需要碳酸锂等商品的期货价格。用户明确要求来源为东财，并新增独立期货 MCP server，不挤入现有股票 market 服务。本次提供最新报价快照；期货不等于现货，主连不等于可交割的具体月份合约。

当前为计划模式，仅完成只读核实。批准后先保存本计划到项目 `docs/futures-mcp-plan.md`，再实施。不会自动部署、重启现有容器、提交或推送代码；不调用付费模型或 Research-Copilot 端到端验收，不改冻结评测基线。

## 推荐范围与对外合同

- 服务：`mcp-futures`，模块 `src.mcp.futures`，Streamable HTTP 路径 `/mcp/futures`。
- 唯一工具：`get_commodity_price(symbol)`，只读，返回 `QueryEnvelope[CommodityPrice]`，`domain="futures"`。
- `symbol` 为东财合约标识（secid），例如：
  - `225.lcm`：碳酸锂主连。
  - `225.lc2705`：碳酸锂具体月份合约；可用性以当时目录为准。
  - `113.cum`：沪铜主连。
- 支持商品期货市场：113/SHFE、114/DCE、115/CZCE、142/INE、225/GFEX；仅主连及具体月份合约。首版不支持股票、金融期货、境外期货、期权、次主连、历史K线或批量查询。
- 输入必填字符串、限制长度和正则，拒绝额外参数；代码大小写可宽容匹配目录，但返回和请求上游时恢复目录的规范大小写。格式校验不是存在性校验。

### 返回字段

| 字段 | 语义 |
| --- | --- |
| `symbol`, `contract_code`, `contract_name` | 规范化的东财完整标识、目录代码、合约名称。 |
| `commodity_code`, `commodity_name`, `exchange` | 经目标交易所目录核实的品种及交易所。 |
| `contract_type` | `main_continuous` 或 `delivery`；主连不推算对应的实际主力合约。 |
| `price` | 最新报价，有限 Decimal，JSON 十进制字符串；不是结算价或日收盘价。 |
| `currency`, `price_unit` | 仅在有可追溯规格证据时填写。当前目录和报价响应没有单位字段，未核验品种为 null，不能一律假定元/吨。 |
| `source_time` | 源 Unix 秒时间戳转换为带 Asia/Shanghai 时区的时间；缺失为 null。是源更新时间，不宣称最后成交时间或交易日。 |
| `source` | 固定 `eastmoney`。 |

`as_of` 继续表示服务器生成响应的 UTC 时间。工具说明明确报价可能延迟，非交易时段可能保持旧值，没有实时性 SLA；单位未核验时不得由下游自行假定。

## 已核实的数据源与读取流程

### 1. 目录确认（只查目标市场及品种）

`GET https://futsse-static.eastmoney.com/redis`

1. `msgid={market_id}`：商品数组。广期所实测包括 `mktid="225"`、`mktshort="GFEX"`、`vcode="lc"`、`vname="碳酸锂"`、`vtype="2"`。
2. 用目录 `vcode`＋允许的后缀精确匹配请求，不盲目截断尾字母；出现歧义时报源格式错误。
3. `msgid={market_id}_{vtype}`：该品种合约数组。碳酸锂实测含 `code="lc2705"`、`name="碳酸锂2705"`，以及 `code="lcm"`、`name="碳酸锂主连"`；目录同时有不在本期范围的 `lcs` 次主连。
4. 校验目录中的交易所、品种、代码与请求一致，按真实 `vtype` 请求，不依靠列表下标或连续编号（郑商所编号有缺口）。同键冲突不能任取一条。

不扫描全市场，不新增缓存，避免缓存主连映射过期；正常路径总共三次 GET，重试另计但共享调用预算。

### 2. 获取报价

`GET https://push2.eastmoney.com/api/qt/stock/get`

参数：`secid=<目录规范标识>`、`fltt=2`、`fields=f43,f57,f58,f59,f86`。

- `f43` 最新价、`f57` 合约代码、`f58` 名称、`f59` 小数精度、`f86` Unix 秒更新时间。
- 真实请求已成功：碳酸锂主连、碳酸锂具体合约、沪铜主连；样例价格为 88300、88260、104700（仅当时快照，不用于固定价格断言）。
- `fltt=2` 返回值不再除100，也不按 `f59` 二次量化。使用 JSON Decimal 解析保留源精度；对非零小数精度品种补一次实现期实测，防止仅整数样例掩盖缩放问题。
- 即使请求 `f13`，实测响应也未返回市场编号，因此市场身份依靠白名单＋目录精确成员核对，报价至少核对 `f57` 与目录代码一致。
- AkShare 的 `futures_zh_realtime` 实为新浪，不能调用后冒充东财。此次不引入 AkShare 依赖，其发布源码仅作为目录证据。

证据：[东财碳酸锂主连页面](https://quote.eastmoney.com/unify/r/225.lcm)、[AkShare 官方期货文档](https://akshare.akfamily.xyz/data/futures/futures.html)、[1.20.38 发布元数据](https://pypi.org/pypi/akshare/1.20.38/json)。发布源码 `futures_hist_em.py` 的目录读取与上述实测一致。广期所规格页面本次正文抓取失败，因此不能仅凭推测补齐币种和单位；实施阶段有可靠规格证据才增加最小单位映射，否则返回 null 并公开说明。

## 实现步骤与关键文件

1. **查询模型与东财 Adapter**
   - 新建 `src/queries/futures_types.py`：输入标识、内部目录记录、`CommodityPrice`，每个对外字段提供说明。
   - 新建 `src/queries/futures_sources.py`：目录定位和报价读取。复用 [source_requests.get](file:///F:/python%20project/FinanceKB/src/queries/source_requests.py#L199) 的超时、取消、重试和错误元数据；不复用股票或基本面私有 HTTP helper，避免引入错误 schema 和 Infoway 依赖。
   - 使用固定上游 URL 与 params，验证 JSON 根容器、必要字段、市场／合约身份、数字、时间戳。错误不暴露响应正文、token 或内部配置。
   - 新建 `src/queries/futures.py`：`get_commodity_price` 通过 `asyncio.to_thread` 调用 Adapter，沿用 QueryEnvelope 和 UTC `as_of`。

2. **独立 MCP 模块**
   - 新建 `src/mcp/futures/__init__.py`、`__main__.py`、`server.py`。
   - 复用 [现有服务装配方式](file:///F:/python%20project/FinanceKB/src/mcp/fundamentals/server.py#L20)、`StrictToolsMCP`、`build_mcp_app`、`run_server` 与 Bearer 鉴权，保留 readOnlyHint 和 outputSchema。
   - [src/queries/base.py:26](file:///F:/python%20project/FinanceKB/src/queries/base.py#L26) 仅为 domain 描述补充 futures，不改变其他域的运行时契约；公共服务说明去掉固定“五个”的数量描述。

3. **部署接线，不执行部署**
   - [docker-compose.yml:187](file:///F:/python%20project/FinanceKB/docker-compose.yml#L187)：新增 `mcp-futures`，继承 `x-mcp-common`，启动命令 `python -m src.mcp.futures`；Caddy `depends_on` 增加其健康检查依赖。
   - [docker-compose.local.yml:33](file:///F:/python%20project/FinanceKB/docker-compose.local.yml#L33)：新服务继承本地结构化 MCP 镜像与只读源码挂载。
   - [deploy/Caddyfile:20](file:///F:/python%20project/FinanceKB/deploy/Caddyfile#L20)：新增 `/mcp/futures` 及子路径到 `mcp-futures:8000` 的反代，保留原路由。
   - 复用现有共享密钥和网络，无新增 API key、宿主端口或权限；Dockerfile 已复制整个 src，无需修改。
   - 更新已有结构化数据源说明文档，记录新路径、标识示例、最新价/单位/连续合约边界；不覆盖历史验收结论。

## 错误与重试合同

- 类型、格式、市场不支持：`invalid_argument`，schema可判定的错误不请求上游。
- 正常、非空目录中无目标品种或合约：`invalid_argument`，提示当前目录未提供（可能已到期），不伪造替代合约。
- 目录请求失败、异常空目录、畸形 JSON、错合约、非法数字或时间戳：`SourceError(source="eastmoney")`，不得伪装成“无此合约”或无价格。
- 合约经确认且报价正常但最新价为 null／空字符串／`-`／`--`：`available=false, items=[]`，`reason="commodity_price_unavailable:<symbol>"`，不填0。
- 合法价格0保留；不人为截断负数。bool、NaN、Infinity、无法解析数值属于格式错误。
- 复用默认三次尝试、1/2秒加抖动、429 Retry-After 和总调用预算；永久错误、源格式错误、正常缺价不重试。没有降级源。

## 验证

1. 新建 `tests/unit/mcp/test_futures_contract.py`：覆盖五所目录、稀疏 vtype、大小写、主连/月合约、交易所/品种错配、次主连拒绝、股票/期权/金融期货拒绝、缺价、0/负数、超28位 Decimal、非有限值、时间戳、单位未知、错误脱敏。mock 响应带 request；全部离线。
2. 验证只查目标品种的三次请求、目录/报价均接入共享重试预算、429/503恢复与耗尽、403和格式错误不重试；失败目录不能误判合约不存在。
3. 更新 [工具注册测试](file:///F:/python%20project/FinanceKB/tests/unit/mcp/test_tool_registry.py#L13) 和 [HTTP集成测试](file:///F:/python%20project/FinanceKB/tests/integration/mcp/test_streamable_http.py#L30)：六个 server、16 个工具；包含临时本地 futures 服务的鉴权、initialize、tools/list、tools/call 和 structuredContent。
4. 收尾主营构成验证：现有三个测试文件的代理结果为140通过，已只读复核测试改动；补超过28位的减法／比例及全局 Decimal context 不变回归，再对“最新报告期 max→min”的内存变异运行测试，确认测试能捕捉错误，不改磁盘源代码。
5. 运行离线范围：`python -m pytest tests/unit/mcp tests/integration/mcp tests/unit/api/test_search_contract.py -q`。变更前基线为105通过、4项既有行情测试失败，分别记录，不把无关旧失败归于本次，也不声称全绿；类型检查既有两项错误不顺手重构。
6. `ruff`、`git diff --check`；Compose 使用 `--env-file .env.local` 做 `config --quiet` 验证，不打印展开后的密钥。Caddy配置用现有可用校验器检查，不重载在线服务。
7. 临时本地 futures 进程免费低频实测：LC主连、一个当时存在的LC月份合约、CU主连、一个已核实的小数报价品种，以及非法参数。核对源与输出，不固定断言会变动的价格。不启动付费采集流程、不改生产环境。

验收成功仅说明代码契约和当次源可用；长期可用率、交易所授权覆盖、下游 Research-Copilot allowlist 接入与实际部署不在本轮承诺内。
