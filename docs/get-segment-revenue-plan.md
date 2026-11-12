# get_segment_revenue 实施计划

## Context

业务取数缺少储能／主营构成收入和可追溯的毛利额。此次在现有 `mcp-fundamentals` 增加 `get_segment_revenue`，读取东财已披露的主营构成，让调用方获取同一报告期、同一分类维度的收入、成本与利润信息。

仅新增一个工具；不新建 MCP server，不处理碳酸锂价格，不做 PDF 抽取、RAG 改造或其他仓库修改。不部署云端、不自动提交／推送、不执行付费端到端验收、不修改冻结评测基线。保留工作区已有文档修改和未跟踪文件。

## 1. 数据源与已核实的限制

复用 `_F10_BASE`，调用已验证的站内接口：

`GET https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax?code=SZ300750`

- 参数为交易所前缀＋六位代码；响应 `zygcfx` 是多个报告期的记录数组，不依赖额外分页请求。
- `MAINOP_TYPE`：字符串 `1`＝行业、`2`＝产品、`3`＝地区；同一总收入的不同分解维度，禁止跨维度相加。
- 真实样例中，宁德时代 2026-06-30 产品项“储能电池系统”收入为 `53260967000` 元、成本为 `40500853000` 元、主营利润为 `12760114000` 元。
- 部分历史行只有收入，成本／利润／毛利率为空。报告日期分布不连续，不能承诺每个季度都有数据。
- 该接口没有正式版本化契约或完整性保证；“主营构成”不等同于法定报告分部。源未提供某细项，不等于公司原始报告没有披露。
- JSON 没有币种、公告日或单季／累计标志：不虚构这些信息，不把报告期末日期当公告日，不计算单季值。

证据：[东财实际响应](https://emweb.securities.eastmoney.com/PC_HSF10/BusinessAnalysis/PageAjax?code=SZ300750)、[AkShare 官方字段说明](https://akshare.akfamily.xyz/data/stock/stock_finance.html#stock-zygc-em)。AkShare 仅用于交叉核对字段及“元”单位，不引入该依赖。

## 2. 对外 Interface

`get_segment_revenue(symbol, report_date=None, dimension="product") -> QueryEnvelope[SegmentRevenue]`

| 参数 | 规则 |
| --- | --- |
| `symbol` | 必填，复用沪深 `StockSymbol` 格式与 `require_stock` 股票身份检查；指数不支持。 |
| `report_date` | 可选 `YYYY-MM-DD` 字符串；严格格式并校验真实日历日期。省略／null 时取整份源数据的最新报告期，再筛维度。显式日期必须精确匹配，不回退最近一期。 |
| `dimension` | `product / industry / region`，默认 `product`；不提供 `all`，避免混合维度被相加。 |

一次只返回一个报告期和一个维度的全部源记录，按 `segment_name` 稳定排序，不设静默截断的 limit、不增加关键词匹配参数。报告期不存在或该期缺少指定维度时，不偷偷返回旧数据。

`SegmentRevenue` 字段：

| 字段 | 来源／语义 |
| --- | --- |
| `symbol`, `report_date`, `dimension`, `segment_name` | `SECUCODE`、`REPORT_DATE`、`MAINOP_TYPE` 映射、原始 `ITEM_NAME`。 |
| `revenue` | `MAIN_BUSINESS_INCOME`，Decimal 或 null。 |
| `cost` | `MAIN_BUSINESS_COST`，Decimal 或 null。 |
| `gross_profit` | 优先 `MAIN_BUSINESS_RPOFIT`；源缺失时仅用同一行 `revenue - cost` 推导。上游称“主营利润”，不得解释为净利润。 |
| `gross_profit_basis` | `source / revenue_minus_cost / null`，标明直接来源或计算方式。 |
| `gross_margin_pct` | `GROSS_RPOFIT_RATIO × 100`；首版只取来源提供值，缺失不推导。 |
| `revenue_share_pct` | `MBI_RATIO × 100`；不重新计算分母，缺失为 null。 |
| `currency`, `unit`, `source` | 分别为 null、`元`、`eastmoney`；没有币种证据时不得硬标 CNY。 |

全部金额／比例使用有限 Decimal，JSON 输出十进制字符串；保留零、负值和源精度，禁止 bool、NaN、Infinity 或不可解析数字。`0.239577` 毛利率输出 `23.9577`，不使用二进制浮点中转。`null / "" / "-" / "--"` 作为缺失，不补零；不使用四舍五入毛利率反推成本。直接提供的利润不被计算值覆盖。

没有单列储能时只返回实际产品名称，不从“电池”“动力电池”等宽口径收入估算储能收入。每行缺少部分指标仍保留该行和 null，不因成本缺失丢弃已披露收入。

## 3. 实现与复用位置

1. **数据模型**：[src/queries/fundamentals_types.py:12](file:///F:/python%20project/FinanceKB/src/queries/fundamentals_types.py#L12)
   - 复用 `StockSymbol`，新增 `SegmentDimension`、`SegmentRevenue`。
   - 各字段提供准确说明，明确比例单位、缺失语义、毛利来源及报告日期口径；无需泛化成多源框架。

2. **单个东财 Adapter**：[src/queries/fundamentals_sources.py:298](file:///F:/python%20project/FinanceKB/src/queries/fundamentals_sources.py#L298)
   - 新增 `fetch_segment_revenue(symbol)`，复用 `_eastmoney`、现有 F10 常量和 Decimal JSON 解析，不新增 HTTP 客户端／重试层，不调用 Infoway。
   - 校验 `zygcfx` 必须为数组、每行必须为对象、证券代码一致、报告日期有效、分类可识别、名称为非空字符串、数值有限。
   - 缺少容器或畸形行属于源格式错误，不能当作正常空数组或跳过。
   - 以代码＋报告期＋维度＋名称识别重复：规范化数据完全相同才去重；相同键但数值冲突时报源错误，不任取最后一条。不要跨维度按名称去重，也不猜测 `RANK` 含义。
   - 网络错误保留既有 `SourceError` 的 code／retryable／attempt_count；新路径的格式错误统一归属 `source=eastmoney`，信息脱敏，不改公共错误契约。

3. **查询封装**：[src/queries/fundamentals.py:34](file:///F:/python%20project/FinanceKB/src/queries/fundamentals.py#L34)
   - 新增 async `get_segment_revenue` 并加入 `__all__`；复用 `_result`、`QueryEnvelope`、`asyncio.to_thread`。
   - 用户日期的语义校验在股票查询和源请求之前完成；其余类型／枚举／多余字段由现有严格 schema 验证。
   - 非空目标切片：`available=true`、`reason=""`；合法空结果用以下稳定 reason，并返回 `items=[]`：
     - `segment_revenue_unavailable`：整份源数据为空。
     - `segment_revenue_unavailable_for_report_date:YYYY-MM-DD`：指定期不存在。
     - `segment_revenue_unavailable_for_dimension:YYYY-MM-DD:product|industry|region`：目标期没有该维度。
   - 网络／解析异常始终通过 MCP 工具错误返回，不能转换成“源未提供”。

4. **注册与发现**：[src/mcp/fundamentals/server.py:13](file:///F:/python%20project/FinanceKB/src/mcp/fundamentals/server.py#L13)
   - 在既有注册列表增加工具，更新基本面 instructions，保留 readOnlyHint、类型化 outputSchema 和原路径。
   - 基本面工具由 4 个变 5 个，平台合计由 14 个变 15 个；服务数量仍为 5。
   - 不改 Caddy、Compose、环境变量、权限或下游 allowlist；下游如使用固定清单，需要另行接入此工具。

继续复用 [StrictToolsMCP](file:///F:/python%20project/FinanceKB/src/mcp/common/server.py#L36) 和 [source_requests.get / RetryPlan](file:///F:/python%20project/FinanceKB/src/queries/source_requests.py#L126)：默认最多三次尝试，退避 1／2 秒加抖动，429 尊重 Retry-After，受总调用预算约束。无数据不重试，格式错误不重试，不泄露响应正文或 token。

## 4. 验证与验收

### 离线测试

优先扩展现有文件，不新建测试框架：

- [tests/unit/mcp/test_fundamentals_contract.py:20](file:///F:/python%20project/FinanceKB/tests/unit/mcp/test_fundamentals_contract.py#L20)：通过 `StrictToolsMCP.call_tool` 验证查询契约，并在既有 `_eastmoney` seam 验证字段映射。
  - SH／SZ 代码、三种维度、源顺序打乱、最新报告期、精确日期匹配、不回退旧期；不能合并或相加跨维度同名数据。
  - 储能真实字段样例、没有单列储能、仅收入有值、全空容器、缺少报告期／维度。
  - Decimal 大数与小数精度、零收入／零成本、负数、占比换算、源利润优先、缺利润时差额推导；缺毛利率仍为 null。
  - 错证券代码、坏日期、未知分类、坏容器、重复一致／冲突、bool／非有限／乱码数值。
  - 必填、错误类型、指数、非法日期、错误枚举、多余字段被拒绝；可本地判定的参数错误不触发上游请求。
  - 通过 HTTP mock 验证新 Adapter 确实走共享重试：503 后成功、429 Retry-After、三次耗尽、403 一次失败、格式错误不重试；检查安全错误元数据。mock Response 必须有 request，禁止真实等待或联网。
- [tests/unit/mcp/test_tool_registry.py:12](file:///F:/python%20project/FinanceKB/tests/unit/mcp/test_tool_registry.py#L12)：添加新名称，核验域隔离、完整参数说明、additionalProperties=false、readOnlyHint 和 outputSchema。
- [tests/integration/mcp/test_streamable_http.py:110](file:///F:/python%20project/FinanceKB/tests/integration/mcp/test_streamable_http.py#L110)：复用临时本地服务／MCP ClientSession；新增基本面离线替身，验证 initialize → tools/list → tools/call、structuredContent 数字字符串及非法参数 isError。保留鉴权检查。

宿主当前已核实 `pydantic=2.14.2`、`mcp=1.29.0`、`httpx=0.28.1`、`pytest=9.1.1`。实施前先跑同一组变更前测试记录基线，再在实现后重跑：

```bash
python -m pytest tests/unit/mcp tests/integration/mcp tests/unit/api/test_search_contract.py -q
```

不运行全项目付费验收；既有行情测试仍有引用旧函数的情况，若基线失败，应与此次引入的问题分开报告，不顺手修改无关行情实现。依赖缺失时如实报告，不以编译通过代替 pytest。

### 免费真实取数冒烟

实现后用独立的临时本地 fundamentals 进程，不重启现有部署，不写真实密钥到日志：

1. 动态发现新工具，核验实际 schema。
2. 低频调用 `300750.SZ / 2026-06-30 / product`，对照东财响应中的“储能电池系统”行、金额与百分比。
3. 补一个沪市股票调用，以及显式不存在报告期的空结果检查；确认 SH／SZ 参数转换一致。
4. 非法日期／非法维度检查工具错误，并核验其他四个基本面工具仍在发现列表中。

单次冒烟只证明当时可取数，不声称长期稳定性；遇源限制不高频重试、不绕过限制。不运行 Research-Copilot 付费端到端流程。

框架依据：[Pydantic Decimal 官方说明](https://pydantic.dev/docs/validation/latest/api/pydantic/standard_library_types/#decimals) 明确 JSON 模式将 Decimal 序列化为字符串；沿用项目已有行为，不增加自定义 serializer。文档无法证明源币种／披露完整性，因此保持上述空值与覆盖限制。
