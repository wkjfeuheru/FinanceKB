"""商品期货目录与最新报价契约。"""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal

from pydantic import BaseModel, Field

SYMBOL_PATTERN = r"^(113|114|115|142|225)\.[A-Za-z]{1,3}(?:[mM]|[0-9]{3,4})$"
FuturesSymbol = Annotated[
    str,
    Field(
        pattern=SYMBOL_PATTERN,
        max_length=16,
        description="东财商品期货标识，如225.lcm碳酸锂主连、225.lc2705月份合约、113.cum沪铜主连；仅支持113/114/115/142/225五所，代码按当前目录确认，大小写不敏感。",
    ),
]


@dataclass(frozen=True)
class CatalogCommodity:
    """目录品种记录：mktid 与 mktshort 已按目标市场校验。"""

    vcode: str
    vname: str
    vtype: str


@dataclass(frozen=True)
class CatalogContract:
    """目录合约记录：vcode 与 vname 已与所属品种核实。"""

    code: str
    name: str


class FuturesContract(BaseModel):
    symbol: FuturesSymbol
    contract_code: str = Field(description="东财目录规范合约代码；郑商所等市场保留源大小写。")
    contract_name: str = Field(description="目录合约名称，主连不等于具体交割月份。")
    commodity_code: str = Field(description="目标交易所目录中的商品代码。")
    commodity_name: str = Field(description="目标交易所目录中的商品名称。")
    exchange: Literal["SHFE", "DCE", "CZCE", "INE", "GFEX"] = Field(description="经目录核实的商品期货交易所。")
    contract_type: Literal["main_continuous", "delivery"] = Field(description="main_continuous为主连展示报价，delivery为具体月份合约；不推算主连对应的交割合约。")


class CommodityPrice(FuturesContract):
    price: Decimal = Field(allow_inf_nan=False, description="东财最新报价，Decimal十进制字符串；fltt=2原值，不二次缩放，不等于现货价、结算价或日收盘价。")
    currency: str | None = Field(default=None, description="有规格证据才填写币种；源快照未提供，未核验时为null，不可自行推断。")
    price_unit: str | None = Field(default=None, description="有规格证据才填写计价单位；未核验时为null，不能一律视为元/吨。")
    source_time: datetime | None = Field(default=None, description="源更新时间，Asia/Shanghai时区；缺失为null，不等于最后成交时间或交易日，不保证实时性。")
    source: Literal["eastmoney"] = Field(default="eastmoney", description="东财期货目录及报价，无降级源。")
