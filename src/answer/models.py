"""答案核验的共用结构与汇总结果。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SupportingQuote(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    citation_index: int
    quote: str = Field(min_length=1)


class ClaimCheck(BaseModel):
    """模型只能输出明确声明及其证据，禁止额外字段和隐式类型转换。"""

    model_config = ConfigDict(extra="forbid", strict=True)
    claim: str = Field(min_length=1)
    citation_indices: list[int]
    status: Literal["supported", "unsupported", "unknown"]
    supporting_quote: str = ""
    supporting_quotes: list[SupportingQuote] | None = None
    reason: str
    item_id: str | None = None


class ClaimReport(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    checks: list[ClaimCheck] = Field(min_length=1)


def _validation_result(status: str, checks: list[dict], issues: list[str], total: int) -> dict:
    verified = sum(
        len(check["citation_indices"]) for check in checks if check["status"] == "supported"
    )
    score = None if status == "unknown" else (verified / total if total else 0.0)
    return {
        "status": status,
        "valid": status == "verified",
        "score": score,
        "total_citations": total,
        "verified_citations": verified,
        "checks": checks,
        "issues": issues,
    }
