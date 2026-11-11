"""聊天相关的 Pydantic 请求/响应模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from src.core.config import (
    CHAT_MAX_HISTORY_MESSAGE_CHARS,
    CHAT_MAX_HISTORY_TURNS,
    CHAT_MAX_QUERY_CHARS,
)
from src.retrieve.filter_policy import validate_explicit_filters


class ChatMessage(BaseModel):
    role: str = Field(..., description="user | assistant")
    content: str = Field(
        ...,
        max_length=CHAT_MAX_HISTORY_MESSAGE_CHARS,
        description=f"单条历史消息上限 {CHAT_MAX_HISTORY_MESSAGE_CHARS} 字符",
    )


class ChatRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        max_length=CHAT_MAX_QUERY_CHARS,
        description=f"用户问题上限 {CHAT_MAX_QUERY_CHARS} 字符（超出请拆分提问）",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        max_length=CHAT_MAX_HISTORY_TURNS,
        description=f"历史消息条数上限 {CHAT_MAX_HISTORY_TURNS}",
    )
    use_rerank: bool = True
    k: int | None = Field(
        default=None, ge=1, le=200,
        description=(
            "粗排召回深度；为空时用默认值 50。"
            "上限 200 是因为实际请求条数为 k×3（过采样），需防误用"
        ),
    )
    rerank_top_n: int | None = Field(
        default=None, ge=1, le=10,
        description="精排截断数：既是重排序返回条数，也是进入 LLM 上下文的证据块上限",
    )
    filters: dict | None = Field(default=None, description="元数据过滤条件")
    infer_filters: bool = Field(
        default=True,
        description=(
            "true 时由 LLM 根据问题自动推断过滤条件；"
            "false 时只用显式传入的 filters——前端清除过滤 chip 后靠它生效"
        ),
    )

    @field_validator("filters")
    @classmethod
    def check_filters(cls, value: dict | None) -> dict | None:
        return validate_explicit_filters(value)

    @field_validator("history")
    @classmethod
    def check_history_budget(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        """历史总长度也要有上限：只限条数时，单条超长消息仍会撑爆窗口。"""
        from src.core.config import CHAT_MAX_HISTORY_TOKENS
        from src.core.token_budget import estimate_tokens, fit_history

        if not value:
            return value
        kept, report = fit_history(
            [message.model_dump() for message in value],
            CHAT_MAX_HISTORY_TOKENS,
            max_turns=CHAT_MAX_HISTORY_TURNS,
        )
        if report["dropped"]:
            raise ValueError(
                f"历史对话过长（{len(value)} 条，超出约 {CHAT_MAX_HISTORY_TOKENS} token 预算），"
                "请精简或开启新会话"
            )
        if estimate_tokens("".join(message.content for message in value)) > CHAT_MAX_HISTORY_TOKENS:
            raise ValueError("历史对话内容过长，请精简后重试")
        return [ChatMessage(**message) for message in kept]


class SourceInfo(BaseModel):
    index: int
    title: str
    source: str
    category: str = ""
    collection: str = ""
    chunk: int | None = None
    score: float
    rerank_score: float | None = None
    preview: str
    # 证据定位：前端据此做页码跳转、元数据 chip 与「表已截断」提示
    date: str = ""
    block_type: str = "text"
    start_page: int = 0
    end_page: int = 0
    image_key: str = ""
    truncated: bool = False
    report_type: str = ""
    # 公开证据身份，不暴露用于生成回答的完整 content。
    id: str | int | None = None
    chunk_id: str | int | None = None
    child_id: str | int | None = None
    chunk_key: str | None = None
    parent_id: str | None = None
    parent_key: str | None = None
    covered_child_ids: list[str | int] = Field(default_factory=list)
    broker: str = ""
    table_id: str | None = None
    document_sha256: str | None = None
    document_version: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo] = []
    rewritten_query: str
    filters: dict = Field(default_factory=dict)
    citation_validation: dict | None = None
    answer_rejected: bool = Field(
        default=False, description="拒答策略触发（检索为空或引用校验低分）"
    )
    low_confidence: bool = Field(
        default=False, description="回答置信度偏低（引用校验分数低于阈值）"
    )
    answer_status: Literal["complete", "partial", "refused", "unverified", "clarification"] = "unverified"
    missing_items: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list, description="覆盖评估降级或未覆盖原因，供界面展示")
