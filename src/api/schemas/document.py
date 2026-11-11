"""文档管理相关的 Pydantic 请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class AsyncUploadResponse(BaseModel):
    task_id: str
    filename: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    filename: str
    result: dict | None = None
    error: str | None = None
    # 入库流水线阶段进度（parsed / chunked / embedded / written）
    progress: dict | None = None


class DocumentMetadataPatch(BaseModel):
    """文档元数据人工修正请求（只带需要改的字段；空串表示清空）。

    核心字段：date / version；研报字段：report_type / broker。
    """

    model_config = ConfigDict(extra="forbid")
    date: str | None = None
    version: str | None = None
    report_type: str | None = None
    broker: str | None = None
