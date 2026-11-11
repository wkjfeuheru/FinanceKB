"""文档管理 API 路由（上传 / 列表 / 删除 / 知识库统计 / 任务状态）。"""

from __future__ import annotations

import logging
from typing import Annotated

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Response,
    UploadFile,
)

from src.core.config import PAGE_RENDER_DPI, POLICY_COLLECTION_NAME, REPORT_COLLECTION_NAME
from src.core.exceptions import IngestionRejectedError
from src.storage.objects import get_storage
from src.utils.audit import log as audit_log
from src.utils.metrics import document_ops
from src.api.dependencies import get_current_user
from src.ingest.documents import get_document_manager
from src.ingest.tasks import get_task_manager
from src.api.schemas.document import (
    AsyncUploadResponse,
    DocumentMetadataPatch,
    TaskStatusResponse,
)

logger = logging.getLogger(__name__)

router = APIRouter()

_KNOWLEDGE_BASES = frozenset({REPORT_COLLECTION_NAME, POLICY_COLLECTION_NAME})


def _resolve_kb(kb: str) -> str:
    """把请求里的 kb 限定在白名单集合内；空值取默认（研报）集合。"""
    # 直接调用路由函数（测试/内部）时 Query 默认值尚未被 FastAPI 解析
    name = kb.strip() if isinstance(kb, str) else ""
    name = name or REPORT_COLLECTION_NAME
    if name not in _KNOWLEDGE_BASES:
        raise HTTPException(status_code=400, detail=f"未知知识库：{name}")
    return name


# --- 文档批量上传（异步：立即返回 task_id，后台解析入库） ---


async def _submit_upload_batch(
    files: list[UploadFile], kb: str
) -> list[AsyncUploadResponse]:
    """两条上传路由的共享处理：批量提交 + 统一异常映射 + 文件关闭。

    语义（两条路由完全一致）：

    * 任一文件失败即中止整批；已入队的文件保持入队，客户端重试时
      会命中内容指纹被增量跳过，不会重复入库；
    * ``ValueError``（非法扩展名/超大文件）→ 400；流水线背压
      （:class:`IngestionRejectedError`）→ 503；其余异常 → 500；
    * 无论成败都在 ``finally`` 中关闭全部上传句柄。
    """
    dm = get_document_manager(_resolve_kb(kb))
    if not files:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    try:
        results = await dm.upload_documents_async(files)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except IngestionRejectedError as exc:
        # 入库流水线背压（队列满）或正在关闭：属于「暂时不可用」，让客户端稍后重试。
        logger.warning("异步上传被流水线拒绝：%s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("异步上传提交失败：%s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for file in files:
            await file.close()
    return [AsyncUploadResponse(**item) for item in results]


@router.post("/documents/upload", response_model=list[AsyncUploadResponse])
async def upload_documents(
    current_user: Annotated[str, Depends(get_current_user)],
    files: list[UploadFile] = File(...),
    kb: str = Form(""),
):
    """上传文档到知识库（支持 .md / .txt / .pdf）。

    异步处理：立即返回 task_id 列表，前端轮询 /tasks/{task_id} 查询进度。
    kb 为目标知识库集合名（空 = 默认研报知识库）。
    """
    results = await _submit_upload_batch(files, kb)
    document_ops.labels(operation="upload").inc(len(results))
    audit_log(
        "upload",
        user=current_user,
        resource=",".join(r.filename for r in results),
        detail=f"异步提交 {len(results)} 个文件",
    )
    return results


# --- 异步上传（兼容别名，与 /documents/upload 行为一致） ---

@router.post("/documents/upload-async", response_model=list[AsyncUploadResponse])
async def upload_documents_async(
    current_user: Annotated[str, Depends(get_current_user)],
    files: list[UploadFile] = File(...),
    kb: str = Form(""),
):
    """异步上传文档：立即返回 task_id，后台解析入库，前端轮询状态。

    兼容别名：业务处理、批量中止、异常映射与文件关闭语义与
    ``/documents/upload`` 完全一致，仅审计标签不同。
    """
    results = await _submit_upload_batch(files, kb)
    audit_log(
        "upload_async",
        user=current_user,
        resource=",".join(r.filename for r in results),
        detail=f"异步提交 {len(results)} 个文件",
    )
    return results


# --- 任务状态查询 ---

@router.get("/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    current_user: Annotated[str, Depends(get_current_user)],
):
    """查询异步上传任务的状态。"""
    tm = get_task_manager()
    task = tm.get(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="任务不存在")
    return TaskStatusResponse(
        task_id=task.id,
        status=task.status.value,
        filename=task.filename,
        result=task.result,
        error=task.error,
        progress=task.progress,
    )


# --- 文档列表 ---

@router.get("/documents")
async def list_documents(
    current_user: Annotated[str, Depends(get_current_user)],
    kb: str = Query(""),
    include_versions: bool = Query(
        False, description="true 时返回各文档的历史版本明细"
    ),
):
    """列出指定知识库所有文档（kb 为空 = 默认知识库）。"""
    dm = get_document_manager(_resolve_kb(kb))
    return dm.list_documents(include_versions=include_versions)


# --- 文档删除 ---

@router.delete("/documents/{source:path}")
async def delete_document(
    source: str,
    current_user: Annotated[str, Depends(get_current_user)],
    kb: str = Query(""),
    version: str = Query("", description="指定删除的版本号；空 = 删除全部版本"),
):
    """删除指定文档的 Milvus 向量记录，保留本地原文件。

    ``version`` 为空时删除该文档的所有版本；指定时仅删除对应版本。
    """
    dm = get_document_manager(_resolve_kb(kb))
    try:
        result = dm.delete_document(source, version=version or None)
        document_ops.labels(operation="delete").inc()
        audit_log("delete", user=current_user, resource=source)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return result


# --- 研报元数据人工修正 ---

@router.patch("/documents/{source:path}/metadata")
async def patch_document_metadata(
    source: str,
    payload: DocumentMetadataPatch,
    current_user: Annotated[str, Depends(get_current_user)],
    kb: str = Query(""),
):
    """人工修正研报元数据（不重新嵌入）。

    抽取链路的最后一道兜底：正则与模型都会抽错，抽错又没有任何补救手段，
    元数据维度就形同虚设。取值非法返回 422 而不是静默写空。
    """
    dm = get_document_manager(_resolve_kb(kb))
    fields = payload.model_dump(exclude_unset=True)
    if not fields:
        raise HTTPException(status_code=400, detail="未提供任何待更新字段")
    try:
        result = dm.update_document_metadata(source, fields)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("研报元数据更新失败：%s", exc)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result.get("missing"):
        raise HTTPException(status_code=404, detail=f"未找到文档：{source}")
    audit_log(
        "update_metadata",
        user=current_user,
        resource=source,
        detail=f"字段：{', '.join(sorted(fields))}",
    )
    return result


# --- 知识库统计 ---
@router.get("/kb/stats")
async def kb_stats(
    current_user: Annotated[str, Depends(get_current_user)],
    kb: str = Query(""),
):
    """知识库统计信息（文档数/块数/集合状态）。"""
    dm = get_document_manager(_resolve_kb(kb))
    return dm.get_stats()


# --- 原文页渲染（页码级溯源） ---

@router.get("/documents/{source:path}/page/{page}")
async def render_document_page(
    source: str,
    page: int,
    current_user: Annotated[str, Depends(get_current_user)],
    kb: str = Query("", description="文档所属知识库集合名；空 = 默认知识库"),
):
    """把原文 PDF 的印刷第 ``page`` 页渲染成 PNG。

    分析师验证一条结论的成本必须足够低：拿到页码就能直接看到那一页。
    ``page`` 与切块 ``start_page`` 相同，是页面上印出来的页码；封面/目录
    无印刷页码时不会走到这里。检测失败时印刷页等于物理页，行为与原来一致。

    对象键按「租户 / 集合 / 文档身份 / 版本」构造（RAG-06），读取时逐级回退到
    历史键，因此升级前的对象无需迁移也能继续预览。
    """
    if page < 1:
        raise HTTPException(status_code=400, detail="页码从 1 开始")

    dm = get_document_manager(_resolve_kb(kb))
    storage = get_storage()
    payload = None
    for key in dm.original_key_candidates(source):
        try:
            payload = await storage.download(key)
            break
        except Exception as exc:  # noqa: BLE001 - 换下一个候选键继续找
            logger.debug("读取原文候选键失败（%s）：%s", key, exc)
    if payload is None:
        logger.warning("读取原文失败（%s，kb=%s）：所有候选键均不可用", source, kb or "default")
        raise HTTPException(status_code=404, detail=f"未找到原文：{source}")

    try:
        import pymupdf

        document = pymupdf.open(stream=payload, filetype="pdf")
    except Exception as exc:  # noqa: BLE001 - 非 PDF（如 .md）不能渲染
        logger.info("原文不是可渲染的 PDF（%s）：%s", source, exc)
        raise HTTPException(status_code=400, detail="原文不是可渲染的 PDF") from exc

    try:
        from src.ingest.page_labels import detect_printed_pages, printed_to_physical

        page_map = detect_printed_pages(document)
        physical = printed_to_physical(page, page_map)
        if physical is None or physical > document.page_count:
            raise HTTPException(
                status_code=404,
                detail=f"页码超出范围（共 {document.page_count} 页）",
            )
        pixmap = document[physical - 1].get_pixmap(dpi=PAGE_RENDER_DPI)
        return Response(content=pixmap.tobytes("png"), media_type="image/png")
    finally:
        document.close()
