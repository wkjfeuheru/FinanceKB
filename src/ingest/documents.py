"""文档上传/解析/管理模块（Agentic RAG）。

负责接收前端上传的文件，通过存储抽象层持久化文件，调用 :class:`KnowledgeBase`
完成切块/嵌入/入库，支持文档列表与删除。
上传统一走异步模式：立即返回 task_id，后台解析入库。
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import UploadFile

from src.core.config import (
    DOCUMENT_PARSE_WORKERS,
    INGEST_PIPELINE_ENABLED,
    KB_COLLECTION_NAME,
    MAX_UPLOAD_SIZE_MB,
    MINERU_SUPPORTED_EXTENSIONS,
    TENANT_ID,
    UPLOAD_DIR,
)
from src.storage.objects import get_storage
from src.storage.objects import object_keys
from src.index.milvus_kb import (
    KnowledgeBase,
    ParsedDocument,
    get_knowledge_base,
)
from src.chunking.public_metadata import resolve_public_metadata
from src.ingest.pipeline import prepare_chunked_document
from src.ingest.tasks import TaskStatus, get_task_manager

logger = logging.getLogger(__name__)

# 支持的文件格式
SUPPORTED_EXTENSIONS = set(MINERU_SUPPORTED_EXTENSIONS)

# 注：向量库写入的「同 collection 串行」由 KnowledgeBase 在资源侧自持写锁保证
# （见 milvus_kb._collection_write_lock），本模块不再维护自己的上传锁——
# 否则流水线的 asyncio 写锁与本模块的 threading 锁互不排斥，
# 对象存储事件回调会与流水线写入并发进入 add_parsed_document。


def _safe_delete(storage, key: str) -> None:
    """安全删除存储对象，忽略不存在的 key。"""
    try:
        storage.delete_sync(key)
    except Exception as exc:
        logger.warning("删除存储对象失败 key=%s: %s", key, exc)


def _list_keys_sync(storage, prefix: str) -> list[str]:
    """尽力列举某前缀下的对象键（后端不支持或失败时返回空列表）。

    宁可不删，也不删错：列举失败时调用方退化为「只删除可精确计算出的键」。
    """
    lister = getattr(storage, "list_keys_sync", None)
    if lister is None:
        return []
    try:
        return list(lister(prefix) or [])
    except Exception as exc:  # noqa: BLE001 - 列举失败不阻断删除主流程
        logger.warning("列举存储对象失败（prefix=%s）：%s", prefix, exc)
        return []


def _upload_temp_dir() -> Path:
    """上传暂存目录：``<UPLOAD_DIR>/.uploading``（不存在则创建）。

    刻意不用 ``tempfile.gettempdir()``：系统临时目录在受限环境里可能被拒绝访问，
    也可能被 OS/沙箱清理进程在「入队」与「解析」之间扫掉，导致解析阶段 ENOENT。
    """
    path = Path(UPLOAD_DIR) / ".uploading"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _unlink_temp_file(path: Path) -> None:
    """删除上传临时文件（不存在时静默忽略）。"""
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:  # pragma: no cover - 防御：清理失败不阻断入库
        logger.debug("上传临时文件清理失败 %s：%s", path, exc)


# ---------------------------------------------------------------------------
# 文档日期自动提取
# ---------------------------------------------------------------------------

def resolve_document_date(parsed: ParsedDocument) -> str:
    """兼容日期入口：只委托可信公共解析器，临时路径仅用于只读 PDF。"""
    markdown = getattr(parsed.chunks, "markdown", None)
    if not markdown:
        markdown = "\n".join(
            chunk.page_content for chunk in getattr(parsed.chunks, "chunks", ())[:5]
        )
    return resolve_public_metadata(parsed.source, markdown or "", pdf_path=parsed.path,
                                   explicit=parsed.metadata).date


class DocumentManager:
    """知识库文档管理器：上传 / 删除 / 列表。

    collection_name 决定该管理器绑定的 Milvus 集合（知识库），
    为空时使用默认知识库。
    """

    def __init__(self, kb: KnowledgeBase | None = None, collection_name: str = ""):
        self._kb = kb
        self._collection_name = collection_name
        self._storage = get_storage()
        self._parse_executor = ThreadPoolExecutor(
            max_workers=DOCUMENT_PARSE_WORKERS,
            thread_name_prefix="document-parser",
        )
        # 后台上传任务的强引用集合：done 后由回调自动丢弃，避免无限增长
        self._bg_tasks: set = set()
        # 流水线开关：关闭时回退到逐文件串行入库
        self._pipeline_enabled = INGEST_PIPELINE_ENABLED

    @property
    def kb(self) -> KnowledgeBase:
        if self._kb is None:
            self._kb = get_knowledge_base(self._collection_name or KB_COLLECTION_NAME)
        return self._kb

    @property
    def _collection(self) -> str:
        """本管理器绑定的集合名（对象键隔离的一段；空集合名退化为默认集合）。"""
        if self._collection_name:
            return self._collection_name
        return getattr(self._kb, "collection_name", "") or KB_COLLECTION_NAME

    @staticmethod
    def _document_version(parsed: ParsedDocument) -> str:
        """文档版本标签：与向量库写入、图片键使用同一套提取规则。"""
        from src.knowledge.document_version import extract_document_version

        return extract_document_version(parsed.source, parsed.metadata)

    def _parse_file(
        self,
        temp_path: Path,
        filename: str,
        ext: str,
        metadata: dict[str, Any] | None = None,
    ) -> ParsedDocument:
        """解析文件并返回 ParsedDocument（同步，供线程池调用）。

        MinerU 统一解析所有支持格式；文本格式由 adapter 做无损直读标准化。
        """
        kb = self.kb
        source = filename
        from src.chunking.metadata_extractor import resolve_report_title

        title = resolve_report_title(temp_path, filename)

        if ext not in MINERU_SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式 {ext}，仅支持 {', '.join(MINERU_SUPPORTED_EXTENSIONS)}"
            )
        # 所有格式统一进入 MinerU parser；文本格式由 adapter 做无损直读标准化。
        parsed = kb.parse_document(
            temp_path, source=source, title=title, metadata=metadata,
        )

        resolved = resolve_public_metadata(
            parsed.source, parsed.chunks.markdown or "", pdf_path=parsed.path,
            explicit=metadata,
        )
        merged_metadata = {**(parsed.metadata or {}), **resolved.as_metadata()}
        parsed = replace(parsed, metadata=merged_metadata)
        logger.info("公共元数据解析：%s -> date=%s broker=%s", filename,
                    resolved.date, resolved.broker)

        return parsed

    # -----------------------------------------------------------------------
    # 异步上传（后台解析入库）
    # -----------------------------------------------------------------------

    async def upload_document_async(
        self,
        file: UploadFile,
    ) -> dict[str, str]:
        """接收上传文件，保存后立即返回 task_id，后台异步解析入库。

        Returns:
            ``{"task_id": "...", "filename": "..."}``
        """
        raw_filename = file.filename or "untitled.md"
        filename = Path(raw_filename).name
        if filename in {"", ".", ".."}:
            raise ValueError("文件名无效")
        ext = Path(filename).suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式 {ext}，仅支持 {', '.join(SUPPORTED_EXTENSIONS)}"
            )

        content = await file.read()
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_UPLOAD_SIZE_MB:
            raise ValueError(
                f"文件大小 {size_mb:.1f}MB 超过限制 {MAX_UPLOAD_SIZE_MB}MB"
            )

        # 上传暂存文件放在**自己的存储目录**下，不用系统临时目录。
        # 系统临时目录在受限环境里不可靠（曾被拒绝访问），而且可能被 OS 清理进程
        # 或沙箱扫掉——实测出现过解析阶段报 ENOENT：文件在入队后被清走，
        # 表现为「单篇入库失败，报 .uploading.pdf 不存在」。放在 UPLOAD_DIR 下
        # 与真实产物同域，既可写也可排障（目录已被 .gitignore 覆盖）。
        temp_path = _upload_temp_dir() / f".{uuid.uuid4().hex}.uploading{ext}"
        await asyncio.to_thread(temp_path.write_bytes, content)

        # 预计算内容哈希，供后台任务增量检查使用
        content_hash = hashlib.sha256(content).hexdigest()

        # 创建任务并启动处理
        tm = get_task_manager()
        task = tm.create(filename=filename)
        tm.update(task.id, status=TaskStatus.PROCESSING)

        if self._pipeline_enabled:
            # 流式路径：**入队即返回**，不创建后台任务、不等同批其它文档。
            # 队列满 / 流水线已关闭时抛 IngestionRejectedError，由路由转换为 503。
            try:
                self._submit_to_pipeline(
                    task.id, temp_path, filename, ext, content, content_hash
                )
            except Exception as exc:
                from src.core.exceptions import friendly_message

                tm.update(task.id, status=TaskStatus.FAILED, error=friendly_message(exc))
                _unlink_temp_file(temp_path)
                logger.error("异步入库提交失败：%s (task_id=%s): %s", filename, task.id, exc)
                raise
        else:
            # 串行回退路径：结果 Future 由后台处理投递「成功统计」或「失败异常」
            loop = asyncio.get_running_loop()
            result_future: asyncio.Future = loop.create_future()
            bg_task = asyncio.create_task(
                self._process_upload_background(
                    task.id,
                    temp_path,
                    filename,
                    ext,
                    content,
                    content_hash,
                    result_future,
                )
            )
            # 持有强引用，便于观察/清理；done 后由回调自动丢弃，避免无限增长
            self._bg_tasks.add(bg_task)
            bg_task.add_done_callback(self._bg_tasks.discard)

        return {"task_id": task.id, "filename": filename}

    async def upload_documents_async(
        self,
        files: list[UploadFile],
    ) -> list[dict[str, str]]:
        """批量异步上传：每个文件立即返回 task_id，后台解析入库。

        Returns:
            ``[{"task_id", "filename"}, ...]``
        """
        results: list[dict[str, str]] = []
        for file in files:
            results.append(await self.upload_document_async(file))
        return results

    def _submit_to_pipeline(
        self,
        task_id: str,
        temp_path: Path,
        filename: str,
        ext: str,
        content: bytes,
        content_hash: str,
        result_future: asyncio.Future | None = None,
    ) -> dict[str, Any]:
        """把一份文档投递进常驻流水线（**同步、非阻塞、入队即返回**）。

        与 :meth:`_ingest_via_pipeline` 的区别：本方法不创建后台任务、不等待
        入库结束，投递成功即返回，因此调用方可以立刻响应上传请求；队列满时
        抛 :class:`~src.core.exceptions.IngestionQueueFullError`
        （路由层转 503），而不是阻塞在 ``await queue.put()`` 上。

        **增量检查在入队前完成**：重复文档不占用流水线容量。

        写库注入**两阶段**回调（``prepare_fn`` 锁外可并行 / ``commit_fn`` 锁内串行），
        使多篇文档的存储上传能并行，而向量库原子区仍保持同 collection 串行。

        Returns:
            命中增量跳过时返回终态结果字典（``skipped=True``），
            否则返回 ``{"task_id", "filename"}`` 表示已入队。
        """
        from src.ingest.pipeline import (
            PipelineItem,
            get_ingestion_pipeline,
        )

        tm = get_task_manager()

        # 增量构建：基于内容哈希检查是否已入库
        if self.kb.get_fingerprint_store().is_unchanged(filename, content_hash):
            result = {
                "filename": filename,
                "source": filename,
                "title": Path(filename).stem,
                "chunk_count": 0,
                "parent_count": 0,
                "skipped": True,
            }
            tm.update(task_id, status=TaskStatus.COMPLETED, result=result)
            _unlink_temp_file(temp_path)
            if result_future is not None and not result_future.done():
                result_future.set_result(result)
            logger.info("异步：文件未变更，跳过入库：%s (task_id=%s)", filename, task_id)
            return result

        def _finish(tid: str, res: dict[str, Any] | None, err: BaseException | None) -> None:
            if err is not None:
                from src.core.exceptions import friendly_message

                tm.update(tid, status=TaskStatus.FAILED, error=friendly_message(err))
                logger.error("异步入库失败：%s (task_id=%s): %s", filename, tid, err)
                return
            tm.update(tid, status=TaskStatus.COMPLETED, result=res)
            logger.info("异步入库完成：%s (task_id=%s)", filename, tid)

        def _on_progress(tid: str, progress: dict[str, Any]) -> None:
            tm.update(tid, progress=progress)

        item = PipelineItem(
            task_id=task_id,
            temp_path=temp_path,
            filename=filename,
            ext=ext,
            content=content,
            content_hash=content_hash,
            result_future=result_future,
            kb=self.kb,
            parse_fn=self._parse_file,
            # 两阶段写库：阶段 A（存储上传）锁外可并行，阶段 B（向量库原子区）锁内串行
            prepare_fn=self._prepare_parsed_document,
            commit_fn=self._commit_parsed_document,
            finish_fn=_finish,
            on_progress=_on_progress,
        )
        get_ingestion_pipeline().submit(item)
        return {"task_id": task_id, "filename": filename}

    async def _ingest_via_pipeline(
        self,
        task_id: str,
        temp_path: Path,
        filename: str,
        ext: str,
        content: bytes,
        content_hash: str,
        result_future: asyncio.Future,
    ) -> None:
        """流水线入库（**等待版**）：投递后等待本文件结算。

        生产上传路径走 :meth:`_submit_to_pipeline`「入队即返回」；
        本方法保留给「调用方需要等到入库结束」的场景与既有测试。
        """
        tm = get_task_manager()
        try:
            self._submit_to_pipeline(
                task_id,
                temp_path,
                filename,
                ext,
                content,
                content_hash,
                result_future,
            )
        except Exception as exc:
            from src.core.exceptions import friendly_message

            tm.update(task_id, status=TaskStatus.FAILED, error=friendly_message(exc))
            if not result_future.done():
                result_future.set_exception(exc)
            _unlink_temp_file(temp_path)
            logger.error("异步入库失败：%s (task_id=%s): %s", filename, task_id, exc)
            return

        # 失败已通过任务终态与 Future 双向上报，这里只负责「等到结算」
        if not result_future.done():
            with contextlib.suppress(Exception):
                await result_future

    async def _process_upload_background(
        self,
        task_id: str,
        temp_path: Path,
        filename: str,
        ext: str,
        content: bytes,
        content_hash: str,
        result_future: asyncio.Future | None = None,
    ) -> None:
        """逐文件串行入库（``INGEST_PIPELINE_ENABLED=false`` 时的回退路径）。

        处理流程与流水线一致（增量检查 → 解析 → 存储 → 写库），
        只是不与其他文档并行。
        """
        tm = get_task_manager()
        try:
            result = await self._process_upload_serially(
                task_id, temp_path, filename, ext, content, content_hash
            )
            if result_future is not None and not result_future.done():
                result_future.set_result(result)
            logger.info("异步入库完成：%s (task_id=%s)", filename, task_id)
        except Exception as exc:
            from src.core.exceptions import friendly_message

            tm.update(task_id, status=TaskStatus.FAILED, error=friendly_message(exc))
            if result_future is not None and not result_future.done():
                result_future.set_exception(exc)
            logger.error("异步入库失败：%s (task_id=%s): %s", filename, task_id, exc)
        finally:
            _unlink_temp_file(temp_path)

    async def _process_upload_serially(
        self,
        task_id: str,
        temp_path: Path,
        filename: str,
        ext: str,
        content: bytes,
        content_hash: str,
    ) -> dict[str, Any]:
        """回退路径的同步处理体：解析 + 持久化，返回任务结果字典。"""
        tm = get_task_manager()

        # 增量构建：基于内容哈希检查是否已入库
        if self.kb.get_fingerprint_store().is_unchanged(filename, content_hash):
            result = {
                "filename": filename,
                "source": filename,
                "title": Path(filename).stem,
                "chunk_count": 0,
                "parent_count": 0,
                "skipped": True,
            }
            tm.update(task_id, status=TaskStatus.COMPLETED, result=result)
            logger.info("异步：文件未变更，跳过入库：%s (task_id=%s)", filename, task_id)
            return result

        # 统一解析（MinerU 唯一路径）
        # 使用自定义 _parse_executor 以真正受 DOCUMENT_PARSE_WORKERS 限制并发
        loop = asyncio.get_running_loop()
        parsed = await loop.run_in_executor(
            self._parse_executor,
            self._parse_file,
            temp_path,
            filename,
            ext,
        )
        # 共享准备流程（切块 + 元数据/图片增强）：与流水线、对象事件入口一致，
        # 关闭流水线时元数据增强不再缺失，正文切块也只执行一次。
        parsed = await prepare_chunked_document(parsed, collection=self._collection)

        result = await self._persist_parsed_document(parsed, content_hash)
        tm.update(task_id, status=TaskStatus.COMPLETED, result=result)
        return result

    async def _persist_parsed_document(
        self,
        parsed: ParsedDocument,
        content_hash: str | None,
        dense_vectors: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """持久化已解析文档（**单回调兼容入口**）：阶段 A + 阶段 B 顺序执行。

        入库流水线的生产路径走**两阶段**（:meth:`_prepare_parsed_document` 锁外、
        :meth:`_commit_parsed_document` 锁内），以便存储上传与其他文档并行；
        本方法是二者的组合，供串行回退路径与既有调用方使用，语义与旧版一致。

        Args:
            parsed: 解析 + 切块结果。
            content_hash: 上传内容哈希（透传指纹存储，避免增量去重失效）。
            dense_vectors: 流水线在 Embedding 线程池中预计算的稠密向量；
                None 时由 :meth:`KnowledgeBase.add_parsed_document` 内联补齐。
        """
        context = await self._prepare_parsed_document(parsed, content_hash, dense_vectors)
        return await self._commit_parsed_document(context)

    async def _prepare_parsed_document(
        self,
        parsed: ParsedDocument,
        content_hash: str | None,
        dense_vectors: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """写库**阶段 A（锁外，可与其他文档并行）**：文档产物落存储后端。

        原文件、解析提取的嵌入图片、解析后的 .md 依次上传。这一整段是网络 IO
        （本地文件系统或 OSS/S3），不持有写锁，因此多篇文档可以并行上传。

        对象键按「租户 / 集合 / 文档身份 / 版本 / 分区」构造
        （见 :mod:`src.storage.objects.object_keys`）：
        原文件与解析产物分区存放，同名不同扩展名、不同集合、不同版本互不覆盖。

        Returns:
            提交上下文，原样交给 :meth:`_commit_parsed_document`。
        """
        storage = self._storage
        filename = parsed.source
        version = self._document_version(parsed)

        # 上传原始文件到存储后端（原文分区，保留扩展名）
        original_key = object_keys.original_key(
            self._collection, filename, version=version, filename=filename
        )
        await storage.upload(original_key, parsed.path.read_bytes())

        # 上传解析提取的嵌入图片到存储后端，并清理临时目录
        await self._upload_extracted_images(parsed, storage, version, self._collection)

        # 解析后的 .md 文件直接上传到存储后端（解析产物分区，永不与原文同名冲突）
        if parsed.chunks.markdown is not None:
            md_key = object_keys.parsed_key(self._collection, filename, version=version)
            await storage.upload(md_key, parsed.chunks.markdown.encode("utf-8"))

        return {
            "parsed": parsed,
            "content_hash": content_hash,
            "dense_vectors": dense_vectors,
        }

    async def _commit_parsed_document(self, context: dict[str, Any]) -> dict[str, Any]:
        """写库**阶段 B（写锁内，同 collection 串行）**：向量库原子区 + 指纹。

        由入库流水线在写锁内调用。这一段必须串行——
        :meth:`KnowledgeBase.add_parsed_document` 的文档间 SimHash 去重是
        "先查再插"，并发会让近似重复文档同时通过检查；同 source 的
        「删旧行 + 插新行」并发也会互相删除。

        Args:
            context: :meth:`_prepare_parsed_document` 的返回值。
        """
        parsed = context["parsed"]
        filename = parsed.source
        stored = await asyncio.to_thread(
            self.kb.add_parsed_document,
            parsed,
            None,
            content_hash=context.get("content_hash"),
            dense_vectors=context.get("dense_vectors"),
        )
        return {
            "filename": filename,
            "source": stored["source"],
            "title": stored["title"],
            "chunk_count": stored["chunk_count"],
            "parent_count": stored["parent_count"],
        }

    def _store_parsed_document(
        self,
        parsed: ParsedDocument,
        content_hash: str | None = None,
        dense_vectors: list[list[float]] | None = None,
    ) -> dict[str, Any]:
        """同步写库（对象存储事件路径）：直接调用 ``add_parsed_document``。

        ``content_hash`` 传入时透传给指纹存储，避免增量去重失效。
        ``dense_vectors`` 由调用方预计算时传入，否则内联嵌入。

        无需在此加锁：``KnowledgeBase.add_parsed_document`` 内部持有
        集合级写锁，与入库流水线的写入阶段共用同一把锁。
        """
        return self.kb.add_parsed_document(
            parsed, content_hash=content_hash, dense_vectors=dense_vectors
        )

    @staticmethod
    async def _upload_extracted_images(
        parsed: ParsedDocument, storage: Any, version: str = "", collection: str = ""
    ) -> None:
        """把解析阶段提取的嵌入图片上传到存储后端。

        图片来自 ``parsed.chunks.images``（``ImageAsset``，已在内存中，无临时文件）；
        单张失败仅告警不阻断入库。整体文本描述由视觉模型在入库链路上生成，
        这里只负责把原始图片落到对象存储，供页码溯源与「查看原图」使用。

        键必须与切块阶段写入向量库的 ``image_key`` 完全一致（同一 ``collection``
        与版本标签），否则前端「查看原图」会取不到对象；``collection`` 为空时
        退回历史键（兼容旧调用方与旧对象）。
        """
        images = getattr(parsed.chunks, "images", None) or []
        if not images:
            return

        from src.ingest.pdf_assets import image_object_key

        uploaded = 0
        for asset in images:
            key = image_object_key(
                parsed.source, asset, collection=collection, version=version
            )
            try:
                await storage.upload(key, asset.data)
                uploaded += 1
            except Exception as exc:
                logger.warning("嵌入图片上传失败 %s: %s", key, exc)

        if uploaded:
            logger.info("嵌入图片上传完成：%s（%d 张）", parsed.source, uploaded)

    def update_document_metadata(self, source: str, metadata: dict[str, Any]) -> dict[str, Any]:
        """人工修正某篇文档的元数据。

        读取、合并与完整校验由 KB 在集合写锁内完成，避免并发修改丢失。
        """
        return self.kb.update_document_metadata(source, metadata)

    async def process_object_event(
        self, temp_path: Path, filename: str, version_id: str
    ) -> dict[str, Any]:
        """Process an already downloaded object and persist its version metadata.

        与上传入口共用同一条准备流程（解析 → 切块 → 增强），因此对象事件
        入库的文档同样带有完整的规范化元数据与图片块，不会因缺少 report_type
        而在写入时失败。
        """
        try:
            content_hash = await asyncio.to_thread(
                lambda: hashlib.sha256(temp_path.read_bytes()).hexdigest()
            )
            parsed = await asyncio.to_thread(
                self._parse_file, temp_path, filename, temp_path.suffix.lower()
            )
            parsed = await prepare_chunked_document(parsed, collection=self._collection)
            result = await asyncio.to_thread(
                self._store_parsed_document, parsed, content_hash
            )
            from src.storage.relational_db.document_manifest import DocumentManifestRepository
            repo = DocumentManifestRepository()
            repo.upsert_document({
                "document_id": f"{TENANT_ID}:{filename}", "tenant_id": TENANT_ID,
                "source": filename, "title": parsed.title,
                "current_version_id": version_id, "status": "active",
                "category": "",
                "date": str((parsed.metadata or {}).get("date", "")),
                "object_key": object_keys.original_key(
                    self._collection, filename, filename=filename,
                    version=self._document_version(parsed),
                ),
            })
            repo.create_version({
                "version_id": version_id, "document_id": f"{TENANT_ID}:{filename}",
                "file_sha256": content_hash,
                "status": "current", "parser_version": "mineru",
                "chunk_count": result["chunk_count"],
            })
            return result
        finally:
            temp_path.unlink(missing_ok=True)

    def soft_delete_source(self, source: str) -> dict[str, Any]:
        """Mark document metadata deleted and exclude its vectors from normal use."""
        from src.storage.relational_db.document_manifest import DocumentManifestRepository
        repo = DocumentManifestRepository()
        document_id = f"{TENANT_ID}:{source}"
        existing = repo.get_document(document_id)
        if existing:
            repo.upsert_document({"document_id": document_id, "status": "deleted", "deleted_at": datetime.now()})
        return self.kb.soft_delete_document(source)

    def delete_document(self, source: str, version: str | None = None) -> dict[str, Any]:
        """删除文档的向量记录，并按引用关系回收对象存储资产（RAG-07）。

        顺序是刻意设计的：

        1. **先删索引**（向量 + 派生行 + 指纹）——索引删除失败时不碰任何资产，
           当前可见版本仍可检索、预览与跳转页码；
        2. 索引删除成功后才回收资产，且**只回收被删版本命名空间下的键**：
           历史版本的删除不会带走当前版本的原文件与解析产物；
        3. 删除整篇文档时按前缀列举回收（含旧协议的 ``docs/{filename}`` 等键），
           列举失败只告警——宁可不删，也不删错。

        ``version=None`` 时删除该文档的全部版本；指定版本时仅删除该版本。
        """
        storage = self._storage

        # 1) 先删索引：失败会直接抛出，资产保持可用
        result = self.kb.remove_document(source, version=version)

        # 2) 再回收资产
        removed_versions = self._collect_removed_versions(source, version)
        try:
            if version:
                for label in removed_versions:
                    self._purge_version_assets(storage, source, label)
            else:
                self._purge_document_assets(storage, source)
        except Exception as exc:  # noqa: BLE001 - 回收失败可重试，不影响索引状态
            logger.warning("文档资产回收失败（source=%s）：%s", source, exc)

        return result

    def _collect_removed_versions(self, source: str, version: str | None) -> list[str]:
        """本次被删除的版本标签（用于资产回收）。

        版本标签直接来自调用方：向量行的 ``version`` 字段就是写入时的同一标签，
        资产键的版本段由 :func:`object_keys.version_segment` 从同一标签推导，
        因此回收不需要读取指纹（指纹只保留最近一次发布的版本，读它会漏掉历史版本）。
        """
        return [version] if version else []

    def _purge_version_assets(self, storage: Any, source: str, version: str) -> None:
        """回收某一版本命名空间下的原文件与解析产物（不动图片，避免误删共享资产）。"""
        keys = [
            object_keys.original_key(
                self._collection, source, version=version, filename=source
            ),
            object_keys.parsed_key(self._collection, source, version=version),
            object_keys.original_key(self._collection, source, version=version),
        ]
        for key in dict.fromkeys(keys):
            _safe_delete(storage, key)

    def _purge_document_assets(self, storage: Any, source: str) -> None:
        """回收整篇文档的全部资产：新命名空间按前缀列举，旧协议键按名计算。"""
        prefix = object_keys.document_root_prefix(self._collection, source)
        listed = _list_keys_sync(storage, prefix)
        if listed:
            for key in listed:
                _safe_delete(storage, key)
        else:
            # 列举不可用（后端不支持 / 对象不存在）：至少清掉可精确计算的键
            for key in (
                object_keys.original_key(self._collection, source, filename=source),
                object_keys.parsed_key(self._collection, source),
            ):
                _safe_delete(storage, key)

        # 旧协议键（历史对象）单独清理，保证升级前后都能回收干净
        for legacy_key in (
            object_keys.legacy_original_key(source),
            object_keys.legacy_parsed_key(source),
        ):
            _safe_delete(storage, legacy_key)

    def list_documents(self, include_versions: bool = False) -> list[dict[str, Any]]:
        """列出知识库所有文档；``include_versions=True`` 时含历史版本明细。"""
        return self.kb.list_documents(include_versions=include_versions)

    def original_key_candidates(self, source: str) -> list[str]:
        """原文件读取候选键：当前可见版本 → 覆盖模式键 → 旧协议键。

        读取端（页码预览等）据此逐级回退，无需知道对象是哪个版本写入的，
        也不要求历史对象先迁移（RAG-06 兼容要求）。
        """
        return object_keys.original_key_candidates(
            self._collection, source, version=self._visible_version(source)
        )

    def _visible_version(self, source: str) -> str:
        """当前可见版本的标签：取自指纹记录（最近一次成功发布）。"""
        try:
            record = self.kb.get_fingerprint_store().get(source) or {}
        except Exception as exc:  # noqa: BLE001 - 指纹不可读时退化为覆盖模式键
            logger.debug("读取指纹失败（source=%s）：%s", source, exc)
            return ""
        return str(record.get("version") or "")

    def get_stats(self) -> dict[str, Any]:
        """知识库统计。"""
        return self.kb.get_stats()


# ---------------------------------------------------------------------------
# 单例（按知识库集合缓存）
# ---------------------------------------------------------------------------

_DM_INSTANCES: dict[str, DocumentManager] = {}
_DM_LOCK = threading.Lock()


def get_document_manager(collection: str = "") -> DocumentManager:
    """获取（并缓存）指定知识库的 DocumentManager 单例（线程安全）。

    collection 为空时返回默认知识库的管理器。
    """
    name = collection or KB_COLLECTION_NAME
    dm = _DM_INSTANCES.get(name)
    if dm is None:
        with _DM_LOCK:
            dm = _DM_INSTANCES.get(name)
            if dm is None:
                dm = DocumentManager(collection_name=name)
                _DM_INSTANCES[name] = dm
    return dm
