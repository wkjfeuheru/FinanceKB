"""ONNX INT8 量化嵌入器。

使用 optimum 导出 HuggingFace 模型为 ONNX，onnxruntime 动态 INT8 量化，
通过 InferenceSession 推理，实现 LangChain Embeddings 接口。
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path

import numpy as np
from numpy.linalg import norm

from finance_rag.src.core.config import (
    EMBED_BATCH_SIZE,
    EMBEDDING_DIM,
    EMBEDDING_MAX_TOKENS,
    EMBEDDING_MODEL,
    EMBEDDING_POOLING,
    MODEL_OFFLINE,
    ONNX_CACHE_DIR,
)
from finance_rag.src.core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class OnnxEmbedder:
    """ONNX INT8 量化嵌入器，实现 LangChain Embeddings 接口。

    首次调用时自动导出 ONNX 并量化，后续直接加载缓存模型。
    """

    _instances: dict[tuple[str, str], OnnxEmbedder] = {}
    _instances_lock = threading.Lock()
    _session_lock = threading.Lock()
    # 类级推理锁：session/tokenizer 为实例级单例，入库流水线的 Embedding 线程池
    # 会并发提交分批任务，这里串行化推理，避免 tokenizer 与 session 的并发使用问题
    # （与 BGEReranker._infer_lock 保持同一约定）。
    _infer_lock = threading.Lock()

    def __new__(
        cls,
        model_name: str = EMBEDDING_MODEL,
        pooling_method: str = EMBEDDING_POOLING,
    ):
        normalized_pooling = cls._normalize_pooling_method(pooling_method)
        key = (model_name, normalized_pooling)
        if key not in cls._instances:
            with cls._instances_lock:
                if key not in cls._instances:
                    cls._instances[key] = super().__new__(cls)
        return cls._instances[key]

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        pooling_method: str = EMBEDDING_POOLING,
    ):
        if hasattr(self, "_initialized"):
            return
        self._model_name = model_name
        self._pooling_method = self._normalize_pooling_method(pooling_method)
        self._cache_dir = Path(ONNX_CACHE_DIR)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = None
        self._tokenizer = None
        self._initialized = True

    @staticmethod
    def _normalize_pooling_method(pooling_method: str) -> str:
        normalized = pooling_method.strip().lower()
        if normalized not in {"cls", "mean"}:
            raise ValueError(
                "pooling_method must be one of: cls, mean; "
                f"got {pooling_method!r}"
            )
        return normalized

    def _get_session_and_tokenizer(self):
        if self._session is not None:
            return self._session, self._tokenizer

        with self._session_lock:
            if self._session is not None:
                return self._session, self._tokenizer
            return self._load_session_and_tokenizer()

    def _load_session_and_tokenizer(self):
        onnx_path = self._cache_dir / f"{self._model_name.replace('/', '_')}.onnx"
        quantized_path = self._cache_dir / f"{self._model_name.replace('/', '_')}_int8.onnx"

        if not quantized_path.exists():
            if MODEL_OFFLINE:
                raise FileNotFoundError(
                    f"离线模式下未找到 ONNX 缓存模型：{quantized_path}"
                )
            logger.info("首次导出并量化 ONNX 模型：%s", self._model_name)
            from onnxruntime.quantization import QuantType, quantize_dynamic
            from optimum.onnxruntime import ORTModelForFeatureExtraction
            from transformers import AutoTokenizer

            model = ORTModelForFeatureExtraction.from_pretrained(
                self._model_name, export=True
            )
            model.save_pretrained(str(onnx_path.parent))
            onnx_file = list(Path(onnx_path.parent).glob("model.onnx"))[0]

            quantize_dynamic(
                model_input=str(onnx_file),
                model_output=str(quantized_path),
                weight_type=QuantType.QInt8,
            )
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            logger.info("ONNX INT8 量化完成：%s", quantized_path)
        else:
            from transformers import AutoTokenizer

            logger.info("加载缓存的 ONNX INT8 模型：%s", quantized_path)
            # 模型已导出过，tokenizer 必然已在本地缓存，直接离线加载避免联网超时
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name, local_files_only=True
            )

        import onnxruntime as ort

        self._session = ort.InferenceSession(
            str(quantized_path),
            providers=["CPUExecutionProvider"],
        )
        return self._session, self._tokenizer

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        session, tokenizer = self._get_session_and_tokenizer()
        with self._infer_lock:
            encoded = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=EMBEDDING_MAX_TOKENS,
                return_tensors="np",
            )
            feed = {}
            for inp in session.get_inputs():
                if inp.name in encoded:
                    feed[inp.name] = encoded[inp.name]
            if not feed:
                feed[session.get_inputs()[0].name] = encoded["input_ids"]

            outputs = session.run(None, feed)
        last_hidden = outputs[0]

        attention_mask = encoded.get("attention_mask")
        if self._pooling_method == "cls":
            embeddings = last_hidden[:, 0, :]
        elif attention_mask is not None:
            mask_expanded = np.expand_dims(attention_mask, -1)
            sum_embeddings = np.sum(last_hidden * mask_expanded, axis=1)
            sum_mask = np.clip(np.sum(attention_mask, axis=1), 1e-9, None)
            embeddings = sum_embeddings / np.expand_dims(sum_mask, -1)
        else:
            embeddings = np.mean(last_hidden, axis=1)

        actual_dimension = int(embeddings.shape[-1])
        if actual_dimension != EMBEDDING_DIM:
            raise EmbeddingError(
                "Embedding dimension mismatch for "
                f"{self._model_name}: actual={actual_dimension}, expected={EMBEDDING_DIM}"
            )

        norms = norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.clip(norms, 1e-9, None)
        return normalized.tolist()

    def warmup(self) -> None:
        """加载模型并执行一次推理，确保服务就绪前模型可用。"""
        self.embed_query("预热请求")
        logger.info("ONNX 嵌入模型预热完成：%s", self._model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[i : i + EMBED_BATCH_SIZE]
            results.extend(self._embed_batch(batch))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._embed_batch([text])[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        """Ragas 走异步嵌入接口；推理本身仍是同步 ONNX。"""
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)
