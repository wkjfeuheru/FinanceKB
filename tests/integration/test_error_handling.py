"""断崖检测 + 指纹存储 + 上下文组装 独立测试（不依赖 Milvus）。"""

import hashlib
import json
import os
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# 断崖检测：直接测**生产实现**
# ---------------------------------------------------------------------------
# 这里曾放一份手抄的 `_apply_cliff_detection` 副本（理由是"避免导入 Milvus 依赖"），
# 结果是这些用例只验证那份副本：生产逻辑怎么改它们都照样通过，等于没有回归保护。
# 该函数是纯函数，导入 hybrid_retriever 并不需要可用的 Milvus 连接。
from finance_rag.src.rag.retrieval.hybrid_retriever import (  # noqa: E402
    _apply_cliff_detection,
)


class FingerprintStore:
    def __init__(self, store_path):
        self._path = Path(store_path)
        self._data = {}
        self._load()

    def _load(self):
        if self._path.exists():
            try:
                self._data = json.loads(self._path.read_text("utf-8"))
            except (json.JSONDecodeError, OSError):
                self._data = {}

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, ensure_ascii=False, indent=2), "utf-8"
        )

    def is_unchanged(self, source, file_hash):
        r = self._data.get(source)
        return r is not None and r.get("hash") == file_hash

    def update(self, source, file_hash, file_path, chunk_count):
        import time
        self._data[source] = {
            "hash": file_hash,
            "file_path": file_path,
            "chunk_count": chunk_count,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        }
        self._save()

    def remove(self, source):
        self._data.pop(source, None)
        self._save()

    @staticmethod
    def hash_file(file_path):
        sha = hashlib.sha256()
        with open(file_path, "rb") as f:
            while chunk := f.read(8192):
                sha.update(chunk)
        return sha.hexdigest()


def build_context(docs):
    if not docs:
        return "（未检索到相关文档）", []
    chunks = []
    sources = []
    for i, doc in enumerate(docs, start=1):
        content = (doc.get("content") or "")[:2000]
        title = doc.get("title", "文档")
        chunks.append(f"[{i}] {title}\n{content}")
        sources.append({
            "index": i, "title": title,
            "source": doc.get("source", ""), "chunk": doc.get("chunk"),
            "score": round(doc.get("score", 0.0), 4),
            "preview": (doc.get("content") or "")[:200],
        })
    return "\n\n---\n\n".join(chunks), sources


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCliffDetection:
    """断崖三条约束：绝对差距 / 相对跌幅 / 兜底下限。

    分数尺度的事实决定了各规则的作用面——稠密/COSINE 分在 [-1, 1]，0.8 才可达；
    RRF 融合分上界约 2/61≈0.033，绝对规则永不可达。因此多数用例用相对跌幅构造。
    """

    def test_no_cliff(self):
        # 最大相邻差距 0.05（远低于 0.8），最大跌幅 5.3%（低于 40%）
        c = [{"score": 0.95}, {"score": 0.90}, {"score": 0.88}, {"score": 0.85}]
        assert len(_apply_cliff_detection(c, min_top_k=1)) == 4

    def test_relative_rule_triggers(self):
        # i=2：0.90 → 0.30，跌幅 66.7% >= 40%（差距 0.60 < 0.8，只有相对规则命中）
        c = [{"score": 0.95}, {"score": 0.92}, {"score": 0.90}, {"score": 0.30}, {"score": 0.25}]
        assert len(_apply_cliff_detection(c, min_top_k=1)) == 3

    def test_absolute_rule_alone_can_trigger(self):
        # 0.95 → 0.10：差距 0.85 >= 0.8。关掉相对规则后仍命中，
        # 证明绝对规则本身可用（分数 <=1 时它其实被相对规则蕴含，见实现注释）
        c = [{"score": 0.95}, {"score": 0.10}]
        assert len(
            _apply_cliff_detection(c, rel_threshold=0, abs_threshold=0.8, min_top_k=1)
        ) == 1

    def test_relative_boundary_is_inclusive(self):
        """「跌幅 >= 40%」含等号：恰好 40% 命中，39% 不命中。"""
        hit = [{"score": 0.50}, {"score": 0.30}]      # 0.20/0.50 = 40%
        miss = [{"score": 0.50}, {"score": 0.305}]    # 0.195/0.50 = 39%

        assert len(_apply_cliff_detection(hit, min_top_k=1)) == 1
        assert len(_apply_cliff_detection(miss, min_top_k=1)) == 2

    def test_absolute_boundary_is_inclusive(self):
        hit = [{"score": 1.00}, {"score": 0.20}]      # 差距 0.80
        miss = [{"score": 1.00}, {"score": 0.21}]     # 差距 0.79

        assert len(_apply_cliff_detection(hit, rel_threshold=0, min_top_k=1)) == 1
        assert len(_apply_cliff_detection(miss, rel_threshold=0, min_top_k=1)) == 2

    def test_floor_keeps_min_top_k(self):
        """兜底：断崖落在首条之后，仍至少保留 min_top_k 条。"""
        c = [{"score": 0.95}] + [{"score": 0.005}] * 19

        assert len(_apply_cliff_detection(c, min_top_k=5)) == 5

    def test_default_floor_is_five(self):
        """生产默认下限为 5（回归点：曾为 3，极端断崖下重排候选过少）。"""
        from finance_rag.src.core.config import RERANKER_CLIFF_MIN_TOP_K

        assert RERANKER_CLIFF_MIN_TOP_K == 5
        c = [{"score": 0.95}] + [{"score": 0.005}] * 19

        assert len(_apply_cliff_detection(c)) == RERANKER_CLIFF_MIN_TOP_K

    def test_earliest_cliff_wins_single_scan(self):
        """单次扫描、最早断崖优先：不做「找最深落差」。

        这里 i=0 与 i=2 都命中；取最早一处则截到 2（受 min_top_k=2 托底），
        若按「最深落差」实现会截到 3。
        """
        c = [{"score": 0.95}, {"score": 0.30}, {"score": 0.29}, {"score": 0.02}]

        assert len(_apply_cliff_detection(c, min_top_k=2)) == 2

    def test_rule_level_disable(self):
        """规则级禁用：只关相对规则时，绝对规则单独生效（此例差距 0.40 < 0.8 不命中）。"""
        c = [{"score": 0.95}, {"score": 0.55}, {"score": 0.40}]

        assert len(_apply_cliff_detection(c, rel_threshold=0, min_top_k=1)) == 3

    def test_disabled_when_both_thresholds_off(self):
        """整体禁用：两条阈值都 <= 0 时原样返回（旧的「threshold<=0 即禁用」已不适用）。"""
        c = [{"score": 0.95}, {"score": 0.10}]

        assert len(
            _apply_cliff_detection(c, rel_threshold=0, abs_threshold=0, min_top_k=1)
        ) == 2

    def test_non_positive_score_is_skipped(self):
        """非正分数不做相对跌幅判定（避免除零），但仍可被绝对规则命中。"""
        c = [{"score": 0.0}, {"score": -0.5}]

        assert len(_apply_cliff_detection(c, min_top_k=1)) == 2
        assert len(
            _apply_cliff_detection(c, rel_threshold=0, abs_threshold=0.4, min_top_k=1)
        ) == 1

    def test_empty(self):
        assert _apply_cliff_detection([], min_top_k=1) == []

    def test_single(self):
        assert len(_apply_cliff_detection([{"score": 0.95}], min_top_k=1)) == 1


class TestFingerprintStore:
    def test_new_file(self):
        with tempfile.TemporaryDirectory() as d:
            store = FingerprintStore(os.path.join(d, ".fp"))
            assert not store.is_unchanged("test.md", "abc")

    def test_match(self):
        with tempfile.TemporaryDirectory() as d:
            store = FingerprintStore(os.path.join(d, ".fp"))
            store.update("test.md", "abc123", "/tmp/test.md", 5)
            assert store.is_unchanged("test.md", "abc123")

    def test_wrong_hash(self):
        with tempfile.TemporaryDirectory() as d:
            store = FingerprintStore(os.path.join(d, ".fp"))
            store.update("test.md", "abc123", "/tmp/test.md", 5)
            assert not store.is_unchanged("test.md", "wrong")

    def test_remove(self):
        with tempfile.TemporaryDirectory() as d:
            store = FingerprintStore(os.path.join(d, ".fp"))
            store.update("test.md", "abc123", "/tmp/test.md", 5)
            store.remove("test.md")
            assert not store.is_unchanged("test.md", "abc123")

    def test_hash_file(self):
        with tempfile.TemporaryDirectory() as d:
            fp = os.path.join(d, "test.txt")
            Path(fp).write_text("hello world", "utf-8")
            h1 = FingerprintStore.hash_file(fp)
            assert len(h1) == 64
            h2 = FingerprintStore.hash_file(fp)
            assert h1 == h2

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as d:
            fp_path = os.path.join(d, ".fp")
            store1 = FingerprintStore(fp_path)
            store1.update("doc.md", "hash123", "/tmp/doc.md", 10)
            store2 = FingerprintStore(fp_path)
            assert store2.is_unchanged("doc.md", "hash123")


class TestContextBuilding:
    def test_build_context(self):
        docs = [
            {"content": "房地产政策", "source": "r.pdf", "title": "报告", "chunk": 1, "score": 0.9},
            {"content": "金融稳定", "source": "r2.pdf", "title": "报告2", "chunk": 2, "score": 0.8},
        ]
        context, sources = build_context(docs)
        assert "[1]" in context
        assert "[2]" in context
        assert len(sources) == 2

    def test_build_context_empty(self):
        context, sources = build_context([])
        assert "未检索到" in context
        assert sources == []
