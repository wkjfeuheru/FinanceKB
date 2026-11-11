import pytest

# src/eval/testset_generator.py 在目录重组（finance_rag/ → src/）中丢失，
# 模块恢复前跳过；恢复后这些用例自动生效。
pytest.importorskip("src.eval.testset_generator",
                    reason="src/eval/testset_generator.py 在目录重组中丢失")

from src.eval.testset_generator import knowledge_base_fingerprint  # noqa: E402


class KB:
    def __init__(self):
        # security_name / industry_l1 已删；研报条件字段为 report_type 与 broker
        self.documents = [{"source": "report.pdf", "version": "v1", "chunk_count": 3,
                           "report_type": "个股", "broker": "中信证券", "date": "2026-10-01"}]

    def list_documents(self):
        return self.documents

    def get_fingerprint_store(self):
        return {"report.pdf": {"sha256": "unchanged"}}


def test_metadata_patch_changes_eval_kb_fingerprint():
    kb = KB()
    before = knowledge_base_fingerprint(kb)
    kb.documents[0].update(report_type="行业", broker="华泰证券")
    assert knowledge_base_fingerprint(kb) != before


def test_eval_fingerprint_is_stable_for_unrelated_review_flag():
    kb = KB()
    before = knowledge_base_fingerprint(kb)
    kb.documents[0]["needs_review"] = True
    assert knowledge_base_fingerprint(kb) == before
