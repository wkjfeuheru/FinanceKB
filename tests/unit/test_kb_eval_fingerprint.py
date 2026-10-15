from finance_rag.src.eval.testset_generator import knowledge_base_fingerprint


class KB:
    def __init__(self):
        self.documents = [{"source": "report.pdf", "version": "v1", "chunk_count": 3,
                           "report_type": "个股", "security_name": "杰瑞股份", "date": "2026-10-01"}]

    def list_documents(self):
        return self.documents

    def get_fingerprint_store(self):
        return {"report.pdf": {"sha256": "unchanged"}}


def test_metadata_patch_changes_eval_kb_fingerprint():
    kb = KB()
    before = knowledge_base_fingerprint(kb)
    kb.documents[0].update(report_type="行业", security_name="", industry_l1="通信")
    assert knowledge_base_fingerprint(kb) != before


def test_eval_fingerprint_is_stable_for_unrelated_review_flag():
    kb = KB()
    before = knowledge_base_fingerprint(kb)
    kb.documents[0]["needs_review"] = True
    assert knowledge_base_fingerprint(kb) == before
