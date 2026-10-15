"""ONNX embedding pooling and configuration guardrails."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from finance_rag.src.core.exceptions import EmbeddingError
from finance_rag.src.infrastructure.vector_store import onnx_embedder as embedder_module
from finance_rag.src.infrastructure.vector_store.onnx_embedder import OnnxEmbedder


class _FakeTokenizer:
    def __call__(self, texts, **kwargs):
        del texts, kwargs
        return {
            "input_ids": np.array([[11, 12]], dtype=np.int64),
            "attention_mask": np.array([[1, 1]], dtype=np.int64),
        }


class _FakeSession:
    def __init__(self, last_hidden: np.ndarray):
        self._last_hidden = last_hidden

    def get_inputs(self):
        return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

    def run(self, output_names, feed):
        del output_names, feed
        return [self._last_hidden]


@pytest.fixture(autouse=True)
def _clear_embedder_singletons():
    OnnxEmbedder._instances.clear()
    yield
    OnnxEmbedder._instances.clear()


def _embedder(monkeypatch, pooling_method: str, *, expected_dimension: int = 2):
    monkeypatch.setattr(embedder_module, "EMBEDDING_DIM", expected_dimension)
    embedder = OnnxEmbedder(model_name="test/model", pooling_method=pooling_method)
    session = _FakeSession(
        np.array([[[3.0, 4.0], [0.0, 2.0]]], dtype=np.float32)
    )
    embedder._session = session
    embedder._tokenizer = _FakeTokenizer()
    return embedder


def test_cls_pooling_uses_normalized_first_token(monkeypatch):
    embedder = _embedder(monkeypatch, "cls")

    vector = embedder.embed_query("ignored")

    assert vector == pytest.approx([0.6, 0.8])


def test_mean_pooling_uses_attention_mask_then_normalizes(monkeypatch):
    embedder = _embedder(monkeypatch, "mean")

    vector = embedder.embed_query("ignored")

    assert vector == pytest.approx([0.4472136, 0.8944272])


def test_model_and_pooling_method_form_singleton_identity(monkeypatch):
    monkeypatch.setattr(embedder_module, "EMBEDDING_DIM", 2)

    cls_embedder = OnnxEmbedder(model_name="test/model", pooling_method="cls")
    mean_embedder = OnnxEmbedder(model_name="test/model", pooling_method="mean")
    other_model = OnnxEmbedder(model_name="test/other", pooling_method="cls")

    assert cls_embedder is not mean_embedder
    assert cls_embedder is not other_model
    assert OnnxEmbedder(model_name="test/model", pooling_method="cls") is cls_embedder


def test_invalid_pooling_method_fails_fast(monkeypatch):
    monkeypatch.setattr(embedder_module, "EMBEDDING_DIM", 2)

    with pytest.raises(ValueError, match="pooling_method.*cls.*mean"):
        OnnxEmbedder(model_name="test/model", pooling_method="max")


def test_invalid_pooling_environment_value_fails_during_config_import():
    env = os.environ.copy()
    env["EMBEDDING_POOLING"] = "max"

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from finance_rag.src.core.config import EMBEDDING_POOLING",
        ],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "EMBEDDING_POOLING" in result.stderr
    assert "cls" in result.stderr
    assert "mean" in result.stderr


def test_embedding_dimension_mismatch_raises_clear_error(monkeypatch):
    embedder = _embedder(monkeypatch, "cls", expected_dimension=3)

    with pytest.raises(
        EmbeddingError,
        match=r"test/model.*actual=2.*expected=3",
    ):
        embedder.embed_query("ignored")
