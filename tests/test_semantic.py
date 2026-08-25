import io
import sys

import pytest

from localrag.bm25 import BM25
from localrag.ingestion import load_and_chunk
from localrag.semantic import (_RRF_K, hybrid_search, is_available, rrf_fuse)
from localrag.tokenize import _char_tokenize


def test_is_available_returns_bool():
    assert isinstance(is_available(), bool)


def test_rrf_fuse_combines_bm25_and_semantic():
    # doc 1 在两路都命中，应比只在 bm25 命中的 doc 2 排得更前
    bm = [(10.0, 0), (8.0, 1), (5.0, 2)]
    sem = [1, 0, 2]
    fused = rrf_fuse(bm, sem, k=_RRF_K)
    idxs = [i for _, i in fused]
    assert set(idxs) == {0, 1, 2}
    scores = [s for s, _ in fused]
    assert scores == sorted(scores, reverse=True)
    rank = {i: r for r, (_, i) in enumerate(fused)}
    assert rank[1] < rank[2], "dual-hit doc should outrank single-source doc"


def test_rrf_fuse_handles_disjoint_sources():
    bm = [(10.0, 0)]
    sem = [1]
    fused = rrf_fuse(bm, sem, k=60)
    assert {i for _, i in fused} == {0, 1}


def test_hybrid_search_degrades_when_encoder_unavailable(tmp_path, monkeypatch):
    """缺省 encoder 且 sentence-transformers 不可用时，应降级为 BM25 并 warn。"""
    p = tmp_path / "kb.md"
    p.write_text("# A\n完播率多少算优秀\n# B\n选题方法论", encoding="utf-8")
    chunks = load_and_chunk(str(p))
    bm = BM25([_char_tokenize(c["text"]) for c in chunks], tokenizer=_char_tokenize)

    # 强制 _build_default_encoder 失败，模拟 sentence-transformers 缺失
    def _boom(model_name=None):
        raise ImportError("simulated missing sentence-transformers")
    monkeypatch.setattr("localrag.semantic._build_default_encoder", _boom)

    captured = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured)

    hits = hybrid_search(bm, chunks, "完播率", topk=2, encoder=None)
    assert hits, "graceful degradation must still return bm25 hits"
    assert all(idx < len(chunks) for _, idx in hits)
    # chunk 0 contains "完播率" and must be the (only) bm25 hit
    assert hits[0][1] == 0
    assert "降级" in captured.getvalue() or "hybrid" in captured.getvalue().lower()


class _FakeEnc:
    """可注入的假 encoder：让 chunk 0 与 query 语义最相似。"""

    def encode(self, texts, convert_to_numpy=True):
        # 返回 list of list[float]：第一个文本(query 时唯一)全 1；chunk 0 全 1，其余 0
        # 这样点积会把 chunk 0 排第一。
        out = []
        for i, _t in enumerate(texts):
            if i == 0 and len(texts) == 1:
                out.append([1.0, 0.0, 0.0])
            else:
                # 多文本：让第一个 chunk(也是 i=0)为 [1,0,0]，其余 [0,1,0] / [0,0,1]
                vec = [0.0, 0.0, 0.0]
                if i < 3:
                    vec[i] = 1.0
                out.append(vec)
        return out


def test_hybrid_search_uses_injected_encoder(tmp_path):
    p = tmp_path / "kb.md"
    p.write_text("# A\n完播率多少算优秀\n# B\n选题方法论", encoding="utf-8")
    chunks = load_and_chunk(str(p))
    bm = BM25([_char_tokenize(c["text"]) for c in chunks], tokenizer=_char_tokenize)
    enc = _FakeEnc()
    hits = hybrid_search(bm, chunks, "完播率", topk=2, encoder=enc)
    assert hits, "encoder path should produce hits"
    idxs = [i for _, i in hits]
    # 语义最相关的是 chunk 0；BM25 也应把它放在前面；融合后 chunk 0 必排第一
    assert idxs[0] == 0
