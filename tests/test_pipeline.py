from localrag.bm25 import BM25
from localrag.ingestion import load_and_chunk
from localrag.metrics import (evaluate_retrieval, latency_percentiles,
                              mrr)
from localrag.pipeline import build_index, restore_or_build, run_query
from localrag.tokenize import _char_tokenize


def test_build_and_query_returns_matched(tmp_path):
    p = tmp_path / "kb.md"
    p.write_text("# A\n完播率多少算优秀\n# B\n选题方法论", encoding="utf-8")
    chunks, bm = build_index(str(p))
    r = run_query(chunks, bm, "完播率")
    assert r["hits"] and r["hits"][0]["matched"], "matched terms must be exposed"


def test_restore_or_build_second_launch_uses_cache(tmp_path):
    p = tmp_path / "kb.md"
    p.write_text("# A\nhello world", encoding="utf-8")
    idx = tmp_path / "idx.json"
    _, _, loaded1 = restore_or_build(str(p), index_path=str(idx))
    _, _, loaded2 = restore_or_build(str(p), index_path=str(idx))
    assert loaded1 is False and loaded2 is True


def test_evaluate_retrieval_shape_and_bounds(tmp_path):
    p = tmp_path / "kb.md"
    p.write_text("# 完播率\n完播率大于30为及格\n# 选题\n选题方法论", encoding="utf-8")
    chunks, bm = build_index(str(p))
    gt = [("完播率多少", ["完播率"])]
    m = evaluate_retrieval(chunks, bm, gt, topk=2)
    assert m["n"] == 1 and m["topk"] == 2
    assert 0 <= m["recall_at_k"] <= 1
    assert 0 <= m["mrr"] <= 1
    assert m["recall_at_k"] == 1.0  # the single gt must hit


def test_latency_percentiles_monotonic():
    docs = [_char_tokenize("hello world"), _char_tokenize("foo bar")]
    bm = BM25(docs, tokenizer=_char_tokenize)
    lat = latency_percentiles(bm, ["hello", "foo", "absent"], topk=2)
    assert lat["n"] == 3
    assert lat["p50"] >= 0 and lat["p95"] >= lat["p50"]


def test_mrr_perfect_ranking_is_one():
    # three queries, each hits at rank 1 -> mrr=1
    assert mrr([1, 1, 1]) == 1.0
    # all miss -> mrr=0
    assert mrr([0, 0]) == 0.0
