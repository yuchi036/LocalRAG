import pytest

from localrag.bm25 import BM25
from localrag.tokenize import _char_tokenize


def test_search_returns_scored_hits():
    docs = [
        _char_tokenize("完播率多少算优秀，短视频要看完"),
        _char_tokenize("选题方法论与内容策划流程"),
        _char_tokenize("互动率和点赞率计算公式"),
    ]
    bm = BM25(docs, tokenizer=_char_tokenize)
    # search() takes a raw query string and tokenizes internally
    hits = bm.search("完播率")
    assert hits, "expected at least one hit"
    top_idx = hits[0][1]
    assert top_idx == 0, "doc 0 contains 完播率 and should rank first"
    for sc, _ in hits:
        assert sc > 0


def test_more_matching_terms_yields_higher_score():
    docs = [_char_tokenize("完播率 互动率 点赞 评论")]
    bm = BM25(docs, tokenizer=_char_tokenize)
    s_one = bm.search("完播率", topk=1)[0][0]
    s_two = bm.search("完播率 互动率", topk=1)[0][0]
    assert s_two > s_one, "adding a matching term must increase score"


def test_matched_terms_explains_recall():
    docs = [_char_tokenize("完播率互动率点赞"), _char_tokenize("选题策划脚本")]
    bm = BM25(docs, tokenizer=_char_tokenize)
    assert bm.matched_terms("完播率", 0) == ["完", "播", "率"]
    assert bm.matched_terms("完播率", 1) == []
