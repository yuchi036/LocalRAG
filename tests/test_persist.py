import os
import pytest

from localrag.persist import INDEX_VERSION, load_index, save_index
from localrag.bm25 import BM25
from localrag.ingestion import load_and_chunk
from localrag.tokenize import _char_tokenize


def _build_index(doc):
    chunks = load_and_chunk(doc)
    bm = BM25([_char_tokenize(c["text"]) for c in chunks], tokenizer=_char_tokenize)
    return chunks, bm


def test_save_load_roundtrip_json(tmp_path):
    p = tmp_path / "x.md"
    p.write_text("# T\nhi\n# U\nbye", encoding="utf-8")
    chunks, bm = _build_index(str(p))
    f = tmp_path / "idx.json"
    save_index(str(f), chunks, bm, str(p))
    loaded = load_index(str(f), doc_path=str(p), tokenizer_name="char")
    assert loaded is not None
    c2, bm2 = loaded
    assert [c["title"] for c in c2] == [c["title"] for c in chunks]


def test_load_returns_none_on_tokenizer_mismatch(tmp_path):
    p = tmp_path / "x.md"; p.write_text("# T\nhi", encoding="utf-8")
    chunks, bm = _build_index(str(p))
    f = tmp_path / "idx.json"
    save_index(str(f), chunks, bm, str(p))
    assert load_index(str(f), doc_path=str(p), tokenizer_name="jieba") is None


def test_load_returns_none_on_doc_change(tmp_path):
    p = tmp_path / "x.md"; p.write_text("# T\nhi", encoding="utf-8")
    chunks, bm = _build_index(str(p))
    f = tmp_path / "idx.json"
    save_index(str(f), chunks, bm, str(p))
    other = tmp_path / "y.md"; other.write_text("# T\nbye", encoding="utf-8")
    assert load_index(str(f), doc_path=str(other), tokenizer_name="char") is None


def test_pickle_roundtrip(tmp_path):
    p = tmp_path / "x.md"; p.write_text("# T\nhi", encoding="utf-8")
    chunks, bm = _build_index(str(p))
    f = tmp_path / "idx.pkl"
    save_index(str(f), chunks, bm, str(p), fmt="pickle")
    loaded = load_index(str(f), doc_path=str(p), tokenizer_name="char")
    assert loaded is not None


def test_index_version_constant():
    assert isinstance(INDEX_VERSION, str) and INDEX_VERSION
