import os
import pytest

from localrag.ingestion import _chunk_markdown, _chunk_plain, load_and_chunk


def test_markdown_chunks_carry_loc():
    text = "# 标题1\n内容一\n# 标题2\n内容二\n内容三"
    chunks = _chunk_markdown(text, source="t.md")
    assert len(chunks) >= 2
    assert all("loc" in c and c["loc"].startswith("L") for c in chunks)
    assert chunks[0]["title"] == "标题1"
    assert chunks[1]["title"] == "标题2"
    # 行号应为正整数范围
    assert "1-" in chunks[0]["loc"]


def test_load_and_chunk_single_file(tmp_path):
    p = tmp_path / "a.md"
    p.write_text("# 标题\nhello world\n# 标题2\nmore content", encoding="utf-8")
    chunks = load_and_chunk(str(p))
    assert len(chunks) == 2
    assert chunks[0]["source"].endswith("a.md")


def test_load_and_chunk_dir_uses_relative_source(tmp_path):
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "n.md").write_text("# X\nbody", encoding="utf-8")
    chunks = load_and_chunk(str(tmp_path))
    assert chunks and "sub" in chunks[0]["source"]
