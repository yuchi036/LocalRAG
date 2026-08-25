# -*- coding: utf-8 -*-
"""文档加载与分块：支持 .md/.txt/.pdf 与目录递归索引（零依赖，PDF 需可选 PyPDF2）。"""
import os
import re
import sys

SUPPORTED_EXT = {".md", ".markdown", ".txt", ".pdf"}


def _read_text(path):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def _read_pdf(path):
    try:
        import PyPDF2
    except ImportError:
        print(f"[warn] 读取 PDF 需 PyPDF2：{os.path.basename(path)}（可 `pip install PyPDF2`）", file=sys.stderr)
        return None
    try:
        reader = PyPDF2.PdfReader(path)
        return "\n".join((p.extract_text() or "") for p in reader.pages)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] PDF 解析失败 {os.path.basename(path)}：{e}", file=sys.stderr)
        return None


def _chunk_markdown(text, source):
    """按 '#' 标题切块，保留标题；无标题内容归入 '(开头)'。"""
    chunks = []
    cur_title = "(开头)"
    cur_body = []
    for ln in text.splitlines():
        if ln.strip().startswith("#"):
            if cur_body:
                chunks.append({"title": cur_title, "text": "".join(cur_body).strip(), "source": source})
            cur_title = ln.strip().lstrip("#").strip()
            cur_body = []
        else:
            cur_body.append(ln)
    if cur_body:
        chunks.append({"title": cur_title, "text": "".join(cur_body).strip(), "source": source})
    return [c for c in chunks if c["text"]]


def _chunk_plain(text, source):
    """无标题的纯文本/PDF：按空行或固定长度切块，保证可检索。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    buf = []
    for p in paras:
        buf.append(p)
        if sum(len(x) for x in buf) > 800:
            chunks.append({"title": source, "text": "\n\n".join(buf), "source": source})
            buf = []
    if buf:
        chunks.append({"title": source, "text": "\n\n".join(buf), "source": source})
    return [c for c in chunks if c["text"]]


def _load_one(fp):
    ext = os.path.splitext(fp)[1].lower()
    source = os.path.basename(fp)
    if ext == ".pdf":
        text = _read_pdf(fp)
        return _chunk_plain(text, source) if text else []
    text = _read_text(fp)
    if ext in (".md", ".markdown"):
        return _chunk_markdown(text, source)
    return _chunk_plain(text, source)


def load_and_chunk(path):
    """path 可为：单个 .md/.txt/.pdf 文件，或一个目录（递归索引所有支持文件）。"""
    if os.path.isdir(path):
        files = []
        for root, _, names in os.walk(path):
            for n in sorted(names):
                if os.path.splitext(n)[1].lower() in SUPPORTED_EXT:
                    files.append(os.path.join(root, n))
        files.sort()
        if not files:
            return []
        chunks = []
        for fp in files:
            chunks.extend(_load_one(fp))
        return chunks
    return _load_one(path)
