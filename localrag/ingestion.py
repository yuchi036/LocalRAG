# -*- coding: utf-8 -*-
"""文档加载与分块：支持 .md/.txt/.pdf 与目录递归索引（零依赖，PDF 需可选 PyPDF2）。

S3 增强——来源可定位：
- source 在目录索引时记为相对路径，单文件记为文件名；
- 每个 chunk 追加 loc 字段（如 "L12-45"，1-based 行号范围），便于回溯原文。
"""
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


def _mk_chunk(title, text, source, start_line=0, end_line=0):
    """构造 chunk；当行号有效时附带 loc（如 L12-45），便于回溯原文。"""
    chunk = {"title": title, "text": text, "source": source}
    if end_line >= start_line >= 1:
        chunk["loc"] = f"L{start_line}-{end_line}"
    return chunk


def _chunk_markdown(text, source):
    """按 '#' 标题切块，保留标题；记录每个块的起止行号（1-based）。"""
    chunks = []
    cur_title = "(开头)"
    cur_body = []
    cur_start = 1
    line_no = 0
    for ln in text.splitlines():
        line_no += 1
        if ln.strip().startswith("#"):
            if cur_body:
                chunks.append(_mk_chunk(cur_title, "".join(cur_body).strip(), source, cur_start, line_no - 1))
            cur_title = ln.strip().lstrip("#").strip()
            cur_body = []
            cur_start = line_no
        else:
            cur_body.append(ln + "\n")
    if cur_body:
        chunks.append(_mk_chunk(cur_title, "".join(cur_body).strip(), source, cur_start, line_no))
    return [c for c in chunks if c["text"]]


def _chunk_plain(text, source):
    """无标题的纯文本/PDF：按空行或固定长度切块，并记录近似行号范围。"""
    paras = []
    for m in re.finditer(r"(?:^|\n[ \t]*\n)([^\n].*?)(\n[ \t]*\n|$)", text, re.S):
        seg = m.group(1).strip()
        if seg:
            paras.append((seg, m.start(1)))
    if not paras:  # 兜底：正则未命中时回退到按空行切分
        for p in re.split(r"\n\s*\n", text):
            p = p.strip()
            if p:
                paras.append((p, text.find(p)))

    chunks = []
    buf = []
    buf_start = None
    buf_chars = 0
    n = len(paras)
    for i, (p, off) in enumerate(paras):
        if buf_start is None:
            buf_start = off
        buf.append(p)
        buf_chars += len(p)
        end_off = off + len(p)
        if buf_chars > 800 or i == n - 1:
            start_line = text.count("\n", 0, buf_start) + 1
            end_line = text.count("\n", 0, end_off) + 1
            chunks.append(_mk_chunk(source, "\n\n".join(buf), source, start_line, end_line))
            buf = []
            buf_start = None
            buf_chars = 0
    return [c for c in chunks if c["text"]]


def _load_one(fp, root=None):
    ext = os.path.splitext(fp)[1].lower()
    source = os.path.relpath(fp, root) if root else os.path.basename(fp)
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
        root = path
        files = []
        for sub, _, names in os.walk(path):
            for nm in sorted(names):
                if os.path.splitext(nm)[1].lower() in SUPPORTED_EXT:
                    files.append(os.path.join(sub, nm))
        files.sort()
        if not files:
            return []
        chunks = []
        for fp in files:
            chunks.extend(_load_one(fp, root))
        return chunks
    return _load_one(path, None)
