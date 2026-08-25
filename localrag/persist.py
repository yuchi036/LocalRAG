# -*- coding: utf-8 -*-
"""索引持久化：把 BM25 状态 + chunks 落盘，二次启动免重建（大语料秒开）。

- 透明优先：默认 JSON（人可直读，符合 LocalRAG「透明可审计」定位）；
  大语料可用 .pkl / .pickle 走 pickle，序列化更快、体积更小。
- 带版本号字段 INDEX_VERSION，结构变更可安全失效旧索引。
- 记录文档签名（mtime/size 或目录递归签名），文档未变则直接加载，变了则重建。
"""
import hashlib
import json
import os
import pickle

from localrag.bm25 import BM25


INDEX_VERSION = "1"


def _doc_signature(doc_path):
    """生成文档签名：文件用 mtime+size；目录递归汇总所有支持文件的 mtime+size。"""
    SUPPORTED = (".md", ".markdown", ".txt", ".pdf")
    if os.path.isdir(doc_path):
        parts = []
        for root, _, names in os.walk(doc_path):
            for n in sorted(names):
                if os.path.splitext(n)[1].lower() in SUPPORTED:
                    fp = os.path.join(root, n)
                    try:
                        st = os.stat(fp)
                        parts.append(f"{os.path.relpath(fp, doc_path)}:{st.st_mtime:.0f}:{st.st_size}")
                    except OSError:
                        continue
        parts.sort()
        joined = "\n".join(parts)
        return "dir:" + hashlib.sha1(joined.encode("utf-8")).hexdigest()[:16]
    try:
        st = os.stat(doc_path)
        return f"file:{st.st_mtime:.0f}:{st.st_size}"
    except OSError:
        return "file:missing"


def save_index(path, chunks, bm25, doc_path, fmt="auto"):
    """保存索引。fmt='auto' 时按扩展名判定（.pkl/.pickle→pickle，否则 JSON）。"""
    if fmt == "auto":
        ext = os.path.splitext(path)[1].lower()
        fmt = "pickle" if ext in (".pkl", ".pickle") else "json"
    payload = {
        "version": INDEX_VERSION,
        "doc_path": doc_path,
        "doc_signature": _doc_signature(doc_path),
        "tokenizer_name": bm25.tokenizer_name,
        "chunks": chunks,
        "bm25": bm25.to_dict(),
    }
    if fmt == "pickle":
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)


def load_index(path, doc_path=None, tokenizer_name=None):
    """加载索引。返回 (chunks, bm25) 或 None（不存在/版本不符/文档已变/分词不符）。

    校验顺序：文件存在 → 版本号 → doc_path 一致 → 文档签名未变 → 分词器一致。
    任一不符即返回 None，由调用方重建并重新保存，保证数据始终与文档同步。
    """
    if not os.path.exists(path):
        return None
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".pkl", ".pickle"):
            with open(path, "rb") as f:
                payload = pickle.load(f)
        else:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
    except (OSError, ValueError, pickle.PickleError):
        return None

    if payload.get("version") != INDEX_VERSION:
        return None
    if doc_path is not None and payload.get("doc_path") != doc_path:
        return None
    if payload.get("doc_signature") != _doc_signature(doc_path or payload.get("doc_path", "")):
        return None
    if tokenizer_name is not None and payload.get("tokenizer_name") != tokenizer_name:
        return None

    chunks = payload["chunks"]
    bm25 = BM25.from_dict(payload["bm25"])
    return chunks, bm25


def make_index_path(doc_path):
    """默认索引文件名：与文档同目录同名的 .localrag_index.json。"""
    if os.path.isdir(doc_path):
        return os.path.join(doc_path, ".localrag_index.json")
    base, _ = os.path.splitext(doc_path)
    return base + ".localrag_index.json"
