# -*- coding: utf-8 -*-
"""中文分词：默认零依赖字级，可选 jieba 短语级。

分词器通过 BM25 构造注入，保证文档与查询使用同一实例（见 S1 修复的全局态 BUG）。
"""
import re
import sys


def _char_tokenize(text):
    """中文按字、英文/数字按词（零依赖，检索层够用）。"""
    text = re.sub(r"[\s\W_]+", " ", text).lower()
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)


def _jieba_tokenize(text):
    import jieba
    jieba.setLogLevel(20)
    text = re.sub(r"[\s\W_]+", " ", text).lower()
    return [t for t in jieba.cut(text) if t.strip()]


def get_tokenizer(segment="auto"):
    """segment: 'char' | 'jieba' | 'auto'（默认 auto：有 jieba 用 jieba，否则字级）。"""
    if segment == "char":
        return _char_tokenize
    need_jieba = segment in ("jieba", "auto")
    if need_jieba:
        try:
            import jieba  # noqa: F401
            return _jieba_tokenize
        except ImportError:
            if segment == "jieba":
                print("[warn] jieba 未安装，已回退字级分词；可 `pip install jieba`", file=sys.stderr)
    return _char_tokenize
