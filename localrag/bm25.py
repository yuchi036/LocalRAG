# -*- coding: utf-8 -*-
"""纯标准库 BM25 检索（零依赖、可逐行读懂）。

设计要点（S1 修复后）：
- tokenizer 在构造时注入并保存为 self.tokenizer；
- search() 用 self.tokenizer(query) 分词，文档与查询分词强一致。
"""
import math

from localrag.tokenize import _char_tokenize, _jieba_tokenize


class BM25:
    def __init__(self, docs, tokenizer=_char_tokenize, k1=1.5, b=0.75):
        self.tokenizer = tokenizer
        self.tokenizer_name = self._name_of(tokenizer)
        self.k1 = k1
        self.b = b
        self.docs = docs
        self.dl = [len(d) for d in docs]
        self.avgdl = sum(self.dl) / (len(self.dl) or 1)
        self.df = {}
        self.f = []
        for d in docs:
            freq = {}
            for t in d:
                freq[t] = freq.get(t, 0) + 1
            self.f.append(freq)
            for t in freq:
                self.df[t] = self.df.get(t, 0) + 1
        self.idf_cache = {}

    def idf(self, t):
        if t not in self.idf_cache:
            n = self.df.get(t, 0)
            self.idf_cache[t] = math.log(1 + (len(self.dl) - n + 0.5) / (n + 0.5))
        return self.idf_cache[t]

    def search(self, query, topk=3):
        q_tokens = self.tokenizer(query)
        if not q_tokens:
            return []
        scored = []
        for idx in range(len(self.docs)):
            sc = self._score_query(q_tokens, idx)
            if sc > 0:
                scored.append((sc, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sc, idx) for sc, idx in scored[:topk]]

    @staticmethod
    def _name_of(tok):
        """把分词器函数映射成可持久化的名字（char / jieba）。"""
        if tok is _jieba_tokenize:
            return "jieba"
        return "char"

    def to_dict(self):
        """导出可序列化状态，供 persist.save_index 落盘（无需重算 df/f）。"""
        return {
            "tokenizer_name": self.tokenizer_name,
            "k1": self.k1,
            "b": self.b,
            "docs": self.docs,
            "dl": self.dl,
            "avgdl": self.avgdl,
            "df": self.df,
            "f": self.f,
        }

    @classmethod
    def from_dict(cls, d):
        """从 to_dict 的状态重建 BM25，跳过耗时的分词与 df/f 重算（秒开）。"""
        from localrag.tokenize import get_tokenizer
        obj = cls.__new__(cls)
        obj.tokenizer = get_tokenizer(d.get("tokenizer_name", "char"))
        obj.tokenizer_name = d.get("tokenizer_name", "char")
        obj.k1 = d["k1"]
        obj.b = d["b"]
        obj.docs = d["docs"]
        obj.dl = d["dl"]
        obj.avgdl = d["avgdl"]
        obj.df = d["df"]
        obj.f = d["f"]
        obj.idf_cache = {}
        return obj

    def _score_query(self, q_tokens, idx):
        score = 0.0
        for t in q_tokens:
            if t not in self.df:
                continue
            f = self.f[idx].get(t, 0)
            if f == 0:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * self.dl[idx] / (self.avgdl or 1))
            score += self.idf(t) * (f * (self.k1 + 1)) / denom
        return score
