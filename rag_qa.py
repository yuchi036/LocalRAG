#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地知识问答（RAG 全链路：Retrieval + Generation，完全离线）
- Retrieval 段：纯标准库实现 BM25，对 Markdown 分块、中文粗粒度切词后召回 Top-K 片段
- Generation 段：可选接入本机 Ollama 模型（如 qwen2.5:1.5b）基于召回片段作答
- 全程零云端依赖：不联网、数据不出本机、无 API 费用

用法：
  # 仅检索（默认，最快）
  python rag_qa.py --doc 生成式AI短视频内容营销知识库.md --query "完播率多少算优秀"

  # 检索 + 本地模型生成（离线 RAG 闭环，需先 `ollama pull qwen2.5:1.5b` 并启动服务）
  python rag_qa.py --doc 生成式AI短视频内容营销知识库.md --query "完播率多少算优秀" --generate
  python rag_qa.py --doc 生成式AI短视频内容营销知识库.md --questions questions.txt --generate --topk 2
"""

import argparse
import json
import math
import re
import sys
import urllib.request


def load_and_chunk(path):
    """读取 Markdown，按 '#' 标题分块，保留层级标题作为 chunk 标题。"""
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    cur_title = "(开头)"
    cur_body = []
    for ln in lines:
        if ln.strip().startswith("#"):
            if cur_body:
                chunks.append({"title": cur_title, "text": "".join(cur_body).strip()})
            cur_title = ln.strip().lstrip("#").strip()
            cur_body = []
        else:
            cur_body.append(ln)
    if cur_body:
        chunks.append({"title": cur_title, "text": "".join(cur_body).strip()})
    # 过滤空块
    chunks = [c for c in chunks if c["text"]]
    return chunks


def tokenize(text):
    """中文粗粒度切词：去标点后按字符 + 英文单词。检索层面够用。"""
    text = re.sub(r"[\s\W_]+", " ", text)
    text = text.lower()
    # 保留连续英文/数字作为 token，中文逐字
    tokens = []
    for m in re.finditer(r"[a-z0-9]+|[\u4e00-\u9fff]", text):
        tok = m.group(0)
        if len(tok) == 1 and "\u4e00" <= tok <= "\u9fff":
            tokens.append(tok)
        else:
            tokens.append(tok)
    return tokens


class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
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
            # 平滑 IDF
            self.idf_cache[t] = math.log(1 + (len(self.dl) - n + 0.5) / (n + 0.5))
        return self.idf_cache[t]

    def score(self, q):
        s = 0.0
        for t in q:
            if t not in self.df:
                continue
            idf = self.idf(t)
            for idx in range(len(self.docs)):
                f = self.f[idx].get(t, 0)
                if f == 0:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.dl[idx] / (self.avgdl or 1))
                s += idf * (f * (self.k1 + 1)) / denom
        return s

    def search(self, query, topk=3):
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scored = []
        for idx in range(len(self.docs)):
            sc = self._score_query(q_tokens, idx)
            if sc > 0:
                scored.append((sc, idx))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(sc, idx) for sc, idx in scored[:topk]]

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


def generate_with_ollama(question, contexts, model="qwen2.5:1.5b", base_url="http://localhost:11434"):
    """本地生成段：把召回片段拼成上下文，调用本机 Ollama 模型作答。零云端依赖。"""
    context = "\n\n".join(f"【{title}】\n{text}" for title, text in contexts)
    prompt = (
        "你是一个严谨的问答助手。请只根据下面提供的「资料」回答问题，"
        "不要使用资料之外的知识；如果资料中没有相关信息，请回答「资料中未提及」。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question}\n\n回答："
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 320},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()
    except Exception as e:  # noqa: BLE001
        return f"（本地模型调用失败：{e}；请确认 Ollama 已启动且模型已拉取）"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="生成式AI短视频内容营销知识库.md")
    ap.add_argument("--questions", default=None, help="问题列表文件，一行一个")
    ap.add_argument("--query", default=None, help="单个问题")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--generate", action="store_true",
                    help="启用本地模型(Ollama)生成答案，实现离线 RAG 闭环")
    ap.add_argument("--model", default="qwen2.5:1.5b", help="Ollama 模型名")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama 服务地址")
    args = ap.parse_args()

    chunks = load_and_chunk(args.doc)
    if not chunks:
        print("未读取到内容块", file=sys.stderr)
        sys.exit(1)
    tokenized = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25(tokenized)

    queries = []
    if args.questions:
        with open(args.questions, "r", encoding="utf-8") as f:
            queries = [l.strip() for l in f if l.strip()]
    if args.query:
        queries.append(args.query)

    for q in queries:
        print("\n" + "=" * 70)
        print("问：", q)
        hits = bm25.search(q, topk=args.topk)
        if not hits:
            print("（未命中相关片段）")
            continue
        for rank, (sc, idx) in enumerate(hits, 1):
            print(f"\n  [{rank}] 命中小节：{chunks[idx]['title']}  (score={sc:.3f})")
            snippet = chunks[idx]["text"].replace("\n", " ")
            print("      " + snippet[:240] + ("…" if len(snippet) > 240 else ""))

        if args.generate:
            contexts = [(chunks[idx]["title"], chunks[idx]["text"]) for _, idx in hits]
            print(f"\n  [本地模型作答 · {args.model}]")
            answer = generate_with_ollama(
                q, contexts, model=args.model, base_url=args.ollama_url
            )
            print("  " + answer.replace("\n", "\n  "))


if __name__ == "__main__":
    main()
