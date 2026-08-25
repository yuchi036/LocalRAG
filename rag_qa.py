#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地知识问答（RAG 全链路：Retrieval + Generation，完全离线）
- Retrieval 段：纯标准库实现 BM25，对 Markdown 分块、中文粗粒度切词后召回 Top-K 片段
- Generation 段：可选接入本机 Ollama 模型（如 qwen2.5:1.5b）基于召回片段作答
- 交互：支持 --query / --questions 跑批、--interactive 交互问答、--serve 零依赖 Web UI
- 全程零云端依赖：不联网、数据不出本机、无 API 费用

用法：
  # 仅检索（默认，最快，零依赖）
  python rag_qa.py --query "完播率多少算优秀"

  # 检索 + 本地模型生成（离线 RAG 闭环，需先 ollama pull qwen2.5:1.5b 并启动服务）
  python rag_qa.py --query "完播率多少算优秀" --generate

  # 交互式问答（边问边答，输入 exit 退出）
  python rag_qa.py --interactive

  # 启动本地网页问答（浏览器打开 http://localhost:8000）
  python rag_qa.py --serve
"""

import argparse
import html
import http.server
import json
import math
import re
import sys
import urllib.parse
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
    tokens = []
    for m in re.finditer(r"[a-z0-9]+|[\u4e00-\u9fff]", text):
        tokens.append(m.group(0))
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

    def search(self, query, topk=3):
        """返回 [(score, idx), ...]，按分数降序，最多 topk 个；分数>0 才入选。"""
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


def build_index(doc_path):
    """构建 BM25 索引并返回 (chunks, bm25)。供 CLI / 交互 / Web 复用。"""
    chunks = load_and_chunk(doc_path)
    bm25 = BM25([tokenize(c["text"]) for c in chunks])
    return chunks, bm25


def run_query(chunks, bm25, question, topk=2, generate=False,
              model="qwen2.5:1.5b", ollama_url="http://localhost:11434"):
    """统一的查询入口：检索（必做）+ 可选本地模型生成，返回结果字典。"""
    hits = bm25.search(question, topk=topk)
    result = {"question": question, "hits": [], "answer": None}
    for sc, idx in hits:
        result["hits"].append({
            "title": chunks[idx]["title"],
            "text": chunks[idx]["text"],
            "score": sc,
        })
    if generate and hits:
        contexts = [(c["title"], c["text"]) for c in result["hits"]]
        result["answer"] = generate_with_ollama(
            question, contexts, model=model, base_url=ollama_url
        )
    return result


def print_result(result, show_answer=True):
    """把 run_query 的结果按人类可读格式打印到终端。"""
    print("\n" + "=" * 70)
    print("问：", result["question"])
    if not result["hits"]:
        print("（未命中相关片段）")
        return
    for rank, h in enumerate(result["hits"], 1):
        print(f"\n  [{rank}] 命中小节：{h['title']}  (score={h['score']:.3f})")
        snippet = h["text"].replace("\n", " ")
        print("      " + snippet[:240] + ("…" if len(snippet) > 240 else ""))
    if show_answer and result.get("answer") is not None:
        print("\n  [本地模型作答]")
        print("  " + result["answer"].replace("\n", "\n  "))


def interactive_mode(chunks, bm25, args):
    """交互式问答：输入问题回车即得结果，输入 exit/quit 退出。"""
    print("=== 交互模式（输入问题回车提问；输入 exit 退出）===")
    while True:
        try:
            q = input("\n你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            break
        if not q:
            continue
        if q.lower() in ("exit", "quit", "q"):
            print("再见。")
            break
        print_result(run_query(chunks, bm25, q, topk=args.topk,
                               generate=args.generate, model=args.model,
                               ollama_url=args.ollama_url))


# ---------------- 极简 Web UI（纯标准库，零依赖，离线）----------------

WEB_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AIGC 知识库 · 本地 RAG 问答</title>
<style>
  body{font-family:system-ui,'Microsoft YaHei',sans-serif;max-width:840px;margin:36px auto;padding:0 16px;background:#0f1115;color:#e6e6e6;line-height:1.6}
  h1{font-size:20px;border-bottom:1px solid #2a2f3a;padding-bottom:10px}
  form{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:18px 0}
  input[type=text]{flex:1;min-width:240px;padding:10px;border:1px solid #2a2f3a;border-radius:8px;background:#161a21;color:#e6e6e6;font-size:15px}
  button{padding:10px 18px;border:0;border-radius:8px;background:#3b82f6;color:#fff;font-size:15px;cursor:pointer}
  button:hover{background:#2563eb}
  label{font-size:14px;color:#9aa4b2}
  hr{border:0;border-top:1px solid #2a2f3a;margin:22px 0}
  .q{font-size:16px;font-weight:600;margin-bottom:12px}
  .hit{background:#161a21;border:1px solid #2a2f3a;border-radius:8px;padding:12px 14px;margin-bottom:10px}
  .hit b{color:#7dd3fc}.sc{color:#64748b;font-size:13px;margin-left:6px}
  .sn{color:#b8c0cc;font-size:14px;margin-top:6px}
  .ans{background:#10231a;border:1px solid #1f4d39;border-radius:8px;padding:12px 14px;margin-top:14px;color:#cfe9d8}
  .miss{color:#f87171}
  .tip{color:#64748b;font-size:13px}
</style></head><body>
<h1>本地 RAG 问答（BM25 检索 + 可选本地模型生成）</h1>
<form method="post" action="/">
  <input type="text" name="q" placeholder="输入你的问题，如：完播率多少算优秀、投流怎么赛马放量" value="{Q}">
  <label><input type="checkbox" name="generate" {GEN}> 启用本地模型</label>
  <button type="submit">提问</button>
</form>
<p class="tip">纯标准库实现，离线可跑；勾选「启用本地模型」需本机已启动 Ollama 并拉取 qwen2.5:1.5b。</p>
<hr>
{RESULTS}
</body></html>"""


def build_results_html(res):
    parts = [f'<div class="q">问：{html.escape(res["question"])}</div>']
    if not res["hits"]:
        parts.append('<div class="miss">未命中相关片段</div>')
    else:
        for i, h in enumerate(res["hits"], 1):
            snippet = html.escape(h["text"].replace("\n", " "))[:240]
            parts.append(
                f'<div class="hit"><b>[{i}] {html.escape(h["title"])}</b>'
                f'<span class="sc">score={h["score"]:.3f}</span>'
                f'<div class="sn">{snippet}</div></div>'
            )
    if res.get("answer"):
        ans = html.escape(res["answer"]).replace("\n", "<br>")
        parts.append(f'<div class="ans"><b>本地模型作答：</b><br>{ans}</div>')
    return "\n".join(parts)


def make_web_handler(chunks, bm25, args):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body, code=200):
            self.send_response(code)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def do_GET(self):
            self._send(WEB_PAGE.replace("{Q}", "").replace("{GEN}", "").replace("{RESULTS}", ""))

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            form = urllib.parse.parse_qs(self.rfile.read(length).decode("utf-8"))
            q = form.get("q", [""])[0].strip()
            use_gen = "generate" in form
            if not q:
                self._send(WEB_PAGE.replace("{Q}", "").replace("{GEN}", "").replace("{RESULTS}", ""))
                return
            res = run_query(chunks, bm25, q, topk=args.topk,
                            generate=use_gen, model=args.model, ollama_url=args.ollama_url)
            page = (WEB_PAGE.replace("{Q}", html.escape(q))
                           .replace("{GEN}", "checked" if use_gen else "")
                           .replace("{RESULTS}", build_results_html(res)))
            self._send(page)

        def log_message(self, *a):  # 静默默认访问日志
            pass

    return Handler


def run_server(chunks, bm25, args):
    Handler = make_web_handler(chunks, bm25, args)
    httpd = http.server.HTTPServer(("", args.port), Handler)
    print(f"本地 RAG Web UI 已启动： http://localhost:{args.port}  （Ctrl+C 退出）")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止。")


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
    ap.add_argument("--interactive", action="store_true", help="交互式问答模式")
    ap.add_argument("--serve", action="store_true", help="启动本地 Web UI（零依赖，浏览器问答）")
    ap.add_argument("--port", type=int, default=8000, help="Web UI 端口")
    args = ap.parse_args()

    chunks, bm25 = build_index(args.doc)
    if not chunks:
        print("未读取到内容块", file=sys.stderr)
        sys.exit(1)

    if args.serve:
        run_server(chunks, bm25, args)
        return
    if args.interactive:
        interactive_mode(chunks, bm25, args)
        return

    queries = []
    if args.questions:
        with open(args.questions, "r", encoding="utf-8") as f:
            queries = [l.strip() for l in f if l.strip()]
    if args.query:
        queries.append(args.query)

    if not queries:
        print("未提供 --query / --questions，且无 --interactive / --serve。")
        print("示例： python rag_qa.py --query \"完播率多少算优秀\"")
        print("       python rag_qa.py --interactive")
        print("       python rag_qa.py --serve")
        ap.print_help()
        return

    for q in queries:
        print_result(run_query(chunks, bm25, q, topk=args.topk,
                               generate=args.generate, model=args.model,
                               ollama_url=args.ollama_url))


if __name__ == "__main__":
    main()
