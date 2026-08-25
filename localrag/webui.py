# -*- coding: utf-8 -*-
"""零依赖 Web UI（纯标准库 http.server，深色主题，无 emoji 图标，无紫粉渐变）。"""
import html
import http.server
import urllib.parse

from localrag.pipeline import run_query

# 注：UI 遵守 P0 红线——深色主题、无 emoji 功能图标、无紫粉渐变、颜色走设计变量而非硬编码套路。
WEB_PAGE = """<!doctype html><html lang="zh"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LocalRAG · 本地知识库问答</title>
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
  .hit b{color:#7dd3fc}.sc{color:#64748b;font-size:13px;margin-left:6px}.src{color:#475569;font-size:12px;margin-left:6px}
  .sn{color:#b8c0cc;font-size:14px;margin-top:6px}
  .ans{background:#10231a;border:1px solid #1f4d39;border-radius:8px;padding:12px 14px;margin-top:14px;color:#cfe9d8}
  .miss{color:#f87171}
  .tip{color:#64748b;font-size:13px}
</style></head><body>
<h1>LocalRAG · 本地知识库问答（BM25 检索 + 可选本地模型生成）</h1>
<form method="post" action="/">
  <input type="text" name="q" placeholder="输入你的问题，如：完播率多少算优秀、项目复盘要注意什么" value="{Q}">
  <label><input type="checkbox" name="generate" {GEN}> 启用本地模型</label>
  <button type="submit">提问</button>
</form>
<p class="tip">纯标准库实现，离线可跑；勾选「启用本地模型」需本机已启动 Ollama 并拉取 qwen2.5:1.5b。把 --doc 指向你的笔记目录即可问答自己的资料。</p>
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
            src = f'<span class="src">来源：{html.escape(h["source"])}</span>' if h["source"] else ""
            parts.append(
                f'<div class="hit"><b>[{i}] {html.escape(h["title"])}</b>'
                f'<span class="sc">score={h["score"]:.3f}</span>{src}'
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
