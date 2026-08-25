# -*- coding: utf-8 -*-
"""零依赖 Web UI（纯标准库 http.server，深色主题，无 emoji 图标，无紫粉渐变）。

S3 增强：每条召回结果附带「赞 / 踩」反馈按钮（SVG 图标，非 emoji），
点击后通过 /feedback 路由将记录追加写入 JSONL；命中来源显示文件名与行号(loc)。
"""
import html
import http.server
import json
import os
import urllib.parse

from localrag.feedback import log_feedback
from localrag.pipeline import run_query

# 注：UI 遵守 P0 红线——深色主题、无 emoji 功能图标、无紫粉渐变、颜色走设计变量而非硬编码套路。
# 反馈图标使用锁定 SVG（currentColor 跟随按钮文字色），杜绝 emoji。
SVG_UP = ('<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
          'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
          '<path d="M7 10v11"/><path d="M7 10l4-7a2 2 0 0 1 2 2v5h5a2 2 0 0 1 2 2.3l-1.2 6A2 2 0 0 1 18 21H7"/></svg>')
SVG_DOWN = ('<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M17 14V3"/><path d="M17 14l-4 7a2 2 0 0 1-2-2v-5H6a2 2 0 0 1-2-2.3l1.2-6A2 2 0 0 1 6 3h11"/></svg>')

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
  .metrics{background:#161a21;border:1px solid #2a2f3a;border-radius:8px;padding:10px 14px;margin:14px 0;font-size:13px;color:#b8c0cc}
  .metrics b{color:#e6e6e6}
  .mt{color:#475569;font-size:12px;margin-top:4px}
  .chips{display:flex;flex-wrap:wrap;gap:8px;align-items:center;margin:12px 0}
  .chip-tip{font-size:13px;color:#64748b}
  .chip{padding:6px 12px;border:1px solid #2a2f3a;border-radius:999px;background:#161a21;color:#b8c0cc;font-size:13px;cursor:pointer}
  .chip:hover{border-color:#3b82f6;color:#e6e6e6}
  .fbrow{display:flex;gap:8px;align-items:center;margin-top:8px}
  .fb-tip{font-size:12px;color:#64748b}
  .fb{display:inline-flex;align-items:center;gap:4px;padding:4px 10px;border:1px solid #2a2f3a;border-radius:6px;background:#161a21;color:#b8c0cc;font-size:13px;cursor:pointer}
  .fb:hover{border-color:#3b82f6;color:#e6e6e6}
  .fb:disabled{opacity:.5;cursor:default}
  .fb-done{font-size:12px;color:#34d399}
</style></head><body>
<h1>LocalRAG · 本地知识库问答（BM25 检索 + 可选本地模型生成）</h1>
<form method="post" action="/">
  <input type="text" name="q" placeholder="输入你的问题，如：完播率多少算优秀、项目复盘要注意什么" value="{Q}">
  <label><input type="checkbox" name="generate" {GEN}> 启用本地模型</label>
  <button type="submit">提问</button>
</form>
{DEMO}
<p class="tip">纯标准库实现，离线可跑；勾选「启用本地模型」需本机已启动 Ollama 并拉取 qwen2.5:1.5b。把 --doc 指向你的笔记目录即可问答自己的资料。对结果点「赞/踩」可沉淀反馈，反哺检索质量。</p>
{METRICS}
<hr>
{RESULTS}
<script>
document.querySelectorAll('button.chip').forEach(function(c){
  c.addEventListener('click', function(){
    var inp = document.querySelector('input[name=q]');
    if(inp){ inp.value = c.dataset.q; inp.form.submit(); }
  });
});
document.querySelectorAll('.fbrow').forEach(function(row){
  row.addEventListener('click', function(e){
    var b = e.target.closest('button.fb'); if(!b || b.disabled) return;
    var d = b.dataset;
    var fd = new URLSearchParams();
    fd.set('q', d.q); fd.set('title', d.title); fd.set('source', d.source); fd.set('loc', d.loc); fd.set('vote', d.vote);
    fetch('/feedback', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body: fd.toString()})
      .then(function(r){return r.json();})
      .then(function(){
        row.querySelectorAll('button.fb').forEach(function(x){x.disabled=true;});
        var s=document.createElement('span'); s.className='fb-done'; s.textContent='已记录';
        row.appendChild(s);
      }).catch(function(){});
  });
});
</script>
</body></html>"""


def build_results_html(res):
    parts = [f'<div class="q">问：{html.escape(res["question"])}</div>']
    if not res["hits"]:
        parts.append('<div class="miss">未命中相关片段</div>')
    else:
        q = html.escape(res["question"])
        for i, h in enumerate(res["hits"], 1):
            snippet = html.escape(h["text"].replace("\n", " "))[:240]
            src = h.get("source", "")
            loc = h.get("loc", "")
            src_bits = []
            if src:
                src_bits.append(f"来源：{html.escape(src)}")
            if loc:
                src_bits.append(html.escape(loc))
            src_span = f'<span class="src">{" · ".join(src_bits)}</span>' if src_bits else ""
            title = html.escape(h["title"])
            src_esc = html.escape(src)
            loc_esc = html.escape(loc)
            parts.append(
                f'<div class="hit"><b>[{i}] {title}</b>'
                f'<span class="sc">score={h["score"]:.3f}</span>{src_span}'
                f'<div class="sn">{snippet}</div>'
                + (f'<div class="mt">匹配词：{" / ".join(html.escape(t) for t in h.get("matched", []))}</div>'
                   if h.get("matched") else "")
                + f'<div class="fbrow"><span class="fb-tip">这条是否有用？</span>'
                f'<button type="button" class="fb" data-vote="up" data-q="{q}" data-title="{title}" '
                f'data-source="{src_esc}" data-loc="{loc_esc}">{SVG_UP}有用</button>'
                f'<button type="button" class="fb" data-vote="down" data-q="{q}" data-title="{title}" '
                f'data-source="{src_esc}" data-loc="{loc_esc}">{SVG_DOWN}没用</button></div></div>'
            )
    if res.get("answer"):
        ans = html.escape(res["answer"]).replace("\n", "<br>")
        parts.append(f'<div class="ans"><b>本地模型作答：</b><br>{ans}</div>')
    return "\n".join(parts)


# 零配置演示的预设问题（点击即问，无需用户自己构思）
DEMO_QUESTIONS = [
    "完播率多少算优秀",
    "选题有什么方法论",
    "短视频投流应该用什么出价方式",
    "北极星指标应该选哪个",
    "AI 生成内容在合规上有哪些风险",
]


def build_demo_html(args):
    """演示模式：渲染预设问题为可点击 chip；非演示模式返回空。"""
    if not getattr(args, "demo", False):
        return ""
    chips = "".join(
        f'<button type="button" class="chip" data-q="{html.escape(q)}">{html.escape(q)}</button>'
        for q in DEMO_QUESTIONS
    )
    return f'<div class="chips"><span class="chip-tip">示例问题（点击直接提问）：</span>{chips}</div>'


def build_metrics_html(args, chunks, bm25):
    """首页基线指标面板：仅对内置示例库有意义（有标注集），其余文档不展示以免误导。"""
    doc = getattr(args, "doc", None)
    if not doc or os.path.basename(doc) != "生成式AI短视频内容营销知识库.md":
        return ""
    try:
        from localrag.metrics import (GROUND_TRUTH, evaluate_retrieval,
                                      latency_percentiles)
        topk = getattr(args, "topk", 3)
        m = evaluate_retrieval(chunks, bm25, GROUND_TRUTH, topk)
        lat = latency_percentiles(bm25, [q for q, _ in GROUND_TRUTH], topk)
    except Exception:  # noqa: BLE001
        return ""
    return (
        f'<div class="metrics"><b>检索质量基线（内置示例库，Top-{m["topk"]}）</b><br>'
        f'Recall@{m["topk"]} = {m["recall_at_k"]:.0%} · Recall@1 = {m["recall_at_1"]:.0%} · '
        f'MRR = {m["mrr"]:.3f} · 覆盖率 = {m["coverage"]:.0%}<br>'
        f'<span class="tip">检索延迟 p50 {lat["p50"]:.2f}ms / p95 {lat["p95"]:.2f}ms（本地 BM25，离线）</span></div>'
    )


def make_web_handler(chunks, bm25, args):
    feedback_path = getattr(args, "feedback", None)
    metrics_html = build_metrics_html(args, chunks, bm25)
    demo_html = build_demo_html(args)

    class Handler(http.server.BaseHTTPRequestHandler):
        def _send(self, body, code=200, content_type="text/html; charset=utf-8"):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.end_headers()
            self.wfile.write(body.encode("utf-8"))

        def _page(self, q="", gen="", results=""):
            return (WEB_PAGE.replace("{Q}", q)
                           .replace("{GEN}", gen)
                           .replace("{RESULTS}", results)
                           .replace("{METRICS}", metrics_html)
                           .replace("{DEMO}", demo_html))

        def do_GET(self):
            self._send(self._page())

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode("utf-8")
            if self.path == "/feedback":
                self._handle_feedback(raw)
                return
            form = urllib.parse.parse_qs(raw)
            q = form.get("q", [""])[0].strip()
            use_gen = "generate" in form
            if not q:
                self._send(self._page())
                return
            res = run_query(chunks, bm25, q, topk=args.topk,
                            generate=use_gen, model=args.model, ollama_url=args.ollama_url)
            page = self._page(html.escape(q), "checked" if use_gen else "",
                              build_results_html(res))
            self._send(page)

        def _handle_feedback(self, raw):
            form = urllib.parse.parse_qs(raw)
            q = form.get("q", [""])[0]
            title = form.get("title", [""])[0]
            source = form.get("source", [""])[0]
            loc = form.get("loc", [""])[0]
            vote = form.get("vote", ["up"])[0]
            if feedback_path:
                rec = log_feedback(feedback_path, q, title, source, loc, vote)
            else:
                rec = {"ok": False, "reason": "feedback disabled"}
            self._send(json.dumps({"ok": True, "rec": rec}, ensure_ascii=False),
                       content_type="application/json; charset=utf-8")

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
