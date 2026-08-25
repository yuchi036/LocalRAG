# -*- coding: utf-8 -*-
"""命令行入口：--query / --questions / --interactive / --serve 与各项参数。"""
import argparse
import sys

from localrag.pipeline import build_index, interactive_mode, print_result, run_query
from localrag.webui import run_server


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="生成式AI短视频内容营销知识库.md",
                    help="知识库：单个 .md/.txt/.pdf 文件，或一个目录（递归索引）")
    ap.add_argument("--questions", default=None, help="问题列表文件，一行一个")
    ap.add_argument("--query", default=None, help="单个问题")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--generate", action="store_true",
                    help="启用本地模型(Ollama)生成答案，实现离线 RAG 闭环")
    ap.add_argument("--model", default="qwen2.5:1.5b", help="Ollama 模型名")
    ap.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama 服务地址")
    ap.add_argument("--segment", default="auto", choices=["auto", "char", "jieba"],
                    help="中文分词：auto(有jieba用jieba,否则字级)/char/jieba")
    ap.add_argument("--interactive", action="store_true", help="交互式问答模式")
    ap.add_argument("--serve", action="store_true", help="启动本地 Web UI（零依赖，浏览器问答）")
    ap.add_argument("--port", type=int, default=8000, help="Web UI 端口")
    args = ap.parse_args()

    chunks, bm25 = build_index(args.doc, segment=args.segment)
    if not chunks:
        print(f"未在 {args.doc} 读取到内容块（检查路径/后缀 .md .txt .pdf）", file=sys.stderr)
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
        print("       python rag_qa.py --doc ./my-notes --serve")
        print("       python rag_qa.py --interactive")
        ap.print_help()
        return

    for q in queries:
        print_result(run_query(chunks, bm25, q, topk=args.topk,
                               generate=args.generate, model=args.model,
                               ollama_url=args.ollama_url))
