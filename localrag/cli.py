# -*- coding: utf-8 -*-
"""命令行入口：--query / --questions / --interactive / --serve / --demo 与各项参数。"""
import argparse
import os
import sys

from localrag.persist import make_index_path
from localrag.pipeline import (interactive_mode, print_result, restore_or_build,
                               run_query)
from localrag.webui import run_server

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PKG_DIR)
BUILTIN_KB = "生成式AI短视频内容营销知识库.md"


def builtin_kb_path():
    """解析内置示例库路径：优先仓库根目录，否则回退到当前目录同名文件。"""
    cand = os.path.join(REPO_ROOT, BUILTIN_KB)
    return cand if os.path.exists(cand) else BUILTIN_KB


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
    ap.add_argument("--demo", action="store_true",
                    help="零配置演示：内置示例库 + 预设问题 + 网页 UI（无需任何参数）")
    ap.add_argument("--port", type=int, default=8000, help="Web UI 端口")
    ap.add_argument("--index", default=None,
                    help="索引缓存路径；默认在文档旁生成 .localrag_index.json 实现秒开")
    ap.add_argument("--no-cache", action="store_true", help="禁用索引落盘（每次启动重建）")
    ap.add_argument("--feedback", default=".localrag_feedback.jsonl",
                    help="Web UI 反馈日志(JSONL)路径，首次点击自动创建")
    args = ap.parse_args()

    if args.demo:
        # 零配置：强制使用内置示例库并打开网页（预设问题由 Web UI 渲染）
        args.doc = builtin_kb_path()
        args.serve = True

    index_path = None if args.no_cache else (args.index or make_index_path(args.doc))
    chunks, bm25, loaded = restore_or_build(args.doc, segment=args.segment, index_path=index_path)
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
