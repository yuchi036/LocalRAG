# -*- coding: utf-8 -*-
"""检索质量评估：基于人工标注集量化 Recall@k / MRR / 覆盖率 / 延迟。

判定：期望小节出现在 Top-K 标题中即命中（模糊子串匹配）。
S1 修复分词一致性后，char / jieba 的对比结论才可信；用 --compare 看实测差异。

用法：
  python evaluate.py                      # 文本报告（默认分词）
  python evaluate.py --topk 3 --md        # 输出 Markdown 表格
  python evaluate.py --segment jieba      # 用 jieba 分词重测
  python evaluate.py --compare            # 诚实对比 char 与 jieba 指标
"""
import argparse
import sys

from localrag.metrics import (GROUND_TRUTH, compare_segments, evaluate_retrieval,
                              latency_percentiles)
from localrag.pipeline import build_index


def _print_report(m, md=False):
    n = m["n"]
    topk = m["topk"]
    print(f"样本数 = {n}   检索 topk = {topk}")
    print(f"Recall@{topk} (命中率) = {m['recall_at_k']:.0%}")
    print(f"Recall@1            = {m['recall_at_1']:.0%}")
    print(f"MRR                 = {m['mrr']:.3f}")
    print(f"覆盖率(加权)        = {m['coverage']:.0%}")
    lat = m.get("latency")
    if lat and lat["n"]:
        print(f"检索延迟(ms) p50/p95/mean = {lat['p50']:.2f} / {lat['p95']:.2f} / {lat['mean']:.2f}  (n={lat['n']})")
    print()

    if md:
        print("| 问题 | 是否命中 | 首命中位 | Top-3 命中小节 |")
        print("|---|---|---|---|")
        for r in m["rows"]:
            rank = r["rank"] if r["rank"] else "-"
            ok = "✅" if r["hit"] else "❌"
            print(f"| {r['q']} | {ok} | {rank} | {' / '.join(r['titles'])} |")
    else:
        for r in m["rows"]:
            ok = "✅" if r["hit"] else "❌"
            print(f"[{ok}] {r['q']}\n     命中小节: {' / '.join(r['titles'])}")

    miss = m["missed"]
    if miss:
        print("\n未命中样本（BM25 局限 / 或文档确实无对应内容）：")
        for q in miss:
            print("  -", q)
        print("提示：这正是「关键词检索 vs 语义检索」的取舍点——")
        print("      概念型查询（如「数字人」）靠词频/短语匹配都会漏召回；")
        print("      jieba 能提升短语级相关度，但无法理解概念；真正的补丁是语义/向量检索（云端 ima 版）。")
    print("\n结论：BM25 在词面吻合的查询上命中率高；概念型查询需更优分词或语义检索补足。")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="生成式AI短视频内容营销知识库.md")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--segment", default="auto", choices=["auto", "char", "jieba"])
    ap.add_argument("--md", action="store_true", help="输出 Markdown 表格")
    ap.add_argument("--compare", action="store_true", help="对比 char 与 jieba 分词指标")
    args = ap.parse_args()

    if args.compare:
        print("=== 分词对比（同一文档、同一标注集，分词一致性已修复，结论可信）===")
        res = compare_segments(args.doc, args.topk)
        if not res:
            print("无可对比的分词（jieba 未安装且 char 也异常），请检查环境。")
            return
        for seg, m in res.items():
            print(f"\n--- 分词：{seg} ---")
            _print_report(m, md=args.md)
        return

    chunks, bm25 = build_index(args.doc, segment=args.segment)
    m = evaluate_retrieval(chunks, bm25, GROUND_TRUTH, args.topk)
    m["latency"] = latency_percentiles(bm25, [q for q, _ in GROUND_TRUTH], args.topk)
    _print_report(m, md=args.md)


if __name__ == "__main__":
    main()
