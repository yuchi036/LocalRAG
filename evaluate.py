#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检索质量评估：用一组人工标注的 (问题, 期望小节) 计算 Recall@k / 命中率。
把"RAG Demo"升级成可量化的小系统。

判定逻辑：期望小节里任一条出现在检索 Top-K 的标题中（模糊子串匹配）即算命中。
刻意保留一条"数字人"样本——它暴露 BM25 在概念型查询上的弱点（与 ima 语义检索对照）。
加 --segment jieba 可观察中文短语分词对"词面吻合"类查询的相关度提升，但"数字人"这类
概念型查询仍会漏召回——这恰好证明**关键词检索存在天花板，只有语义/向量检索能真正补上**
（即本项目云端 ima 版的定位）。

用法：
  python evaluate.py                 # 文本报告（默认分词）
  python evaluate.py --topk 3 --md  # 输出 Markdown 表格
  python evaluate.py --segment jieba  # 用 jieba 分词重测，观察短语级差异
"""

import argparse

from rag_qa import load_and_chunk, get_tokenizer, BM25


GROUND_TRUTH = [
    ("完播率和互动率多少算优秀，为什么重要", ["5.1 内容侧", "4.1 平台算法逻辑"]),
    ("生成式推荐相比传统推荐有什么优势", ["4.2 生成式推荐"]),
    ("AI 生成内容在合规上有哪些风险", ["8. 合规与风险", "3.4 质量门禁"]),
    ("短视频投流应该用什么出价方式，怎么赛马放量", ["4.3 付费投流策略"]),
    ("北极星指标应该选哪个，内容号和带货号有什么不同", ["5.3 北极星指标"]),
    ("单条内容成本和回本周期怎么算，降本增效杠杆在哪", ["7. 成本结构与 ROI 模型"]),
    ("选题有什么方法论", ["6.1 选题方法论"]),
    ("归因模型有哪些，小团队怎么选", ["6.2 归因模型"]),
    # ↓ 刻意保留：字级分词无法把"数字人"当整体概念匹配，预期会漏召回（jieba 可补回）
    ("数字人技术对短视频内容生产有什么价值", ["3.2 素材层"]),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--doc", default="生成式AI短视频内容营销知识库.md")
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--segment", default="auto", choices=["auto", "char", "jieba"])
    ap.add_argument("--md", action="store_true", help="输出 Markdown 表格")
    args = ap.parse_args()

    chunks = load_and_chunk(args.doc)
    tok = get_tokenizer(args.segment)
    # 文档与查询使用同一个 tokenizer 实例，杜绝分词不一致（旧版全局 _SEGMENT 的 BUG 已修）
    bm25 = BM25([tok(c["text"]) for c in chunks], tokenizer=tok)

    hits_at_1 = 0
    hits_at_k = 0
    rows = []
    for q, expected in GROUND_TRUTH:
        res = bm25.search(q, topk=args.topk)
        titles = [chunks[idx]["title"] for _, idx in res]
        hit = any(any(exp in t for exp in expected) for t in titles)
        top1_hit = any(exp in titles[0] for exp in expected) if titles else False
        hits_at_k += int(hit)
        hits_at_1 += int(top1_hit)
        rows.append((q, "✅" if hit else "❌", " / ".join(titles[:3])))

    n = len(GROUND_TRUTH)
    print(f"样本数 = {n}   检索 topk = {args.topk}   分词 = {args.segment}")
    print(f"Recall@{args.topk} (命中率) = {hits_at_k}/{n} = {hits_at_k / n:.0%}")
    print(f"Recall@1            = {hits_at_1}/{n} = {hits_at_1 / n:.0%}")
    print()

    if args.md:
        print("| 问题 | 是否命中 | Top-3 命中小节 |")
        print("|---|---|---|")
        for q, ok, tt in rows:
            print(f"| {q} | {ok} | {tt} |")
    else:
        for q, ok, tt in rows:
            print(f"[{ok}] {q}\n     命中小节: {tt}")

    miss = [q for q, ok, _ in rows if ok == "❌"]
    if miss:
        print("\n未命中样本（BM25 局限 / 或文档确实无对应内容）：")
        for m in miss:
            print("  -", m)
        print("提示：这正是「关键词检索 vs 语义检索」的取舍点——")
        print("      概念型查询（如「数字人」）靠词频/短语匹配都会漏召回；")
        print("      jieba 能提升短语级相关度，但无法理解概念；真正的补丁是语义/向量检索（云端 ima 版）。")
    print("\n结论：BM25 在词面吻合的查询上命中率高；概念型查询需更优分词或语义检索补足。")


if __name__ == "__main__":
    main()
