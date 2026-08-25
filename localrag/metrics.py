# -*- coding: utf-8 -*-
"""可量化检索指标：Recall@k、MRR、覆盖率、延迟分位与基线评估。

把"检索好不好"从主观感受变成可复现的数字，支撑效果对比（char vs jieba）
与简历说服力；Web UI 首页也复用本模块展示基线，做到透明可量化。
"""
import time

from localrag.bm25 import BM25
from localrag.ingestion import load_and_chunk
from localrag.tokenize import get_tokenizer

# 人工标注集：(问题, 期望命中的小节标题列表)
# 刻意保留一条"数字人"样本——它暴露 BM25 在概念型查询上的弱点（与 ima 语义检索对照）。
GROUND_TRUTH = [
    ("完播率和互动率多少算优秀，为什么重要", ["5.1 内容侧", "4.1 平台算法逻辑"]),
    ("生成式推荐相比传统推荐有什么优势", ["4.2 生成式推荐"]),
    ("AI 生成内容在合规上有哪些风险", ["8. 合规与风险", "3.4 质量门禁"]),
    ("短视频投流应该用什么出价方式，怎么赛马放量", ["4.3 付费投流策略"]),
    ("北极星指标应该选哪个，内容号和带货号有什么不同", ["5.3 北极星指标"]),
    ("单条内容成本和回本周期怎么算，降本增效杠杆在哪", ["7. 成本结构与 ROI 模型"]),
    ("选题有什么方法论", ["6.1 选题方法论"]),
    ("归因模型有哪些，小团队怎么选", ["6.2 归因模型"]),
    ("数字人技术对短视频内容生产有什么价值", ["3.2 素材层"]),
]


def _title_hit(expected, titles):
    """期望小节中任一条出现在检索标题里即算命中（模糊子串匹配）。"""
    return any(any(exp in t for exp in expected) for t in titles)


def mrr(per_query_rank):
    """平均倒数排名：每条查询首个命中位置的倒数，miss 记 0。"""
    if not per_query_rank:
        return 0.0
    return sum(1.0 / r for r in per_query_rank if r > 0) / len(per_query_rank)


def latency_percentiles(bm25, queries, topk, warmup=3):
    """对一组查询测检索延迟（毫秒），返回 p50/p95/mean 与样本数。"""
    # 先 warmup，避免首查冷启动干扰分位估计
    for _ in range(warmup):
        if queries:
            bm25.search(queries[0], topk=topk)
    samples = []
    for q in queries:
        t0 = time.perf_counter()
        bm25.search(q, topk=topk)
        samples.append((time.perf_counter() - t0) * 1000.0)
    if not samples:
        return {"p50": 0.0, "p95": 0.0, "mean": 0.0, "n": 0}
    samples.sort()
    n = len(samples)

    def pct(p):
        i = min(n - 1, int(round((p / 100) * (n - 1))))
        return samples[i]

    return {"p50": pct(50), "p95": pct(95), "mean": sum(samples) / n, "n": n}


def evaluate_retrieval(chunks, bm25, ground_truth, topk):
    """对标注集跑检索，返回结构化指标 + 逐条结果（供 CLI / Web 复用）。"""
    per_hit = []
    per_rank = []
    per_coverage = []
    rows = []
    for q, expected in ground_truth:
        res = bm25.search(q, topk=topk)
        titles = [chunks[idx]["title"] for _, idx in res]
        hit = _title_hit(expected, titles)
        rank = 0
        for r, t in enumerate(titles, 1):
            if _title_hit(expected, [t]):
                rank = r
                break
        found = sum(1 for exp in expected if any(exp in t for t in titles))
        per_hit.append(hit)
        per_rank.append(rank)
        per_coverage.append(found / len(expected))
        rows.append({"q": q, "hit": hit, "rank": rank,
                     "found": found, "expected": len(expected), "titles": titles[:3]})
    n = len(ground_truth)
    missed = [r["q"] for r in rows if not r["hit"]]
    return {
        "n": n,
        "topk": topk,
        "recall_at_k": sum(per_hit) / n,
        "recall_at_1": sum(1 for r in per_rank if r == 1) / n,
        "mrr": mrr(per_rank),
        "coverage": sum(per_coverage) / n,  # 期望答案的加权覆盖率（更细粒度）
        "rows": rows,
        "missed": missed,
    }


def compare_segments(doc_path, topk, segments=("char", "jieba")):
    """诚实对比不同分词下的检索指标；jieba 未安装则跳过并标注。"""
    import sys
    out = {}
    chunks = load_and_chunk(doc_path)
    jieba_ok = False
    try:
        import jieba  # noqa: F401
        jieba_ok = True
    except ImportError:
        pass
    for seg in segments:
        if seg == "jieba" and not jieba_ok:
            print("[skip] jieba 未安装，跳过 jieba 对比（可 pip install jieba）", file=sys.stderr)
            continue
        tok = get_tokenizer(seg)
        name = BM25._name_of(tok)
        bm25 = BM25([tok(c["text"]) for c in chunks], tokenizer=tok)
        lat = latency_percentiles(bm25, [q for q, _ in GROUND_TRUTH], topk)
        m = evaluate_retrieval(chunks, bm25, GROUND_TRUTH, topk)
        m["latency"] = lat
        m["segment"] = name
        out[name] = m
    return out
