# -*- coding: utf-8 -*-
"""可选语义检索：通过 RRF 与 BM25 融合；sentence-transformers 缺失时自动降级。

设计目标：
- 零依赖：核心 RRF 与相似度用纯 Python，不引入 numpy。
- 懒加载：sentence-transformers 仅在真正编码时才尝试导入，未安装即降级。
- 可测试：encoder 抽象出来，测试可注入假 encoder 验证融合逻辑。
"""
import sys


_RRF_K = 60  # RRF 标准常数 k


def is_available():
    """sentence-transformers 是否已安装（探测，不实际加载模型）。"""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def rrf_fuse(bm25_hits, sem_ranked_idx, k=_RRF_K):
    """Reciprocal Rank Fusion：把两个排名列表融合为单一排序。

    bm25_hits     : [(bm25_score, idx), ...]  按 bm25 分数降序
    sem_ranked_idx: [idx, ...]                 按语义相似度降序
    返回          : [(rrf_score, idx), ...]    按 RRF 分数降序
    """
    ranks = {}
    for rank, (_, idx) in enumerate(bm25_hits, 1):
        ranks[idx] = ranks.get(idx, 0.0) + 1.0 / (k + rank)
    for rank, idx in enumerate(sem_ranked_idx, 1):
        ranks[idx] = ranks.get(idx, 0.0) + 1.0 / (k + rank)
    fused = sorted(ranks.items(), key=lambda x: x[1], reverse=True)
    return [(sc, idx) for idx, sc in fused]


def _as_list(vec):
    """兼容 numpy 数组与原生 list 的统一取列表方法。"""
    return vec.tolist() if hasattr(vec, "tolist") else vec


def _dot(a, b):
    """纯 Python 点积；要求输入已归一化（sentence-transformers 默认归一化）。"""
    s = 0.0
    for x, y in zip(a, b):
        s += x * y
    return s


def _build_default_encoder(model_name=None):
    """懒加载 sentence-transformers；未安装或模型加载失败时抛 ImportError/RuntimeError。"""
    import sentence_transformers  # noqa: F401
    from sentence_transformers import SentenceTransformer
    name = model_name or "shibing624/text2vec-base-chinese"
    return SentenceTransformer(name)


def hybrid_search(bm25, chunks, query, topk=3, encoder=None, model_name=None,
                  overfetch=10):
    """BM25 + 语义 RRF 融合检索。

    encoder 缺省时尝试构建默认语义编码器；任何失败（未安装/模型加载/编码异常）
    都降级为纯 BM25 并向 stderr 打印说明，保证零依赖时仍能正常工作。
    """
    bm25_hits = bm25.search(query, topk=max(topk * 3, overfetch))
    enc = encoder
    if enc is None:
        try:
            enc = _build_default_encoder(model_name)
        except Exception as e:  # ImportError, OSError, RuntimeError, etc.
            print(f"[hybrid] 语义检索不可用，已降级为纯 BM25（{type(e).__name__}：{e}）",
                  file=sys.stderr)
            return bm25_hits[:topk]
    try:
        texts = [c["text"] for c in chunks]
        q_vec = _as_list(enc.encode([query], convert_to_numpy=True)[0])
        d_vecs = [_as_list(v) for v in enc.encode(texts, convert_to_numpy=True)]
        sims = [_dot(d, q_vec) for d in d_vecs]
        sem_ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        fused = rrf_fuse(bm25_hits, sem_ranked)
        return [(sc, idx) for sc, idx in fused[:topk]]
    except Exception as e:
        print(f"[hybrid] 语义编码失败，已降级为纯 BM25（{type(e).__name__}：{e}）",
              file=sys.stderr)
        return bm25_hits[:topk]
