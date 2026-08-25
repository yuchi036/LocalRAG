# -*- coding: utf-8 -*-
"""检索-生成编排：构建索引、统一查询入口、结果打印、交互模式。"""
from localrag.bm25 import BM25
from localrag.generation import generate_with_ollama
from localrag.ingestion import load_and_chunk
from localrag.persist import load_index, make_index_path, save_index
from localrag.semantic import hybrid_search
from localrag.tokenize import get_tokenizer


def build_index(doc_path, segment="auto"):
    """构建 BM25 索引并返回 (chunks, bm25)。segment 决定中英文分词方式。"""
    chunks = load_and_chunk(doc_path)
    tok = get_tokenizer(segment)
    bm25 = BM25([tok(c["text"]) for c in chunks], tokenizer=tok)
    return chunks, bm25


def restore_or_build(doc_path, segment="auto", index_path=None, verbose=True):
    """优先从索引加载（文档未变则秒开）；否则构建并落盘。返回 (chunks, bm25, loaded)。

    - index_path 为 None 时不持久化，直接构建（保持 CLI 默认零副作用）。
    - 索引存在但与文档/分词不符时自动重建并覆盖保存。
    """
    tok = get_tokenizer(segment)
    if index_path:
        expected_name = BM25._name_of(tok)
        restored = load_index(index_path, doc_path=doc_path, tokenizer_name=expected_name)
        if restored is not None:
            if verbose:
                print(f"[index] 命中缓存索引：{index_path}（跳过重建）")
            return restored[0], restored[1], True
        chunks, bm25 = build_index(doc_path, segment=segment)
        save_index(index_path, chunks, bm25, doc_path)
        if verbose:
            print(f"[index] 已构建并保存索引：{index_path}")
        return chunks, bm25, False
    chunks, bm25 = build_index(doc_path, segment=segment)
    return chunks, bm25, False


def run_query(chunks, bm25, question, topk=2, generate=False,
              model="qwen2.5:1.5b", ollama_url="http://localhost:11434",
              hybrid=False, semantic_model=None, encoder=None):
    """统一的查询入口：检索（必做）+ 可选本地模型生成，返回结果字典。

    hybrid=True 时走 BM25 + 语义 RRF 融合（需 sentence-transformers；缺失自动降级）。
    匹配词始终基于 BM25 词面解释，与 hybrid 无关，保证「为何召回」的可读性。
    """
    if hybrid:
        hits = hybrid_search(bm25, chunks, question, topk=topk,
                             model_name=semantic_model, encoder=encoder)
    else:
        hits = bm25.search(question, topk=topk)
    result = {"question": question, "hits": [], "answer": None}
    for sc, idx in hits:
        result["hits"].append({
            "title": chunks[idx]["title"],
            "text": chunks[idx]["text"],
            "source": chunks[idx].get("source", ""),
            "loc": chunks[idx].get("loc", ""),
            "score": sc,
            "matched": bm25.matched_terms(question, idx),
        })
    if generate and hits:
        contexts = [(c["title"], c["text"], c["source"]) for c in result["hits"]]
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
        src = f"  · 来源：{h['source']}"
        if h.get("loc"):
            src += f" ({h['loc']})"
        print(f"\n  [{rank}] 命中小节：{h['title']}  (score={h['score']:.3f}){src}")
        if h.get("matched"):
            print("      匹配词：" + " / ".join(h["matched"]))
        snippet = h["text"].replace("\n", " ")
        print("      " + snippet[:240] + ("…" if len(snippet) > 240 else ""))
    if show_answer and result.get("answer") is not None:
        print("\n  [本地模型作答]")
        print("  " + result["answer"].replace("\n", "\n  "))


def interactive_mode(chunks, bm25, args, hybrid=False, semantic_model=None,
                     encoder=None):
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
                               ollama_url=args.ollama_url,
                               hybrid=hybrid, semantic_model=semantic_model,
                               encoder=encoder))
