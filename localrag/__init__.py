# -*- coding: utf-8 -*-
"""LocalRAG —— 隐私优先的本地知识库问答（RAG 全链路，完全离线、零依赖）。

把你的 Markdown / 文本 / PDF 笔记变成可本地检索、可问答的助手：
- Retrieval 段：纯标准库 BM25 检索（默认零依赖；可选 jieba 提升中文分词）。
- Generation 段：可选接入本机 Ollama 模型基于召回片段作答，数据不出本机。
- 输入：单个文件、或整个目录（递归索引所有 .md/.txt/.pdf）。
- 交互：CLI（--query/--interactive/--serve）、可量化评估（evaluation）。
- 工程化（S3）：索引持久化秒开、来源行号可定位、Web UI 赞/踩反馈沉淀。
"""
from localrag.cli import main

__version__ = "0.3.0"
__all__ = ["main"]
