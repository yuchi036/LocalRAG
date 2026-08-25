# -*- coding: utf-8 -*-
"""LocalRAG —— 隐私优先的本地知识库问答（RAG 全链路，完全离线、零依赖）。

把你的 Markdown / 文本 / PDF 笔记变成可本地检索、可问答的助手：
- Retrieval 段：纯标准库 BM25 检索（默认零依赖；可选 jieba 提升中文分词）。
  可选 --hybrid 启用 BM25 + 语义 RRF 融合（sentence-transformers），解决概念型查询。
  sentence-transformers 缺失时自动降级为纯 BM25，零依赖默认不破坏。
- Generation 段：可选接入本机 Ollama 模型基于召回片段作答，数据不出本机。
- 输入：单个文件、或整个目录（递归索引所有 .md/.txt/.pdf）。
- 交互：CLI（--query/--interactive/--serve/--demo）、可量化评估（evaluation）。
- 工程化（S3）：索引持久化秒开、来源行号可定位、Web UI 赞/踩反馈沉淀。
- 可量化与可解释（S4）：Recall@k / MRR / 覆盖率 / 延迟分位的检索质量基线，
  以及「匹配词」解释每条结果为何被召回，把效果从主观感受变成可复现数字。
- 打包与测试（S6）：pyproject.toml 可安装，29 项 pytest 覆盖核心逻辑。
"""
from localrag.cli import main

__version__ = "0.5.0"
__all__ = ["main"]
