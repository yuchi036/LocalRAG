<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/dependencies-zero%20(default)-orange" alt="Dependencies">
  <img src="https://img.shields.io/badge/privacy-offline%20%2F%20local-success" alt="Privacy">
</p>

<h1 align="center">LocalRAG · 隐私优先的本地知识库问答</h1>

<p align="center">
  <b>Turn your Markdown / text / PDF notes into a private, offline, searchable Q&amp;A assistant — zero dependencies by default.</b><br>
  <b>把你的本地笔记（Markdown / 文本 / PDF）变成一个隐私优先、离线、可问答的助手 —— 默认零依赖。</b>
</p>

---

## English

### Why this exists (the problem)
Everyone accumulates scattered local notes — lecture notes, research papers, meeting docs, personal wikis. But asking questions over them usually means either (a) pasting them into a **cloud** AI tool (privacy risk, cost, rate limits) or (b) manually searching. **LocalRAG** lets you run a complete Retrieval-Augmented Generation (RAG) pipeline **entirely on your own machine**, with **no API keys, no cloud, no cost**.

### What it does
- **Retrieval** — a from-scratch BM25 engine in the Python standard library (no external packages). Point it at a single file or an entire folder; it indexes every `.md` / `.txt` / `.pdf` and recalls the most relevant chunks for your question.
- **Generation (optional)** — if you have [Ollama](https://ollama.com) running locally with a model like `qwen2.5:1.5b`, it generates an answer grounded in the retrieved chunks. Data never leaves your machine.
- **Interfaces** — command line, interactive REPL, and a **zero-dependency web UI** (`--serve`) you can open in a browser. The web UI shows 赞/踩 feedback buttons to collect real user signals.
- **Engineered for reuse** — indexes are cached for instant reload (`.localrag_index.json`), every citation points to a source file **and line range** (`L80-87`), and a `--demo` is coming for zero-config showcases.

### Quick start
```bash
# 1) Clone & run (no install needed — Python stdlib only)
git clone https://github.com/yuchi036/LocalRAG.git
cd LocalRAG

# 2) Ask a question over the built-in example knowledge base
python rag_qa.py --query "完播率多少算优秀"

# 3) Or point it at YOUR OWN notes folder and open the web UI
python rag_qa.py --doc ./my-notes --serve
# → open http://localhost:8000
```
> Optional enhancements: `pip install jieba` (better Chinese word segmentation) and `pip install PyPDF2` (read PDFs).

### Optional: local model generation (fully offline)
```bash
ollama pull qwen2.5:1.5b      # one-time
python rag_qa.py --doc ./my-notes --query "项目复盘要注意什么" --generate
```

### Built-in example: an AIGC × short-video content-marketing knowledge base
The repo ships with `生成式AI短视频内容营销知识库.md` — a domain knowledge base covering the full funnel (topic → script → assets → distribution → attribution), metrics, cost models, and compliance. It doubles as both a ready-to-use demo and a showcase of the author's domain expertise.

### Evaluation (honest about limits)
`evaluate.py` measures retrieval quality on a hand-labeled set. Default tokenizer reaches **Recall@3 ≈ 89%**. Importantly, a *conceptual* query like "数字人" still misses under lexical retrieval — even with `jieba` phrase segmentation — which is exactly why **semantic / vector retrieval** (the cloud `ima` variant) is the real fix. This project demonstrates both the lexical baseline and the motivation for going semantic.

### Use cases (who it helps)
- **Students** — chat with your lecture / textbook Markdown.
- **Researchers** — query a folder of paper notes offline.
- **Teams** — turn internal Markdown docs into a private searchable assistant.
- **Privacy-conscious users** — everything stays local; nothing is sent to any server.

### Roadmap
- [ ] Vector/semantic retrieval backend (FAISS / sentence-transformers, optional)
- [ ] Better PDF & DOCX parsing
- [ ] Conversation memory across turns
- [ ] One-command installer / Docker

---

## 中文

### 为什么做这个（痛点）
每个人的电脑里都散落着本地笔记：课程笔记、论文、会议纪要、个人 wiki。但想"问"它们，通常只能 (a) 粘贴到**云端** AI 工具（隐私风险、收费、限流），或 (b) 手动翻找。**LocalRAG** 让你在**自己的机器上**跑完整套 RAG 流程，**无需 API Key、不上云、零费用**。

### 它能做什么
- **检索段**：用 Python 标准库从零实现的 BM25 引擎（无任何第三方包）。指向单个文件或整个目录，自动索引所有 `.md` / `.txt` / `.pdf`，按问题召回最相关片段。
- **生成段（可选）**：若本机运行着 [Ollama](https://ollama.com) 并拉取了如 `qwen2.5:1.5b` 的模型，可基于召回片段生成答案；数据不出本机。
- **多种入口**：命令行、交互式 REPL，以及**零依赖网页 UI**（`--serve`，浏览器直接打开）；网页端每条结果带「赞/踩」按钮，沉淀真实反馈。
- **工程化打磨**：索引自动缓存、二次启动秒开（`.localrag_index.json`）；每条引用标注**来源文件与行号**（`L80-87`），可直接回溯原文。

### 快速开始
```bash
git clone https://github.com/yuchi036/LocalRAG.git
cd LocalRAG
python rag_qa.py --query "完播率多少算优秀"          # 用内置示例问答
python rag_qa.py --doc ./my-notes --serve           # 指向你自己的笔记目录并开网页
# → 浏览器打开 http://localhost:8000
```
> 可选增强：`pip install jieba`（更准的中文分词）、`pip install PyPDF2`（读 PDF）。

### 可选：本地模型生成（完全离线）
```bash
ollama pull qwen2.5:1.5b      # 仅需一次
python rag_qa.py --doc ./my-notes --query "项目复盘要注意什么" --generate
```

### 内置示例：AIGC × 短视频内容营销知识库
仓库自带 `生成式AI短视频内容营销知识库.md` —— 覆盖"选题→脚本→素材→投放→归因"全链路、指标体系、成本模型与合规红线。它既是开箱即用的 Demo，也是作者领域专业度的展示。

### 检索质量评估（坦诚说明局限）
`evaluate.py` 用人工标注集量化检索质量。默认分词 **Recall@3 ≈ 89%**。关键的是，"数字人"这类**概念型**查询在关键词检索下仍会漏召回——即便换用 `jieba` 短语分词也补不回来，这恰恰说明**语义/向量检索**（云端 ima 版）才是真正的补丁。本项目同时展示了"词法检索基线"与"为何要走向语义"的动机。

### 适用人群（帮到谁）
- **学生**：和你的课程/教材 Markdown 对话。
- **研究者**：离线检索一整文件夹论文笔记。
- **团队**：把内部 Markdown 文档变成私有可搜索助手。
- **注重隐私者**：一切在本地，不上传任何服务器。

### 路线图
- [ ] 语义/向量检索后端（FAISS / sentence-transformers，可选）
- [ ] 更强的 PDF / DOCX 解析
- [ ] 多轮对话记忆
- [ ] 一键安装 / Docker

---

## Project structure

```
LocalRAG/
├─ localrag/                       # 核心包（模块化，单文件 <200 行）
│  ├─ bm25.py                      # 纯标准库 BM25 检索（tokenizer 注入，状态可序列化）
│  ├─ ingestion.py                 # 文档加载与分块（.md/.txt/.pdf，目录递归，来源带行号）
│  ├─ tokenize.py                  # 中文分词：零依赖字级 / 可选 jieba
│  ├─ generation.py                # 本地生成段（Ollama，数据不出本机）
│  ├─ pipeline.py                  # 编排：索引构建/持久化、查询、交互模式
│  ├─ persist.py                   # 索引持久化（JSON/pickle，带版本号，二次启动秒开）
│  ├─ feedback.py                  # 赞/踩反馈日志（追加式 JSONL）
│  ├─ webui.py                     # 零依赖 Web UI（深色主题，含赞/踩反馈按钮）
│  ├─ metrics.py                   # 可量化指标（Recall@k/MRR/覆盖率/延迟，分词对比）
│  ├─ evaluation.py                # 检索质量评估（基于 metrics，CLI 报告）
│  ├─ cli.py                       # 命令行入口（rag_qa.py 转发至此）
│  └─ __main__.py                  # 支持 python -m localrag
├─ rag_qa.py / evaluate.py         # 向后兼容入口（转发到 localrag 包）
├─ 生成式AI短视频内容营销知识库.md   # 内置示例知识库（AIGC 内容营销）
├─ questions.txt / demo_qa.md      # 示例问题与两段式问答演练
├─ ima_practice.md / ima_questions.md  # 云端语义检索（ima）实践与对照
├─ requirements.txt                # 可选依赖（jieba / PyPDF2）
├─ LICENSE                         # MIT
└─ README.md
```

## Architecture

```
用户问题 ──► [Retrieval] BM25 从本地知识库召回 Top-K 片段 ──► [Generation] 本地模型基于片段作答（可选）
            （标准库实现，支持文件/目录/PDF，可选 jieba 分词）        （Ollama，数据不出本机）
                      │
            [Persist] 索引落盘(JSON/pickle, 带版本号) → 二次启动秒开；来源标注行号(Lx-y)可回溯
            [Feedback] Web UI 赞/踩 → JSONL 沉淀反馈，反哺检索质量评估
```

## What's new (v0.3.0)
- **Index persistence** — indexes are cached next to the document (`.localrag_index.json`); a second launch loads in milliseconds instead of rebuilding. Use `--index PATH` to control the path, or `--no-cache` to disable. Large corpora can also use `--index cache.pkl` (pickle, faster + smaller).
- **Locatable citations** — every recalled chunk carries a source file and line range (`L80-87`), so you can jump straight back to the original text.
- **Feedback loop** — the web UI shows 赞/踩 (thumbs up / down) buttons on each result; clicks are appended to a JSONL log (`--feedback PATH`, created on first use) for later retrieval-quality analysis.

## What's new (v0.4.0)
- **Quantified retrieval quality** — `evaluate.py` reports **Recall@k, Recall@1, MRR, coverage, and search latency (p50/p95)** on a hand-labeled set. The web homepage shows the live baseline for the built-in KB. `evaluate.py --compare` honestly contrasts `char` vs `jieba` segmentation (skips jieba if not installed).
- **Explainable retrieval** — every result now lists the **matched terms** (which query tokens actually fired), so you can see *why* a chunk was recalled. CLI and web UI both surface this.

## Demo (zero-config)
No arguments needed — it picks the built-in knowledge base, opens the web UI with preset example questions, and works fully offline:
```bash
python -m localrag --demo        # → http://localhost:8000  (click any 示例问题 to ask)
```
This is the fastest way to *see* the product: search + matched-term explanations + live quality baseline + 赞/踩 feedback, all in the browser.

## License
[MIT](LICENSE) — free to use, modify, and distribute.
