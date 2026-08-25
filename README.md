# AIGC 短视频内容营销 · 知识库 + 本地 RAG 问答 Demo

> 一个面向「生成式 AI × 短视频内容营销」领域的**可检索知识库**，外加一个**本地 RAG 问答 Demo**——Retrieval 用自研 BM25，Generation 可接本机 Ollama 模型，实现完全离线闭环。
> 选题契合当下 AI 应用层最热方向（AIGC / 视频生成 / 智能投放），且紧贴真实业务（短视频内容生产与买量），可作为简历/面试的项目经历。

---

## 1. 这个项目是什么

- **知识库文档**：`生成式AI短视频内容营销知识库.md` —— 覆盖从选题、脚本生成、素材生产、智能投放到归因复盘的全链路，并整理指标体系、成本模型、合规红线与前沿趋势。
- **RAG 检索层**：`rag_qa.py` —— 纯 Python 标准库实现的 BM25 检索引擎，对 Markdown 分块、中文粗粒度切词、按问题召回 Top-K 相关片段；支持命令行、交互式（`--interactive`）、零依赖网页（`--serve`）、本地模型生成（`--generate`）。**无任何第三方依赖，离线可跑**。
- **检索质量评估**：`evaluate.py` —— 用标注集计算 Recall@k / 命中率，量化检索效果并暴露 BM25 概念弱点。
- **问答演练**：`demo_qa.md` —— 6 个业务问题经「检索命中 → 基于片段作答并标注来源」的两段式流程产出，体现 RAG 的**可追溯**特性。

## 2. RAG 原理（两段式）

```
用户提问 ──► [Retrieval] BM25 从知识库召回相关片段 ──► [Generation] LLM 基于片段作答（标注来源）
```

- **Retrieval（本项目已实现）**：把知识库按标题切分为 chunk，做 BM25 打分召回最相关片段。
- **Generation（由 LLM / 云端知识库完成）**：将召回片段 + 问题送入大模型，生成带引用的答案。
- 价值：回答**锚定在私有知识上**，减少幻觉，且每个结论可追溯来源——这正是企业知识库/RAG 应用的核心卖点。

## 2.1 云端版 RAG 实践：ima 知识库（加分项）

在本地 BM25 版之外，又用 **ima 知识库（腾讯）** 做了云端版 RAG 知识问答，二者对照：

- **本地版**：自己写 BM25 检索（懂原理）。
- **云端版**：通过 ima 知识库 MCP 接口完成 `上传 → 解析 → 语义检索` 全流程——`create_media` 拿 COS 凭证、COS 上传、`add_knowledge` 入库（后台自动切块+向量化）、`search_knowledge` 提问。零代码，语义检索。

实测对照（详见 `ima_practice.md`）：

| 提问 | ima 语义检索 | 本地 BM25 |
|---|---|---|
| 数字人技术对内容生产有什么价值 | ✅ 正确召回 | ❌ 漏掉 3.3 素材层（数字人） |
| 短视频投流出价方式 / 赛马放量 | ✅ 命中 4.3 | ✅ 命中 |
| 完播率多少算优秀 | ✅ 命中 5.1 指标表 | ✅ 命中 |
| 生成式推荐 vs 协同过滤 | ✅ 命中文档 | ✅ 命中 |

**核心洞察**：关键词检索按「词面重合」打分，概念型查询（如「数字人」）易漏召回；语义检索基于向量相似度能正确理解概念——这正是 RAG 里「关键词 vs 语义」检索的取舍。本地版证明「懂原理」，云端版证明「能落地」。

## 3. 本地运行

```bash
# 方式一：跑批检索（默认，零依赖、离线）
python rag_qa.py --doc 生成式AI短视频内容营销知识库.md --questions questions.txt --topk 2

# 方式二：单条提问
python rag_qa.py --doc 生成式AI短视频内容营销知识库.md --query "完播率多少算优秀"

# 方式三：交互式问答（边问边答，输入 exit 退出）
python rag_qa.py --interactive

# 方式四：本地网页问答（浏览器打开 http://localhost:8000，零依赖）
python rag_qa.py --serve

# 检索 + 本地模型生成（离线 RAG 闭环，需先装 Ollama）
# 1) 安装 Ollama 并启动服务  → 2) ollama pull qwen2.5:1.5b  → 3) 运行：
python rag_qa.py --query "完播率多少算优秀" --generate
```

> 方式三/四 的所有检索与生成都在本机完成，不依赖任何外部服务；`--serve` 的网页同样用 Python 标准库实现，无需安装 Flask 等框架。

## 3.1 检索质量评估（把 Demo 升级成可量化系统）

```bash
python evaluate.py            # 文本报告：Recall@k / 命中率
python evaluate.py --topk 3 --md   # 输出 Markdown 表格
```

脚本用一组人工标注的 (问题, 期望小节) 计算检索命中率，并刻意保留「数字人」这类概念型查询以暴露 BM25 的局限（与 ima 语义检索对照）。实测 Recall@3 ≈ 89%，概念型查询需语义检索补足。

## 4. Git 工作流（本项目走过的真实流程）

```bash
git init -b main
git add 生成式AI短视频内容营销知识库.md README.md .gitignore
git commit -m "docs: 初始化 AIGC 短视频内容营销知识库"
git checkout -b rag-demo
git add rag_qa.py questions.txt demo_qa.md
git commit -m "feat: 加入 BM25 本地检索与问答 Demo（RAG 实践）"
git checkout main && git merge rag-demo --no-edit
git remote add origin <你的仓库URL>
git push -u origin main
```

分支策略：`main` 承载文档与说明，`rag-demo` 承载 RAG 代码，合并回主干——一次标准的 feature-branch 工作流。

## 5. 简历 / 面试可用表述

> **生成式 AI 内容营销知识库 & RAG 问答（个人项目）**
> - 围绕 AIGC 短视频内容营销领域，独立整理覆盖「选题—生成—投放—归因」全链路的可检索知识库（含指标体系、成本模型、合规红线）。
> - 用 Python 标准库实现 BM25 检索引擎（零依赖、离线可跑），并接入本机 Ollama 模型（qwen2.5:1.5b）完成生成段，打通「检索召回 + 本地模型生成」两段式 RAG、实现完全离线闭环，产出带来源标注的问答 Demo；并用评估脚本量化检索召回质量（Recall@3≈89%），客观呈现关键词检索在概念型查询上的局限。
> - 另以 ima 知识库（腾讯）做云端版 RAG 实践：通过 API 完成上传、解析与语义问答，对比发现关键词检索在概念型查询上召回偏弱、语义检索更准，从而厘清 RAG 中关键词与语义检索的取舍。
> - 体现了对 RAG 两段式、可追溯性、检索路线取舍的理解，以及与 AI 应用/内容增长业务的结合能力。

## 6. 可选延伸 & 已完成

- ✅ **云端版（已完成）**：通过 ima 知识库 API 完成上传、解析与语义检索，详见 `ima_practice.md`。
- ✅ **生成层本地化（已完成）**：接入本机 Ollama 模型（qwen2.5:1.5b）做 Generation 段，实现「检索 + 生成」全链路**离线闭环**（`rag_qa.py` 的 `--generate`）。运行前 `ollama pull qwen2.5:1.5b` 并启动 Ollama 服务即可，数据不出本机、无 API 费用。
- ✅ **检索质量评估（已完成）**：`evaluate.py` 用标注集计算 Recall@k / 命中率，量化检索效果（实测 Recall@3 ≈ 89%），并把 BM25 在概念型查询上的弱点暴露出来，作为「关键词 vs 语义」取舍的实证。
- ✅ **交互式 & 网页问答（已完成）**：`rag_qa.py` 新增 `--interactive` 与 `--serve`（纯标准库 Web UI），让项目从一个"脚本"变成"可上手把玩的工具"，作品集演示更直观。

---
© 个人学习项目 · 内容为公开的行业知识整理，不含任何内部/涉密信息。
