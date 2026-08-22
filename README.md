# AIGC 短视频内容营销 · 知识库 + 本地 RAG 问答 Demo

> 一个面向「生成式 AI × 短视频内容营销」领域的**可检索知识库**，外加一个**零依赖的本地 RAG 问答 Demo**。
> 选题契合当下 AI 应用层最热方向（AIGC / 视频生成 / 智能投放），且紧贴真实业务（短视频内容生产与买量），可作为简历/面试的项目经历。

---

## 1. 这个项目是什么

- **知识库文档**：`生成式AI短视频内容营销知识库.md` —— 覆盖从选题、脚本生成、素材生产、智能投放到归因复盘的全链路，并整理指标体系、成本模型、合规红线与前沿趋势。
- **RAG 检索层**：`rag_qa.py` —— 纯 Python 标准库实现的 BM25 检索引擎，对 Markdown 分块、中文粗粒度切词、按问题召回 Top-K 相关片段。**无任何第三方依赖，离线可跑**。
- **问答演练**：`demo_qa.md` —— 6 个业务问题经「检索命中 → 基于片段作答并标注来源」的两段式流程产出，体现 RAG 的**可追溯**特性。

## 2. RAG 原理（两段式）

```
用户提问 ──► [Retrieval] BM25 从知识库召回相关片段 ──► [Generation] LLM 基于片段作答（标注来源）
```

- **Retrieval（本项目已实现）**：把知识库按标题切分为 chunk，做 BM25 打分召回最相关片段。
- **Generation（由 LLM / 云端知识库完成）**：将召回片段 + 问题送入大模型，生成带引用的答案。
- 价值：回答**锚定在私有知识上**，减少幻觉，且每个结论可追溯来源——这正是企业知识库/RAG 应用的核心卖点。

## 3. 本地运行

```bash
# 需要 Python 3，无需安装任何包
python rag_qa.py --doc 生成式AI短视频内容营销知识库.md --questions questions.txt --topk 2

# 或单条提问
python rag_qa.py --doc 生成式AI短视频内容营销知识库.md --query "完播率多少算优秀"
```

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

> **生成式 AI 内容营销知识库 & 本地 RAG 问答（个人项目）**
> - 围绕 AIGC 短视频内容营销领域，独立整理覆盖「选题—生成—投放—归因」全链路的可检索知识库（含指标体系、成本模型、合规红线）。
> - 用 Python 标准库实现 BM25 检索引擎（零依赖、离线可跑），打通「检索召回 + LLM 生成」两段式 RAG，产出带来源标注的问答 Demo。
> - 体现了对 RAG 可追溯性的理解，以及与 AI 应用/内容增长业务的结合能力。

## 6. 可选延伸

- **云端版**：把知识库文档接入 ima 知识库等云端 RAG，体验端到端托管问答。
- **生成层本地化**：接入本地大模型（如 Ollama）完成 Generation 段，实现完全离线闭环。
- **评估**：用命中率/相关性打分量化检索效果，把 Demo 升级成可评估的小系统。

---
© 个人学习项目 · 内容为公开的行业知识整理，不含任何内部/涉密信息。
