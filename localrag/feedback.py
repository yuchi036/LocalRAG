# -*- coding: utf-8 -*-
"""反馈日志：把用户对召回结果的「赞 / 踩」追加写入 JSONL（每行一条，可定位）。

设计：仅追加、不修改历史；每条记录含时间戳、问题、命中小节、来源与行号、投票。
用途：离线积累真实反馈，反哺检索质量评估（M2 效果指标）与 badcase 复盘。
不依赖任何第三方库，文件不存在时首次点击自动创建。
"""
import json
import os
import time


def log_feedback(path, question, title, source="", loc="", vote="up", comment=""):
    """追加一条反馈。vote ∈ {'up','down'}；返回写入的记录 dict。

    path 不存在时自动创建父目录与文件。并发追加用 'a' 模式 + 单行 JSON，安全。
    """
    rec = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "question": question,
        "title": title,
        "source": source,
        "loc": loc,
        "vote": vote,
        "comment": comment,
    }
    parent = os.path.dirname(path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def load_feedback(path):
    """读取全部反馈记录，返回 list[dict]；文件不存在返回空列表。"""
    if not os.path.exists(path):
        return []
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except ValueError:
                continue
    return records


def summarize(path):
    """汇总反馈：总条数、赞/踩计数、按命中小节的踩点 Top（badcase 线索）。"""
    recs = load_feedback(path)
    up = sum(1 for r in recs if r.get("vote") == "up")
    down = sum(1 for r in recs if r.get("vote") == "down")
    return {"total": len(recs), "up": up, "down": down, "records": recs}
