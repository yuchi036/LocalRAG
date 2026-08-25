# -*- coding: utf-8 -*-
"""本地生成段：调用本机 Ollama 作答，数据不出本机（零云端依赖）。"""
import json
import urllib.request


def generate_with_ollama(question, contexts, model="qwen2.5:1.5b", base_url="http://localhost:11434"):
    """本地生成段：把召回片段拼成上下文，调用本机 Ollama 模型作答。零云端依赖。"""
    context = "\n\n".join(f"【{title}】（来源：{src}）\n{text}" for title, text, src in contexts)
    prompt = (
        "你是一个严谨的问答助手。请只根据下面提供的「资料」回答问题，"
        "不要使用资料之外的知识；如果资料中没有相关信息，请回答「资料中未提及」。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question}\n\n回答："
    )
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.2, "num_predict": 320},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "").strip()
    except Exception as e:  # noqa: BLE001
        return f"（本地模型调用失败：{e}；请确认 Ollama 已启动且模型已拉取）"
