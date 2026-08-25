from localrag.tokenize import _char_tokenize, get_tokenizer


def test_char_tokenize_chinese_and_english():
    toks = _char_tokenize("完播率 Rate 30%")
    assert "完" in toks and "播" in toks and "率" in toks
    assert "rate" in toks and "30" in toks


def test_get_tokenizer_auto_returns_callable():
    t = get_tokenizer("auto")
    assert callable(t)
    out = t("完播率")
    assert isinstance(out, list) and out


def test_get_tokenizer_char_explicit():
    assert get_tokenizer("char") is _char_tokenize


def test_get_tokenizer_jieba_falls_back_when_missing(monkeypatch):
    # Simulate jieba not installed by making the import inside get_tokenizer fail
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "jieba" or name.startswith("jieba."):
            raise ImportError("simulated missing jieba")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    # jieba segment requested but missing -> falls back to char with a warning
    import sys, io
    captured = io.StringIO()
    monkeypatch.setattr(sys, "stderr", captured)
    t = get_tokenizer("jieba")
    assert t is _char_tokenize
    assert "jieba" in captured.getvalue()
