import importlib
import os
import sys

from localrag import __version__, cli


def test_package_version_is_string():
    assert isinstance(__version__, str) and __version__


def test_cli_main_is_callable():
    assert callable(cli.main)


def test_backward_compat_shims_import():
    # These shims must import cleanly and expose a runnable main
    rg = importlib.import_module("rag_qa")
    ev = importlib.import_module("evaluate")
    assert callable(rg.main)
    assert callable(ev.main)


def test_builtin_kb_resolution():
    # builtin_kb_path must return a string (existing file in repo, or fallback name)
    from localrag.cli import builtin_kb_path
    p = builtin_kb_path()
    assert isinstance(p, str) and p.endswith(".md")
