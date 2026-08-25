#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 向后兼容入口：核心实现已迁入 localrag 包，本文件仅做转发。
# 运行方式保持不变：python rag_qa.py --query "..." / --serve / --interactive
from localrag.cli import main

if __name__ == "__main__":
    main()
