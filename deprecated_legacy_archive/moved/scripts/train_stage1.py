#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
兼容入口：scripts.v1.train_stage1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.v1.train_stage1 import main


if __name__ == "__main__":
    raise SystemExit(main())
