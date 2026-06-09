# -*- coding: utf-8 -*-
"""Paths for experiment logging.

Defaults to the repository root; override with the PROJECT_DIR env var."""

import os

PROJECT_DIR = os.environ.get(
    "PROJECT_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

SAVE_MODE = "scratch"

SCRATCH_FILE = os.path.join(PROJECT_DIR, "experiments_scratch.parquet")
FINAL_FILE = os.path.join(PROJECT_DIR, "experiments_final.parquet")
