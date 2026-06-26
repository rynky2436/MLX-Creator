"""Subprocess runner for Qwen-Image (mflux) — runs the generation on this
process's MAIN thread, where mflux's VAE decode is stable. Reads JSON params on
stdin, prints '__RESULT__<json>' on stdout.
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import qwen_engine  # noqa: E402

params = json.loads(sys.stdin.read())
result = qwen_engine._generate_core(**params)
print("__RESULT__" + json.dumps(result))
