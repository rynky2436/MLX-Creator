"""Subprocess runner for Wan video (mlx-video). Isolates the generation so all
its memory is reclaimed by the OS on exit. Reads JSON params on stdin, prints
'__RESULT__<json>' on stdout.
"""
import json
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import video_engine  # noqa: E402

params = json.loads(sys.stdin.read())
result = video_engine._generate_core(**params)
print("__RESULT__" + json.dumps(result))
