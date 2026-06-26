"""App settings (settings.json) — including the configurable models folder.

settings.json lives in the app root and is git-ignored (user-specific). The
models folder is read at import by the engines, so changing it takes effect on
the next restart.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SETTINGS_FILE = ROOT / "settings.json"
DEFAULTS = {"models_dir": str(ROOT / "models")}


def load() -> dict:
    try:
        return {**DEFAULTS, **json.loads(SETTINGS_FILE.read_text())}
    except Exception:
        return dict(DEFAULTS)


def get(key: str, default=None):
    return load().get(key, DEFAULTS.get(key, default))


def update(values: dict) -> dict:
    s = load()
    s.update(values)
    SETTINGS_FILE.write_text(json.dumps(s, indent=2))
    return s


def models_dir() -> Path:
    p = Path(get("models_dir")).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return p
