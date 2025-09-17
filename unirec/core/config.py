from typing import Any, Dict, List, Optional
import json, os

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

class StageCfg(dict):
    """Lightweight stage config (dict-like)."""
    pass

class Config(dict):
    pass

def load_config(path: str) -> Config:
    if path.endswith(".json"):
        obj = json.load(open(path, "r", encoding="utf-8"))
    else:
        if yaml is None:
            raise RuntimeError("PyYAML not installed. Use .json config or install pyyaml.")
        obj = yaml.safe_load(open(path, "r", encoding="utf-8"))
    return Config(obj)
    