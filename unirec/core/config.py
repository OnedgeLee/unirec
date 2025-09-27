import yaml


class StageCfg(dict):
    """Lightweight stage config (dict-like)."""

    pass


class Config(dict):
    pass


def load_config(path: str) -> Config:
    return Config(yaml.safe_load(open(path, "r", encoding="utf-8")))
