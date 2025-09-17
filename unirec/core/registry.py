from importlib import import_module
from typing import Dict, Type
from .interfaces import Component

REGISTRY: Dict[str, Dict[str, Type[Component]]] = {}

def register(kind: str):
    def deco(cls):
        REGISTRY.setdefault(kind, {})[f"{cls.__module__}.{cls.__name__}"] = cls
        return cls
    return deco

def create(kind: str, impl: str, **params) -> Component:
    # Try registry first
    cls = REGISTRY.get(kind, {}).get(impl)
    if cls is None:
        if ":" in impl:
            mod, name = impl.split(":")
        else:
            mod, name = impl.rsplit(".", 1)
        module = import_module(mod)
        cls = getattr(module, name)
    return cls(**params)
