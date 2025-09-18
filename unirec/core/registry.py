from importlib import import_module
from types import ModuleType
from typing import Any
from .interfaces import Component

REGISTRY: dict[str, dict[str, type[Component]]] = {}


def register(kind: str):
    def deco(comp_type: type[Component]):
        REGISTRY.setdefault(kind, {})[
            f"{comp_type.__module__}.{comp_type.__name__}"
        ] = comp_type
        return comp_type

    return deco


def create(kind: str, impl: str, **params: Any) -> Component:
    # Try registry first
    comp_type: type[Component] | None = REGISTRY.get(kind, {}).get(impl)
    if comp_type is None:
        mod: str = ""
        name: str = ""
        if ":" in impl:
            mod, name = impl.split(":")
        else:
            mod, name = impl.rsplit(".", 1)
        module: ModuleType = import_module(mod)
        comp_type = getattr(module, name)
    return comp_type(**params)
