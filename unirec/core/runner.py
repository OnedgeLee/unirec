from typing import Any
from .config import Config
from .interface import Component
from .state import PipelineState
from .registry import create


def run_pipeline(cfg: Config, context: dict[str, Any]) -> PipelineState:
    state: PipelineState = PipelineState(context=context)
    resources: dict[str, Any] = cfg.get("resources", {})
    for stg in cfg.get("pipeline", []):
        kind: str = stg["kind"]
        impl: str = stg["impl"]
        params: dict[str, Any] = stg.get("params", {})
        comp: Component = create(kind, impl, **params)
        comp.setup(resources)
        # expose stage id so retrievers can log with correct source
        comp.params.setdefault("id", stg.get("id", kind))
        state = comp.run(state)
    return state
