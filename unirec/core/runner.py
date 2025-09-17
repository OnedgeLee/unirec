from typing import Dict, Any
from .config import Config
from .state import PipelineState
from .registry import create

def run_pipeline(cfg: Config, context: Dict[str, Any]) -> PipelineState:
    state = PipelineState(context=context)
    resources = cfg.get("resources", {})
    for stg in cfg.get("pipeline", []):
        kind = stg["kind"]
        impl = stg["impl"]
        params = stg.get("params", {})
        comp = create(kind, impl, **params)
        comp.setup(resources)
        # expose stage id so retrievers can log with correct source
        comp.params.setdefault("id", stg.get("id", kind))
        state = comp.run(state)
    return state
