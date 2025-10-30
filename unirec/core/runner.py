from typing import Any
from .config import Config
from .interfaces import Component
from .resources import Resources
from .state import PipelineState
from .registry import create


def run_pipeline(cfg: Config, context: dict[str, Any]) -> PipelineState:
    state: PipelineState = PipelineState(user=context["UserEncodable"], context=context)
    
    # Support both dict (backward compatibility) and Resources
    resources_config = cfg.get("resources", {})
    if isinstance(resources_config, dict):
        # Check if we should convert to Resources object
        # For now, keep backward compatibility by using dict directly
        resources: Resources | dict[str, Any] = resources_config
    else:
        resources = resources_config
    
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
