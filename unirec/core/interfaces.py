from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Tuple, Iterable
from .state import PipelineState, Candidate, CandidateSet, Slate

# -------- Shared data structures for policy outputs (for OPE/logging) --------
@dataclass
class PerItemDecision:
    item_id: int
    mu: float                      # predicted mean reward (e.g., CTR)
    sigma: float = 0.0             # uncertainty (UCB/TS etc.)
    score: float = 0.0             # internal score used for ranking
    propensity: float = 0.0        # pi_e(a|s): probability of being chosen at its slot (if stochastic)
    features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyOutput:
    slate: Slate
    slot_propensity: List[float]           # length = |slate.items| ; pi_e of each chosen action at each slot
    per_item: List[PerItemDecision]        # aligned to slate.items
    aux: Dict[str, Any] = field(default_factory=dict)

# -------- Base component interface --------
class Component(ABC):
    """Base class for all pipeline components.

    Lifecycle:
      - setup(resources): one-time init, load models/index/artifacts
      - run(state): consume & produce specific fields according to contract below
      - close(): optional teardown
    """
    component_kind: str = "component"

    def __init__(self, **params):
        self.params = params
        self.resources: Dict[str, Any] = {}
        self.id: str = params.get("id", self.__class__.__name__)

    def setup(self, resources: Dict[str, Any]):
        self.resources = resources

    def close(self):  # pragma: no cover
        pass

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState:
        ...

# -------- Optional mixins for training & index management --------
class Trainable(ABC):
    @abstractmethod
    def fit(self, dataset_spec: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """Train the component. Returns artifact metadata (paths, metrics)."""
        ...

    def load_artifacts(self, **kwargs):  # pragma: no cover
        pass

    def save_artifacts(self, **kwargs):  # pragma: no cover
        pass

class IndexOps(ABC):
    def add_items(self, items: Iterable[Tuple[int, Any]]):  # (item_id, vector/feat)
        raise NotImplementedError
    def update_items(self, items: Iterable[Tuple[int, Any]]):
        raise NotImplementedError
    def remove_items(self, item_ids: Iterable[int]):
        raise NotImplementedError
    def index_stats(self) -> Dict[str, Any]:
        return {}

# -------- Concrete contracts per role --------
class Retriever(Component, IndexOps):
    component_kind = "retriever"
    @abstractmethod
    def search_one(self, state: PipelineState, k: int) -> List[Candidate]:
        """Return a list of Candidate for the current user context."""
        ...

    def run(self, state: PipelineState) -> PipelineState:
        k = int(self.params.get("topk", 500))
        cands = self.search_one(state, k=k)
        # Contract: store under logs['retrieval'][self.id]
        state.logs.setdefault("retrieval", {})[self.id] = cands
        return state

class Merger(Component):
    component_kind = "merger"
    @abstractmethod
    def merge(self, pools: Dict[str, List[Candidate]], user_id: int, topk: int) -> CandidateSet:
        ...

    def run(self, state: PipelineState) -> PipelineState:
        pools = state.logs.get("retrieval", {})
        topk = int(self.params.get("topk", 800))
        user_id = state.context.get("user_id", -1)
        state.candset = self.merge(pools, user_id=user_id, topk=topk)
        return state

class Ranker(Component):
    component_kind = "ranker"
    @abstractmethod
    def select_slate(self, state: PipelineState) -> PolicyOutput:
        """Consumes state.candset and returns PolicyOutput with slate + propensities."""
        ...

    def run(self, state: PipelineState) -> PipelineState:
        out = self.select_slate(state)
        state.slate = out.slate
        # Contract: store policy diagnostics for OPE/logging
        state.logs.setdefault("policy", {})[self.id] = out
        return state

class Reranker(Component):
    component_kind = "reranker"
    @abstractmethod
    def rerank(self, state: PipelineState) -> CandidateSet:
        ...

    def run(self, state: PipelineState) -> PipelineState:
        state.candset = self.rerank(state)
        return state

class Evaluator(Component):
    component_kind = "evaluator"
    @abstractmethod
    def evaluate(self, state: PipelineState) -> Dict[str, Any]:
        ...

    def run(self, state: PipelineState) -> PipelineState:
        report = self.evaluate(state)
        state.logs.setdefault("reports", {}).update(report)
        return state

class OPEEstimator(Component):
    component_kind = "ope"
    @abstractmethod
    def estimate(self, state: PipelineState) -> Dict[str, Any]:
        ...

    def run(self, state: PipelineState) -> PipelineState:
        report = self.estimate(state)
        state.logs.setdefault("ope", {}).update(report)
        return state
