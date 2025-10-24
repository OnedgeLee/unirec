from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from collections.abc import Iterable
from torch import Tensor
from typing import Any, ClassVar, Generic, Mapping, TypeVar, final
from .fingerprint import Fingerprintable
from .state import PipelineState, Candidate, CandidateSet, Slate
from .version import Versioned


# -------- Shared data structures for policy outputs (for OPE/logging) --------
@dataclass
class PerItemDecision:
    item_id: int
    mu: float  # predicted mean reward (e.g., CTR)
    sigma: float = 0.0  # uncertainty (UCB/TS etc.)
    score: float = 0.0  # internal score used for ranking
    propensity: float = (
        0.0  # pi_e(a|s): probability of being chosen at its slot (if stochastic)
    )
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class PolicyOutput:
    slate: Slate
    slot_propensity: list[
        float
    ]  # length = |slate.items| ; pi_e of each chosen action at each slot
    per_item: list[PerItemDecision]  # aligned to slate.items
    aux: dict[str, Any] = field(default_factory=dict)


# -------- Base component interface --------
class Component(ABC):
    """Base class for all pipeline components.

    Lifecycle:
      - setup(resources): one-time init, load models/index/artifacts
      - run(state): consume & produce specific fields according to contract below
      - close(): optional teardown
    """

    component_kind: ClassVar[str] = "component"

    def __init__(self, **params: Any):
        self.params: dict[str, Any] = params
        self.resources: dict[str, Any] = {}
        self.id: str = params.get("id", self.__class__.__name__)

    @final
    def require_param(
        self, key: str, expected: type, default_value: Any | None = None
    ) -> Any:
        if key in self.params:
            val = self.params[key]
        elif default_value is not None:
            self.params[key] = default_value
            val = default_value
        else:
            raise KeyError(f"{type(self).__name__}: missing required param '{key}'")

        if not isinstance(val, expected):
            raise TypeError(
                f"{self.__class__.__name__}: param '{key}' must be {expected}, got {type(val).__name__}"
            )
        return val

    @final
    def optional_param(self, key: str, expected: type) -> Any | None:
        if key not in self.params:
            return None
        val = self.params.get(key)
        if not isinstance(val, expected):
            raise TypeError(
                f"{self.__class__.__name__}: param '{key}' must be {expected}, got {type(val).__name__}"
            )

        return val

    def setup(self, resources: dict[str, Any]):
        self.resources = resources

    def close(self):  # pragma: no cover
        pass

    @abstractmethod
    def run(self, state: PipelineState) -> PipelineState: ...


# -------- Optional mixins for training & index management --------
class Trainable(ABC):
    @abstractmethod
    def fit(self, dataset_spec: dict[str, Any], **kwargs) -> dict[str, Any]:
        """Train the component. Returns artifact metadata (paths, metrics)."""
        ...

    def load_artifacts(self, **kwargs):  # pragma: no cover
        pass

    def save_artifacts(self, **kwargs):  # pragma: no cover
        pass


class Context(Versioned):
    @property
    @abstractmethod
    def meta(self) -> Mapping[str, Any]: ...


TContext = TypeVar("TContext", bound=Context)


class Profile(Context, Generic[TContext]):
    @property
    @abstractmethod
    def id(self) -> int: ...


class Session(Context, Generic[TContext]): ...


class Request(Context, Fingerprintable): ...


class Encodable(Versioned, Generic[TContext]):
    """Encodable payload that *is a Context* via delegation.

    This base holds a concrete context instance and exposes profile/session
    by delegating to it. Subclasses can add fields/methods needed for encoding.
    """

    def __init__(
        self,
        profile: Profile[TContext],
        session: Session[TContext],
        meta: Mapping[str, Any] | None = None,
    ):
        self._profile: Profile[TContext] = profile
        self._session: Session[TContext] = session
        self._meta: Mapping[str, Any] = {} if meta is None else meta

        self._vec_profile_cached: dict[str, Tensor] = {}
        self._vec_session_cached: dict[str, Tensor] = {}
        self._vec_enc_cached: dict[tuple[str, str | None], Tensor] = {}

    @property
    def profile(self) -> Profile[TContext]:
        return self._profile

    @property
    def session(self) -> Session[TContext]:
        return self._session

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._meta

    def cache(
        self,
        encoder: "Encoder[TContext]",
        vec_profile: Tensor,
        vec_session: Tensor,
        vec_enc: Tensor,
        request: Request | None = None,
    ):
        self._vec_profile_cached[encoder.key] = vec_profile
        self._vec_session_cached[encoder.key] = vec_session
        self._vec_enc_cached[
            (
                encoder.key,
                None if request is None else request.key,
            )
        ] = vec_enc

    def get_vec_profile_cached(self, encoder: "Encoder[TContext]") -> Tensor | None:
        return self._vec_profile_cached.get(encoder.key)

    def get_vec_session_cached(self, encoder: "Encoder[TContext]") -> Tensor | None:
        return self._vec_session_cached.get(encoder.key)

    def get_vec_enc_cached(
        self, encoder: "Encoder[TContext]", request: Request | None
    ) -> Tensor | None:
        return self._vec_enc_cached.get((encoder.key, request.key))


class Encoded(Versioned, Generic[TContext]):
    """Encoded output that *shares the same Context* by referencing origin.

    Encoded delegates profile/session to the origin Encodable; request is optional and carried only for logging/repro.
    """

    def __init__(
        self,
        vector: Tensor,
        origin: Encodable[TContext],
        request: Request | None = None,
    ):
        self._vector: Tensor = vector
        self._origin: Encodable[TContext] = origin
        self._request: Request | None = request

    @property
    def vector(self) -> Tensor:
        return self._vector

    @property
    def profile(self) -> Profile[TContext]:
        return self._origin.profile

    @property
    def session(self) -> Session[TContext]:
        return self._origin.session

    @property
    def meta(self) -> Mapping[str, Any]:
        return self._origin.meta

    @property
    def request(self) -> Request | None:
        return self._request


class Encoder(Versioned, Generic[TContext], Fingerprintable):
    @abstractmethod
    def encode(
        self,
        encodable: Encodable[TContext],
        *,
        request: Request | None = None,
        **kwargs: Any,
    ) -> Encoded[TContext]: ...


class IndexOps(ABC):
    def add_items(self, items: Iterable[tuple[int, Any]]):  # (item_id, vector/feat)
        raise NotImplementedError

    def update_items(self, items: Iterable[tuple[int, Any]]):
        raise NotImplementedError

    def remove_items(self, item_ids: Iterable[int]):
        raise NotImplementedError

    def index_stats(self) -> dict[str, Any]:
        return {}


# -------- Concrete contracts per role --------
class CandidateRetriever(Component, IndexOps):
    component_kind: ClassVar[str] = "candidate_retriever"

    @abstractmethod
    def search_one(self, state: PipelineState, k: int) -> list[Candidate]:
        """Return a list of Candidate for the current user context."""
        ...

    @final
    def run(self, state: PipelineState) -> PipelineState:
        k: int = int(self.params.get("topk", 500))
        cands: list[Candidate] = self.search_one(state, k=k)
        # Contract: store under logs['retrieval'][self.id]
        state.logs.setdefault("retrieval", {})[self.id] = cands
        return state


class CandidateMerger(Component):
    component_kind: ClassVar[str] = "candidate_merger"

    @abstractmethod
    def merge(
        self, pools: dict[str, list[Candidate]], user_id: int, topk: int
    ) -> CandidateSet: ...

    @final
    def run(self, state: PipelineState) -> PipelineState:
        pools: dict[str, list[Candidate]] = state.logs.get("retrieval", {})
        topk: int = int(self.params.get("topk", 800))
        user_id: int = state.user.profile.id
        state.candset = self.merge(pools, user_id=user_id, topk=topk)
        return state


class CandidateShaper(Component):
    component_kind: ClassVar[str] = "candidate_shaper"

    @abstractmethod
    def shape(self, state: PipelineState) -> CandidateSet: ...

    @final
    def run(self, state: PipelineState) -> PipelineState:
        state.candset = self.shape(state)
        return state


class SlatePolicy(Component):
    component_kind: ClassVar[str] = "slate_policy"

    @abstractmethod
    def select_slate(self, state: PipelineState) -> PolicyOutput:
        """Consumes state.candset and returns PolicyOutput with slate + propensities."""
        ...

    @final
    def run(self, state: PipelineState) -> PipelineState:
        out: PolicyOutput = self.select_slate(state)
        state.slate = out.slate
        # Contract: store policy diagnostics for OPE/logging
        state.logs.setdefault("policy", {})[self.id] = out
        return state


class Evaluator(Component):
    component_kind: ClassVar[str] = "evaluator"

    @abstractmethod
    def evaluate(self, state: PipelineState) -> dict[str, Any]: ...

    @final
    def run(self, state: PipelineState) -> PipelineState:
        report: dict[str, Any] = self.evaluate(state)
        state.logs.setdefault("reports", {}).update(report)
        return state


class OPEEstimator(Component):
    component_kind: ClassVar[str] = "ope"

    @abstractmethod
    def estimate(self, state: PipelineState) -> dict[str, Any]: ...

    @final
    def run(self, state: PipelineState) -> PipelineState:
        report: dict[str, Any] = self.estimate(state)
        state.logs.setdefault("ope", {}).update(report)
        return state
