from dataclasses import dataclass, field
from typing import Any, final, TYPE_CHECKING

if TYPE_CHECKING:
    from ..data.context import RequestContext
    from ..data.encodable import UserEncodable


@dataclass
class Candidate:
    item_id: int
    score: float
    source: str = "retrieval"
    features: dict[str, Any] = field(default_factory=dict)


@dataclass
class CandidateSet:
    user_id: int
    candidates: list[Candidate]

    @final
    def topk(self, k: int) -> "CandidateSet":
        c: list[Candidate] = sorted(
            self.candidates, key=lambda x: x.score, reverse=True
        )[:k]
        return CandidateSet(self.user_id, c)


@dataclass
class Slate:
    user_id: int
    items: list[int]
    aux: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineState:
    user: "UserEncodable"
    context: dict[str, Any]  # Temporal context for incomplete scaffolding
    request: "RequestContext | None" = None
    candset: CandidateSet | None = None
    slate: Slate | None = None
    logs: dict[str, Any] = field(default_factory=dict)
