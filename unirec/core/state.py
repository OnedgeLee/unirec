from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class Candidate:
    item_id: int
    score: float
    source: str = "retrieval"
    features: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CandidateSet:
    user_id: int
    candidates: List[Candidate]
    def topk(self, k: int) -> "CandidateSet":
        c = sorted(self.candidates, key=lambda x: x.score, reverse=True)[:k]
        return CandidateSet(self.user_id, c)

@dataclass
class Slate:
    user_id: int
    items: List[int]
    aux: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PipelineState:
    context: Dict[str, Any]
    candset: Optional[CandidateSet] = None
    slate: Optional[Slate] = None
    logs: Dict[str, Any] = field(default_factory=dict)
