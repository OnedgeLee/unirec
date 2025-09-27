from typing import cast, override
from ...core.registry import register
from ...core.interfaces import Merger
from ...core.state import PipelineState, CandidateSet, Candidate


@register("merger")
class WeightedUnion(Merger):
    @override
    def merge(
        self, pools: dict[str, list[Candidate]], user_id: int, topk: int
    ) -> CandidateSet:
        weights: dict[str, float] = cast(
            dict[str, float], self.params.get("weights", {})
        )
        pool: dict[int, float] = {}
        srcmap: dict[int, set[str]] = {}
        for src, cands in pools.items():
            w: float = float(weights.get(src, 1.0))
            for c in cands:
                pool[c.item_id] = pool.get(c.item_id, 0.0) + w * c.score
                srcmap.setdefault(c.item_id, set()).add(src)
        merged: list[Candidate] = [
            Candidate(item_id=k, score=v, source=",".join(sorted(srcmap[k])))
            for k, v in pool.items()
        ]
        merged.sort(key=lambda x: x.score, reverse=True)
        return CandidateSet(user_id=user_id, candidates=merged[:topk])
