from collections import defaultdict
from ...core.registry import register
from ...core.interfaces import Merger
from ...core.state import PipelineState, CandidateSet, Candidate

@register("merger")
class WeightedUnion(Merger):
    def merge(self, pools, user_id: int, topk: int) -> CandidateSet:
        weights = self.params.get("weights", {})
        pool = defaultdict(float)
        srcmap = {}
        for src, cands in pools.items():
            w = float(weights.get(src, 1.0))
            for c in cands:
                pool[c.item_id] += w * c.score
                srcmap.setdefault(c.item_id, set()).add(src)
        merged = [Candidate(item_id=k, score=v, source=",".join(sorted(srcmap[k]))) for k, v in pool.items()]
        merged.sort(key=lambda x: x.score, reverse=True)
        return CandidateSet(user_id=user_id, candidates=merged[:topk])
