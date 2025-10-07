import numpy as np
from numpy.typing import NDArray
from typing import Any, cast, override
from ...core.registry import register
from ...core.interfaces import CandidateShaper
from ...core.state import Candidate, CandidateSet, PipelineState, Slate
from ..common import ArraySource, cosine, load_array


@register("candidate_shaper")
class MMR(CandidateShaper):
    """Maximal Marginal Relevance candidate shaper."""

    @override
    def __init__(self, **params: Any):
        super().__init__(**params)
        self.lmb: float = self.require_param("lambda", float, 0.2)

    @override
    def setup(self, resources: dict[str, Any]):
        super().setup(resources)
        self.item_emb: NDArray[np.float32] = load_array(
            cast(ArraySource, self.resources["item_embeddings"])
        )

    @override
    def shape(self, state: PipelineState) -> CandidateSet:
        if state.candset is None:
            raise KeyError(
                f"{type(self).__name__}: missing required PipelineState 'candset'"
            )
        cands: list[Candidate] = state.candset.candidates
        selected: list = []
        selected_ids: set = set()
        while cands and len(selected) < len(state.candset.candidates):
            best: Candidate
            best_score: float = -1e18
            for c in cands:
                if c.item_id in selected_ids:
                    continue
                # relevance = current score, redundancy = max sim to selected
                rel: float = c.score
                red: float = 0.0
                for s in selected:
                    red = max(
                        red, cosine(self.item_emb[c.item_id], self.item_emb[s.item_id])
                    )
                score: float = (1 - self.lmb) * rel - self.lmb * red
                if score > best_score:
                    best, best_score = c, score
            selected.append(best)
            selected_ids.add(best.item_id)
            cands.remove(best)
        return CandidateSet(user_id=state.candset.user_id, candidates=selected)
