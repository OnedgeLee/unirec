import numpy as np
from ...core.registry import register
from ...core.interfaces import Reranker
from ...core.state import PipelineState, Slate
from ..common import cosine

@register("reranker")
class MMR(Reranker):
    """Maximal Marginal Relevance reranker."""
    def setup(self, resources):
        self.lmb = float(self.params.get("lambda", 0.2))
        emb_res = resources.get("item_embeddings")
        self.item_emb = np.load(emb_res).astype(np.float32) if isinstance(emb_res, str) else emb_res
    def rerank(self, state: PipelineState) -> CandidateSet:
        if state.candset is None:
            return state
        cands = state.candset.candidates
        selected, selected_ids = [], set()
        while cands and len(selected) < len(state.candset.candidates):
            best, best_score = None, -1e18
            for c in cands:
                if c.item_id in selected_ids:
                    continue
                # relevance = current score, redundancy = max sim to selected
                rel = c.score
                red = 0.0
                for s in selected:
                    red = max(red, cosine(self.item_emb[c.item_id], self.item_emb[s.item_id]))
                score = (1-self.lmb)*rel - self.lmb*red
                if score > best_score:
                    best, best_score = c, score
            selected.append(best)
            selected_ids.add(best.item_id)
            cands.remove(best)
        return CandidateSet(user_id=state.candset.user_id, candidates=selected)

class CosineSim:
    def __init__(self, item_emb): self.item_emb = item_emb
    def __call__(self, i, j): return cosine(self.item_emb[i], self.item_emb[j])
