from ...core.registry import register
from ...core.interfaces import Evaluator
from ...core.state import PipelineState
from .metrics import ndcg_at_k, recall_at_k, ild_at_k, entropy_at_k, coverage_at_k
import numpy as np

@register("evaluator")
class OfflineEvaluator(Evaluator):
    """Compute offline metrics given per-user slates and ground truth.

    Expects in state.logs:
      gt: {user_id: set(item_ids)}
      slates: {user_id: [item_ids...]}
      resources for ILD/Entropy: item_embeddings, categories(optional)
    """
    def setup(self, resources):
        self.item_emb = resources.get("item_embeddings")
        if isinstance(self.item_emb, str):
            import numpy as np
            self.item_emb = np.load(self.item_emb)
        self.categories = resources.get("categories", {})

    def run(self, state: PipelineState) -> PipelineState:
        logs = state.logs
        gt = logs.get("gt", {})
        slates = logs.get("slates", {})
        K = int(self.params.get("K", 10))
        ndcgs, recalls, ilds, ents = [], [], [], []
        all_slates = []
        for u, slate in slates.items():
            g = set(gt.get(u, []))
            all_slates.append(slate)
            ndcgs.append(ndcg_at_k(g, slate, K))
            recalls.append(recall_at_k(g, slate, K))
            ilds.append(ild_at_k(slate, self.item_emb, K) if self.item_emb is not None else 0.0)
            ents.append(entropy_at_k(slate, self.categories, K) if self.categories else 0.0)
        report = {
            f"NDCG@{K}": float(np.mean(ndcgs) if ndcgs else 0.0),
            f"Recall@{K}": float(np.mean(recalls) if recalls else 0.0),
            f"ILD@{K}": float(np.mean(ilds) if ilds else 0.0),
            f"Entropy@{K}": float(np.mean(ents) if ents else 0.0),
            f"Coverage@{K}": coverage_at_k(all_slates, len(self.item_emb) if self.item_emb is not None else 1, K),
        }
        state.logs.setdefault("reports", {}).update(report)
        return state
