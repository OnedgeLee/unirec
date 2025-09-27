import numpy as np
from common import ArraySource, load_array
from numpy.typing import NDArray
from typing import Any, cast, override
from ...core.registry import register
from ...core.interfaces import Evaluator
from ...core.state import PipelineState
from .metrics import ndcg_at_k, recall_at_k, ild_at_k, entropy_at_k, coverage_at_k


@register("evaluator")
class OfflineEvaluator(Evaluator):
    """Compute offline metrics given per-user slates and ground truth.

    Expects in state.logs:
      gt: {user_id: set(item_ids)}
      slates: {user_id: [item_ids...]}
      resources for ILD/Entropy: item_embeddings, categories(optional)
    """

    @override
    def __init__(self, **params: Any):
        super().__init__(**params)
        self.K: int = self.require_param("K", int)

    @override
    def setup(self, resources: dict[str, Any]):
        super().setup(resources)
        self.item_emb: NDArray[np.float32] = load_array(
            cast(ArraySource, self.resources["item_embeddings"])
        )
        self.categories: dict[int, str] | None = self.resources.get("categories", {})

    @override
    def evaluate(self, state: PipelineState) -> dict[str, Any]:
        logs: dict[str, Any] = state.logs
        gt: dict[str, set[int]] = logs.get("gt", {})
        slates: dict[str, list[int]] = logs.get("slates", {})
        K: int = int(self.params.get("K", 10))
        ndcgs: list[float] = []
        recalls: list[float] = []
        ilds: list[float] = []
        ents: list[float] = []
        all_slates: list[list[int]] = []
        for u, slate in slates.items():
            g: set[int] = set(gt.get(u, []))
            all_slates.append(slate)
            ndcgs.append(ndcg_at_k(g, slate, K))
            recalls.append(recall_at_k(g, slate, K))
            ilds.append(ild_at_k(slate, self.item_emb, K))
            ents.append(
                entropy_at_k(slate, self.categories, K) if self.categories else 0.0
            )
        report: dict[str, Any] = {
            f"NDCG@{K}": float(np.mean(ndcgs) if ndcgs else 0.0),
            f"Recall@{K}": float(np.mean(recalls) if recalls else 0.0),
            f"ILD@{K}": float(np.mean(ilds) if ilds else 0.0),
            f"Entropy@{K}": float(np.mean(ents) if ents else 0.0),
            f"Coverage@{K}": coverage_at_k(all_slates, len(self.item_emb)),
        }
        return report
