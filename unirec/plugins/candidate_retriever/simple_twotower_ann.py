import numpy as np
from numpy.typing import NDArray
from typing import Any, cast, override
from ...core.registry import register
from ...core.interfaces import CandidateRetriever
from ...core.state import Candidate, PipelineState
from ..common import ArraySource, encode_user, load_array


@register("candidate_retriever")
class SimpleTwotowerAnn(CandidateRetriever):
    """Minimal two-tower style candidate retriever using brute-force cosine for scaffold.
    Replace with FAISS/HNSW index for production.
    """

    @override
    def __init__(self, **params: Any):
        super().__init__(**params)
        self.topk: int = self.require_param("K", int, 500)

    @override
    def setup(self, resources: dict[str, Any]):
        super().setup(resources)
        self.item_emb: NDArray[np.float32] = load_array(
            cast(ArraySource, self.resources["item_embeddings"])
        )
        self.item_ids: list[int] | None = self.resources.get(
            "item_ids"
        )  # optional list same order as embeddings

    @override
    def search_one(self, state: PipelineState, k: int) -> list[Candidate]:
        embs: NDArray[np.float32] = self.item_emb
        d: int = embs.shape[1]
        u: NDArray[np.float32] = encode_user(state.context, embs, d)
        sims: NDArray[np.float32] = (embs @ u) / (
            np.linalg.norm(embs, axis=1) * (np.linalg.norm(u) + 1e-9)
        )
        idx: NDArray[np.intp] = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        cands: list[Candidate] = []
        for j in idx.tolist():
            item_id: int = (
                int(self.item_ids[j]) if self.item_ids is not None else int(j)
            )
            cands.append(
                Candidate(item_id=item_id, score=float(sims[j]), source=self.id)
            )
        return cands
