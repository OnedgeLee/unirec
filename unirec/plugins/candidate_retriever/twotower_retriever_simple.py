import numpy as np
from numpy.typing import NDArray
from typing import Any, cast, override
from ...core.registry import register
from ...core.interfaces import CandidateRetriever
from ...core.state import Candidate, PipelineState
from ..common import ArraySource, load_array
from ...data.encodable import UserEncodable
from ...models.encoders import ItemEncoder, UserEncoder


@register("candidate_retriever")
class TwotowerRetrieverSimple(CandidateRetriever):
    """Minimal two-tower style candidate retriever using brute-force cosine for scaffold."""

    @override
    def __init__(self, **params: Any):
        super().__init__(**params)
        self.user_encoder: UserEncoder = self.require_param("user_encoder", UserEncoder)
        self.item_encoder: ItemEncoder = self.require_param("item_encoder", ItemEncoder)
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
        user_encodable: UserEncodable = state.user
        u: NDArray[np.float32] = (
            self.user_encoder.encode(
                user_encodable, request=state.request, item_memmap=embs, emb_dim=d
            )
            .vector.numpy()
            .astype(np.float32, copy=False)
        )
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
