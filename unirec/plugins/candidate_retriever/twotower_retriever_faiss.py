import faiss
import os
import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional, cast, override
from ...core.registry import register
from ...core.interfaces import CandidateRetriever
from ...core.state import Candidate, PipelineState
from ...data.encodable import UserEncodable
from ...models.encoders import ItemEncoder, UserEncoder


@register("retriever")
class TwotowerRetrieverFaiss(CandidateRetriever):
    @override
    def __init__(self, **params: Any):
        super().__init__(**params)
        self.user_encoder: UserEncoder = self.require_param("user_encoder", UserEncoder)
        self.item_encoder: ItemEncoder = self.require_param("item_encoder", ItemEncoder)
        self.topk: int = self.require_param("K", int, 500)
        self.use_gpu: bool = self.require_param("use_gpu", bool, False)
        self.gpu_id: int = self.require_param("gpu", int, 0)
        self.query_normalized: bool = self.require_param("query_normalized", bool, True)

        self.nprobe: int | None = self.optional_param("nprobe", int)
        self.ef_search: int | None = self.optional_param("ef_search", int)

        # sets on setup with resources
        self.index: Any = None
        self.item_memmap: NDArray[np.float32] | None = None
        self.item_ids: list[int] | None = None
        self.dim: int | None = None

    @override
    def setup(self, resources: dict[str, Any]) -> None:
        super().setup(resources)

        faiss_path = cast(str, self.resources.get("faiss_index_path"))
        if not faiss_path or not os.path.exists(faiss_path):
            raise FileNotFoundError("resources.faiss_index_path is invalid")

        self.index = faiss.read_index(faiss_path)

        if self.use_gpu:
            try:
                res = faiss.StandardGpuResources()
                self.index = faiss.index_cpu_to_gpu(res, self.gpu_id, self.index)
            except Exception:
                pass

        if self.nprobe is not None and hasattr(self.index, "nprobe"):
            self.index.nprobe = int(self.nprobe)
        if self.ef_search is not None and hasattr(self.index, "hnsw"):
            self.index.hnsw.efSearch = int(self.ef_search)

        self.dim = getattr(self.index, "d", None)

        ids_path = cast(Optional[str], self.resources.get("item_ids_path"))
        if ids_path and os.path.exists(ids_path):
            self.item_ids = np.load(ids_path).astype(np.int64, copy=False).tolist()

        emb_path = cast(Optional[str], self.resources.get("item_emb_path"))
        if emb_path and os.path.exists(emb_path):
            self.item_memmap = np.load(emb_path, mmap_mode="r")

        self.user_encoder.setup(self.dim, self.item_memmap)

    @override
    def search_one(self, state: PipelineState, k: int) -> list[Candidate]:
        if self.index is None:
            raise RuntimeError("FAISS index not loaded")
        user_encodable: UserEncodable = state.user
        u: NDArray[np.float32] = (
            self.user_encoder.encode(user_encodable, request=state.request)
            .vector.numpy()
            .astype(np.float32, copy=False)
        )
        if not self.query_normalized:
            u /= np.linalg.norm(u) + 1e-12

        k_eff = max(0, min(k, self.topk))
        if k_eff == 0:
            return []

        distances, indices = self.index.search(u.reshape(1, -1), k_eff)
        ids = indices[0]
        dists = distances[0]

        out: list[Candidate] = []
        for pos, idx in enumerate(ids.tolist()):
            if idx < 0:
                continue
            if self.item_ids is None:
                item_id = int(idx)
            else:
                item_id = self.item_ids[int(idx)]
            out.append(
                Candidate(item_id=item_id, score=float(dists[pos]), source=self.id)
            )
        return out
