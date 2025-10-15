import faiss
import os
import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional, cast, override
from ...core.registry import register
from ...core.interfaces import CandidateRetriever
from ...core.state import Candidate, PipelineState
from ...models.encoders import encode_user


@register("retriever")
class TwotowerRetrieverFaiss(CandidateRetriever):
    @override
    def __init__(self, **params: Any):
        super().__init__(**params)
        self.topk: int = self.require_param("K", int, 500)
        self.use_gpu: bool = bool(self.params.get("use_gpu", False))
        self.gpu_id: int = int(self.params.get("gpu", 0))
        self.nprobe: int | None = cast(int | None, self.params.get("nprobe"))
        self.ef_search: int | None = cast(int | None, self.params.get("ef_search"))
        self.assume_norm_query: bool = bool(
            self.params.get("assume_normalized_query", True)
        )

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

        self.dim = int(getattr(self.index, "d", 0)) or None

        ids_path = cast(Optional[str], self.resources.get("item_ids_path"))
        if ids_path and os.path.exists(ids_path):
            self.item_ids = np.load(ids_path).astype(np.int64, copy=False).tolist()

        emb_path = cast(Optional[str], self.resources.get("item_emb_path"))
        if emb_path and os.path.exists(emb_path):
            self.item_memmap = np.load(emb_path, mmap_mode="r")

    @override
    def search_one(self, state: PipelineState, k: int) -> list[Candidate]:
        if self.index is None:
            raise RuntimeError("FAISS index not loaded")
        d = int(self.dim or 0)
        u: NDArray[np.float32] = encode_user(state.context, self.item_memmap, d).astype(
            np.float32, copy=False
        )
        if not self.assume_norm_query:
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
