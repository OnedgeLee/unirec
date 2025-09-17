import numpy as np
from ...core.registry import register
from ...core.interfaces import Retriever
from ...core.state import PipelineState, Candidate
from ..common import encode_user

@register("retriever")
class TwoTowerANN(Retriever):
    """Minimal dual-tower style retriever using brute-force cosine for scaffold.
    Replace with FAISS/HNSW index for production.
    """
    def setup(self, resources):
        emb_res = resources.get("item_embeddings")
        if isinstance(emb_res, str):
            self.item_emb = np.load(emb_res).astype(np.float32)
        else:
            self.item_emb = emb_res
        self.item_ids = resources.get("item_ids")  # optional list same order as embeddings
        self.topk = int(self.params.get("topk", 500))
        self.id_name = self.params.get("id", "retriever")

    def search_one(self, state: PipelineState, k: int) -> list[Candidate]:
        embs = self.item_emb
        d = embs.shape[1]
        u = encode_user(state.context, embs, d)
        sims = (embs @ u) / (np.linalg.norm(embs, axis=1)* (np.linalg.norm(u)+1e-9))
        idx = np.argpartition(-sims, min(k, len(sims)-1))[:k]
        idx = idx[np.argsort(-sims[idx])]
        cands = []
        for j in idx.tolist():
            item_id = int(self.item_ids[j]) if self.item_ids is not None else int(j)
            cands.append(Candidate(item_id=item_id, score=float(sims[j]), source=self.id_name))
        return cands
