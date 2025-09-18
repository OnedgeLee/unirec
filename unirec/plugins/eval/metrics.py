import numpy as np
from numpy.typing import NDArray


def dcg_at_k(rels: list[int], k: int) -> float:
    rels_k: NDArray[np.float32] = np.asarray(rels, dtype=np.float32)[:k]
    if rels_k.size:
        return float(
            (rels_k[0]) + np.sum(rels_k[1:] / np.log2(np.arange(2, rels_k.size + 1)))
        )
    return 0.0


def ndcg_at_k(gt_items: set[int], pred_items: list[int], k: int = 10) -> float:
    rels: list[int] = [1 if i in gt_items else 0 for i in pred_items[:k]]
    idcg: float = dcg_at_k(sorted(rels, reverse=True), k)
    dcg: float = dcg_at_k(rels, k)
    return 0.0 if idcg == 0 else dcg / idcg


def recall_at_k(gt_items: set[int], pred_items: list[int], k: int = 10) -> float:
    if not gt_items:
        return 0.0
    return len(set(pred_items[:k]) & gt_items) / len(gt_items)


def ild_at_k(
    pred_items: list[int], item_emb: NDArray[np.float32], k: int = 10
) -> float:
    items: list[int] = pred_items[:k]
    if len(items) < 2:
        return 0.0
    sims: list[float] = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            a: NDArray[np.float32] = item_emb[items[i]]
            b: NDArray[np.float32] = item_emb[items[j]]
            sims.append(
                float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            )
    # diversity = 1 - avg cosine sim
    return float(1.0 - (np.mean(sims) if sims else 0.0))


def entropy_at_k(
    pred_items: list[int], categories: dict[int, str], k: int = 10
) -> float:
    xs: list[str] = [categories.get(i, "_unk") for i in pred_items[:k]]
    if not xs:
        return 0.0
    _, counts = np.unique(xs, return_counts=True)
    p: NDArray[np.float32] = (counts / counts.sum()).astype(np.float32)
    return float(-(p * np.log(p + 1e-12)).sum())


def coverage_at_k(
    all_pred_items: list[list[int]], catalog_size: int, k: int = 10
) -> float:
    seen: set = set()
    for sl in all_pred_items:
        seen.update(sl[:k])
    return len(seen) / max(1, catalog_size)
