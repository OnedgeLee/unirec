import numpy as np
from typing import List, Dict, Set

def dcg_at_k(rels: List[int], k: int) -> float:
    rels = np.asfarray(rels)[:k]
    if rels.size:
        return float((rels[0]) + np.sum(rels[1:] / np.log2(np.arange(2, rels.size + 1))))
    return 0.0

def ndcg_at_k(gt_items: Set[int], pred_items: List[int], k: int = 10) -> float:
    rels = [1 if i in gt_items else 0 for i in pred_items[:k]]
    idcg = dcg_at_k(sorted(rels, reverse=True), k)
    dcg = dcg_at_k(rels, k)
    return 0.0 if idcg == 0 else dcg / idcg

def recall_at_k(gt_items: Set[int], pred_items: List[int], k: int = 10) -> float:
    if not gt_items: return 0.0
    return len(set(pred_items[:k]) & gt_items) / len(gt_items)

def ild_at_k(pred_items: List[int], item_emb: np.ndarray, k: int = 10) -> float:
    items = pred_items[:k]
    if len(items) < 2: return 0.0
    sims = []
    for i in range(len(items)):
        for j in range(i+1, len(items)):
            a = item_emb[items[i]]; b = item_emb[items[j]]
            sims.append(float(np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b)+1e-9)))
    # diversity = 1 - avg cosine sim
    return 1.0 - (np.mean(sims) if sims else 0.0)

def entropy_at_k(pred_items: List[int], categories: Dict[int, str], k: int = 10) -> float:
    from math import log
    xs = [categories.get(i, "_unk") for i in pred_items[:k]]
    if not xs: return 0.0
    _, counts = np.unique(xs, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p + 1e-12)).sum())

def coverage_at_k(all_pred_items: List[List[int]], catalog_size: int, k: int = 10) -> float:
    seen = set()
    for sl in all_pred_items:
        seen.update(sl[:k])
    return len(seen) / max(1, catalog_size)
