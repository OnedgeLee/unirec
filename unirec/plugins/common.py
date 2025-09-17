import numpy as np
from typing import Dict, Any

def l2_normalize(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / n

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32); b = b.astype(np.float32)
    return float(np.dot(a, b) / (np.linalg.norm(a)+1e-9) / (np.linalg.norm(b)+1e-9))

def encode_user(context: Dict[str, Any], item_emb: np.ndarray, d: int) -> np.ndarray:
    """Deterministic encoder: average of recent clicked item embeddings,
    or a user-id seeded random vector if no history.
    Replace with trained user tower.
    """
    history = context.get("recent_items", [])
    if history:
        # assume item ids map 1:1 to index if within bounds; otherwise random
        embs = []
        for iid in history[-10:]:
            if 0 <= iid < len(item_emb):
                embs.append(item_emb[iid])
        if embs:
            v = np.mean(np.stack(embs, axis=0), axis=0)
            return l2_normalize(v)[0] if v.ndim == 2 else v / (np.linalg.norm(v)+1e-9)
    # fallback: user_id seed
    rs = np.random.RandomState(context.get("user_id", 0) % 2**31)
    v = rs.randn(d).astype(np.float32)
    return v / (np.linalg.norm(v)+1e-9)

def predict_ctr(context: Dict[str, Any], item_vec: np.ndarray, user_vec: np.ndarray) -> float:
    """CTR predictor: sigmoid of scaled cosine. Replace with your ranker."""
    s = cosine(user_vec, item_vec)
    return 1.0 / (1.0 + np.exp(-3.0 * s))

def uncertainty_stub(context: Dict[str, Any], item_id: int) -> float:
    """Very rough uncertainty proxy: smaller for head items, larger for tail (if freq provided)."""
    freq = context.get("freq", {}).get(item_id, 5.0)
    return 1.0 / (freq ** 0.5)
