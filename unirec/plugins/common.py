import numpy as np
from os import PathLike
from numpy.random import RandomState
from numpy.typing import ArrayLike, NDArray
from typing import Any, BinaryIO, TextIO


type ArraySource = ArrayLike | str | bytes | PathLike[str] | PathLike[bytes]


def l2_normalize(x: NDArray[np.float32], eps: float = 1e-9) -> NDArray[np.float32]:
    n: NDArray[np.float32] = np.linalg.norm(x, axis=-1, keepdims=True) + eps
    return x / n


def cosine(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) + 1e-9) / (np.linalg.norm(b) + 1e-9))


def encode_user(
    context: dict[str, Any], item_emb: NDArray[np.float32], d: int
) -> NDArray[np.float32]:
    """Deterministic encoder: average of recent clicked item embeddings,
    or a user-id seeded random vector if no history.
    Replace with trained user tower.
    """
    history: list[int] = context.get("recent_items", [])
    v: NDArray[np.float32]
    if history:
        # assume item ids map 1:1 to index if within bounds; otherwise random
        embs: list[NDArray[np.float32]] = []
        for iid in history[-10:]:
            if 0 <= iid < len(item_emb):
                embs.append(item_emb[iid])
        if embs:
            v = np.mean(np.stack(embs, axis=0), axis=0)
            return l2_normalize(v)[0] if v.ndim == 2 else v / (np.linalg.norm(v) + 1e-9)
    # fallback: user_id seed
    rs: RandomState = RandomState(context.get("user_id", 0) % 2**31)
    v = rs.randn(d).astype(np.float32)
    return v / (np.linalg.norm(v) + 1e-9)


def predict_ctr(
    context: dict[str, Any],
    item_vec: NDArray[np.float32],
    user_vec: NDArray[np.float32],
) -> float:
    """CTR predictor: sigmoid of scaled cosine. Replace with your ranker."""
    s: float = cosine(user_vec, item_vec)
    return 1.0 / (1.0 + np.exp(-3.0 * s))


def uncertainty_stub(context: dict[str, Any], item_id: int) -> float:
    """Very rough uncertainty proxy: smaller for head items, larger for tail (if freq provided)."""
    freq: float = context.get("freq", {}).get(item_id, 5.0)
    return 1.0 / (freq**0.5)


def load_array(source: ArraySource) -> NDArray[np.float32]:
    if isinstance(source, (str, bytes, bytearray, memoryview, PathLike)):
        return np.load(source).astype(np.float32)
    return np.array(source, dtype=np.float32)
