import numpy as np
from numpy.random import RandomState
from numpy.typing import NDArray


def ips(
    r: NDArray[np.float32],
    pi_e: NDArray[np.float32],
    pi_b: NDArray[np.float32],
    clip: float | None = None,
):
    w: NDArray[np.float32] = np.clip(
        pi_e / np.maximum(pi_b, 1e-9), 0, clip if clip else np.inf
    )
    return float(np.mean(w * r))


def snips(
    r: NDArray[np.float32],
    pi_e: NDArray[np.float32],
    pi_b: NDArray[np.float32],
    clip: float | None = None,
):
    w: NDArray[np.float32] = np.clip(
        pi_e / np.maximum(pi_b, 1e-9), 0, clip if clip else np.inf
    )
    num: float = float(np.sum(w * r))
    den: float = float(np.sum(w) + 1e-12)
    return num / den


def doubly_robust(
    r: NDArray[np.float32],
    qhat: NDArray[np.float32],
    a_idx: NDArray[np.int32],
    pi_e: NDArray[np.float32],
    pi_b: NDArray[np.float32],
):
    w: NDArray[np.float32] = pi_e / np.maximum(pi_b, 1e-9)
    a: NDArray[np.float32] = qhat[np.arange(len(a_idx)), a_idx]
    return float(np.mean(a + w * (r - a)))


def paired_bootstrap(x: NDArray[np.float32], y, B: int = 1000, seed: int | None = 0):
    rs: RandomState = RandomState(seed)
    n: int = len(x)
    diffs: list[np.float32] = []
    for _ in range(B):
        idx: NDArray[np.int32] = rs.randint(0, n, n, dtype=np.int32)
        diffs.append(np.mean(x[idx] - y[idx]))
    diffs_arr: NDArray[np.float32] = np.array(diffs)
    lo, hi = np.percentile(diffs_arr, [2.5, 97.5])
    return float(np.mean(diffs_arr)), (float(lo), float(hi))
