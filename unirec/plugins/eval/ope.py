import numpy as np

def ips(r, pi_e, pi_b, clip=None):
    w = np.clip(pi_e / np.maximum(pi_b, 1e-9), 0, clip if clip else np.inf)
    return float(np.mean(w * r))

def snips(r, pi_e, pi_b, clip=None):
    w = np.clip(pi_e / np.maximum(pi_b, 1e-9), 0, clip if clip else np.inf)
    num = float(np.sum(w * r)); den = float(np.sum(w) + 1e-12)
    return num / den

def doubly_robust(r, qhat, a_idx, pi_e, pi_b):
    w = pi_e / np.maximum(pi_b, 1e-9)
    a = qhat[np.arange(len(a_idx)), a_idx]
    return float(np.mean(a + w * (r - a)))

def paired_bootstrap(x, y, B=1000, seed=0):
    rs = np.random.RandomState(seed)
    n = len(x); diffs = []
    for _ in range(B):
        idx = rs.randint(0, n, n)
        diffs.append(np.mean(x[idx] - y[idx]))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(np.mean(diffs)), (float(lo), float(hi))
