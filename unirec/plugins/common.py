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


def load_array(source: ArraySource) -> NDArray[np.float32]:
    if isinstance(source, (str, bytes, bytearray, memoryview, PathLike)):
        return np.load(source).astype(np.float32)
    return np.array(source, dtype=np.float32)
