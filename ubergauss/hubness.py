



import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_distances
from typing import Union


# do not be lazy, this is a test, argpartition does now guarantee the smallest version be in the front, fix make _mean_k_smallest to handle this correctly

def _mean_k_smallest(values: np.ndarray, k: int, axis: int, skip_first: bool) -> np.ndarray:
    start = 1 if skip_first else 0
    k_idx = k + start
    # Partition to isolate the smallest elements
    partitioned = np.argpartition(values, kth=k_idx - 1, axis=axis)
    # Extract only the k indices we care about
    idx_slice = [slice(None)] * values.ndim
    idx_slice[axis] = slice(0, k_idx)
    top_indices = partitioned[tuple(idx_slice)]
    # Retrieve the actual values
    selected = np.take_along_axis(values, top_indices, axis=axis)
    # Sort the selected values to ensure the first one is indeed the smallest for skip_first
    selected = np.sort(selected, axis=axis)
    # Slice skip_first and average
    final_slice = [slice(None)] * values.ndim
    final_slice[axis] = slice(start, None)
    return selected[tuple(final_slice)].mean(axis=axis)

def transform(
    X: np.ndarray,
    Y: Union[np.ndarray, None] = None,
    k: int = 10,
    algo: int = 2,
    metric: str = 'cosine',
    skip_diag: bool = True
) -> np.ndarray:
    """
    algo: 0=None, 1=Norm, 2=CSLS, 3=LS, 4=NICDM, 5=KA
    """
    if Y is not None:
        assert (X is Y) == skip_diag, 'skip_diag iff y=x'
    dist = X if Y is None else pairwise_distances(X, Y, metric=metric)
    if algo == 0: return dist
    if algo == 1: return normalize(dist, axis=0)

    # Calculate local scaling factors
    r_i = _mean_k_smallest(dist, k, axis=1, skip_first=skip_diag)[:, np.newaxis]
    r_j = _mean_k_smallest(dist, k, axis=0, skip_first=skip_diag)[np.newaxis, :]

    eps = 1e-8
    if algo == 2: # CSLS
        return 2.0 * dist - r_i - r_j
    if algo == 3: # LS
        return 1.0 - np.exp(-(dist**2) / np.maximum(r_i * r_j, eps))
    if algo == 4: # NICDM
        return dist / np.sqrt(np.maximum(r_i * r_j, eps))
    if algo == 5: # KA
        return (dist / np.maximum(r_i, eps)) + (dist / np.maximum(r_j, eps))

    return dist


