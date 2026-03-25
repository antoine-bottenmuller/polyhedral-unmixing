import warnings
import itertools
import numpy as np
from typing import Tuple, Optional
from sklearn.metrics import accuracy_score
from src.unmixing import scalar, normed

# Abundance RMSE
def RMSE(A_gt:np.ndarray, A_hat:np.ndarray, individual:bool=False) -> float:
    """Shape: (..., n_endmembers) if not individual, where prod(...) == n_pixels; any shape if individual, where array pixels are gray-level."""
    try:
        A_gt  = np.asarray(A_gt )
        A_hat = np.asarray(A_hat)
    except:
        raise ValueError("Input variables must be ndarrays")
    if A_gt.ndim != A_hat.ndim:
        raise ValueError("Input arrays must have same dimension")
    if np.sum(np.array(A_gt.shape) != np.array(A_hat.shape)) != 0:
        raise ValueError("Input arrays must have same shape")
    if A_gt.ndim == 0:
        return np.abs(A_gt - A_hat).item()
    if individual:
        return np.sqrt(np.mean(np.square(A_gt - A_hat))).item()
    try:
        oshape:tuple = A_gt.shape
        A_gt  = A_gt .reshape(int(np.prod(oshape[:-1])), oshape[-1])
        A_hat = A_hat.reshape(int(np.prod(oshape[:-1])), oshape[-1])
    except:
        raise ValueError("Cannot reshape evaluation arrays to 2D arrays in non-individual evaluation mode")
    if A_gt.shape[0] < A_gt.shape[1]:
        warnings.warn("The number of rows (n_pixels) is less than the number of columns (n_endmembers). Did you mean the transpose matrix?")
    return np.mean(np.sqrt(np.mean(np.square(A_gt - A_hat), axis=0))).item()

# Endmember SAD
def SAD(M_gt:np.ndarray, M_hat:np.ndarray, individual:bool=False) -> float:
    """Shape: (n_endmembers, n_bands) if not individual; any shape if individual, where each array represents one unique vector."""
    try:
        M_gt  = np.asarray(M_gt )
        M_hat = np.asarray(M_hat)
    except:
        raise ValueError("Input variables must be ndarrays")
    if M_gt.ndim != M_hat.ndim:
        raise ValueError("Input arrays must have same dimension")
    if np.sum(np.array(M_gt.shape) != np.array(M_hat.shape)) != 0:
        raise ValueError("Input arrays must have same shape")
    if M_gt.ndim == 0:
        return float(0.0)
    if individual:
        return np.arccos(scalar(normed(M_gt.flatten()),normed(M_hat.flatten()))).item()
    if M_gt.ndim not in {1,2}:
        raise ValueError("Input arrays must be of dimension 1 or 2")
    if M_gt.ndim == 1:
        warnings.warn("Input arrays are of dimension 1, while individual=False. Considered as individual=True; otherwise, the SAD would be exactly 0.")
    elif M_gt.shape[0] > M_gt.shape[1]:
        warnings.warn("The number of rows (n_endmembers) is larger than the number of columns (n_bands). Did you mean the transpose matrix?")
    return np.mean(np.arccos(scalar(normed(M_gt),normed(M_hat)))).item()

# Permute (A_hat, M_hat) to align with (_, M_gt)
def permute_to_GT_M(M_hat:np.ndarray, A_hat:Optional[np.ndarray], M_gt:np.ndarray) -> Tuple[np.ndarray, np.ndarray]|np.ndarray:
    """
    Permute the rows of M_hat and A_hat so that M_hat's rows are aligned with the ones of M_gt, minimizing SAD.\n
    * Endmember matrix shape: (n_endmembers, n_bands); 
    * Abundance matrix shape: (..., n_endmembers).\n
    If A_hat is None, then only permuted M_hat is returned. Otherwise, permuted M_hat and A_hat are both returned.
    """
    if M_hat is None or M_gt is None:
        raise ValueError("M_hat and M_gt arguments must both be given")
    
    n_endm = M_gt.shape[0]
    permutations = np.asarray(list(itertools.permutations(np.arange(n_endm), n_endm))).tolist()

    idx = permutations[0]
    val = SAD(M_gt, M_hat[idx])
    for i in range(1,len(permutations)):
        new_idx = permutations[i]
        new_val = SAD(M_gt, M_hat[new_idx])
        if new_val < val:
            idx = new_idx
            val = new_val

    M_hat = M_hat[idx]
    if A_hat is None:
        return M_hat

    A_hat = A_hat[...,idx]
    return M_hat, A_hat

# Permute (A_hat, M_hat) to align with (A_gt, _)
def permute_to_GT_A(M_hat:Optional[np.ndarray], A_hat:np.ndarray, A_gt:np.ndarray) -> Tuple[np.ndarray, np.ndarray]|np.ndarray:
    """
    Permute the columns of M_hat and A_hat so that A_hat's columns are aligned with the ones of A_gt, minimizing RMSE.\n
    * Endmember matrix shape: (n_endmembers, n_bands); 
    * Abundance matrix shape: (..., n_endmembers).\n
    If M_hat is None, then only permuted A_hat is returned. Otherwise, permuted M_hat and A_hat are both returned.
    """
    if A_hat is None or A_gt is None:
        raise ValueError("M_hat and M_gt arguments must both be given")
    
    n_endm = A_gt.shape[-1]
    permutations = np.asarray(list(itertools.permutations(np.arange(n_endm), n_endm))).tolist()

    idx = permutations[0]
    val = RMSE(A_gt, A_hat[...,idx])
    for i in range(1,len(permutations)):
        new_idx = permutations[i]
        new_val = RMSE(A_gt, A_hat[...,new_idx])
        if new_val < val:
            idx = new_idx
            val = new_val

    A_hat = A_hat[...,idx]
    if M_hat is None:
        return A_hat
    
    M_hat = M_hat[idx]
    return M_hat, A_hat

# Compute classification map accuracy
def _map_accuracy(y_pred:np.ndarray, y_labl:np.ndarray) -> np.ndarray:
    yp = np.zeros(y_pred.shape[:-1], np.int32)
    yl = np.zeros(y_labl.shape[:-1], np.int32)
    for val in range(y_pred.shape[-1]):
        yp[y_pred[...,val].astype(bool)] = val
    for val in range(y_labl.shape[-1]):
        yl[y_labl[...,val].astype(bool)] = val
    return accuracy_score(yp, yl, normalize=True)

# Re-order predicted class labels over map accuracy
def reorder_C(y_pred:np.ndarray, y_labl:np.ndarray) -> np.ndarray:
    """Re-order labels of predicted 2D classification map y_pred to align with y_labl (Ground-Truth)."""
    u_pred = np.unique(y_pred)
    u_labl = np.unique(y_labl)
    im_pred = np.asarray([y_pred.flatten()==val for val in u_pred], dtype=bool).T
    im_labl = np.asarray([y_labl.flatten()==val for val in u_labl], dtype=bool).T
    permutations = np.asarray(list(itertools.permutations(np.arange(len(u_labl)), len(u_labl)))).tolist()
    idx = permutations[0]
    acc = _map_accuracy(im_labl, im_pred[:,idx])
    for i in range(1,len(permutations)):
        new_idx = permutations[i]
        new_acc = _map_accuracy(im_labl, im_pred[:,new_idx])
        if new_acc > acc:
            idx = new_idx
            acc = new_acc
    new_y_pred = np.zeros(im_pred.shape[:1], y_pred.dtype)
    for i in idx: new_y_pred[im_pred[:,idx[i]]] = u_pred[i]
    return new_y_pred.reshape(*y_pred.shape)

