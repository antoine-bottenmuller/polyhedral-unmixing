import warnings
import itertools
import numpy as np
from skimage.metrics import structural_similarity as SSIM
from sklearn.metrics import accuracy_score, precision_score
from scipy.optimize import linear_sum_assignment
from typing import Tuple, Optional, Literal
from src.unmixing import scalar, normed


#%%
# Evaluation metrics: SAD, RMSE, DSSIM
###

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

# Abundance map DSSIM
def DSSIM(
    im1,
    im2,
    *,
    win_size=None,
    gradient=False,
    data_range=None,
    channel_axis=None,
    gaussian_weights=False,
    full=False,
    **kwargs,
):
    """
    Compute the mean structural DISsimilarity index between two images.
    Please pay attention to the `data_range` parameter with floating-point images.

    Parameters
    ----------
    im1, im2 : ndarray
        Images. Any dimensionality with same shape.
    win_size : int or None, optional
        The side-length of the sliding window used in comparison. Must be an
        odd value. If `gaussian_weights` is True, this is ignored and the
        window size will depend on `sigma`.
    gradient : bool, optional
        If True, also return the gradient with respect to im2.
    data_range : float, optional
        The data range of the input image (difference between maximum and
        minimum possible values). By default, this is estimated from the image
        data type. This estimate may be wrong for floating-point image data.
        Therefore it is recommended to always pass this scalar value explicitly
        (see note below).
    channel_axis : int or None, optional
        If None, the image is assumed to be a grayscale (single channel) image.
        Otherwise, this parameter indicates which axis of the array corresponds
        to channels.

        .. versionadded:: 0.19
           ``channel_axis`` was added in 0.19.
    gaussian_weights : bool, optional
        If True, each patch has its mean and variance spatially weighted by a
        normalized Gaussian kernel of width sigma=1.5.
    full : bool, optional
        If True, also return the full structural similarity image.

    Other Parameters
    ----------------
    use_sample_covariance : bool
        If True, normalize covariances by N-1 rather than, N where N is the
        number of pixels within the sliding window.
    K1 : float
        Algorithm parameter, K1 (small constant).
    K2 : float
        Algorithm parameter, K2 (small constant).
    sigma : float
        Standard deviation for the Gaussian when `gaussian_weights` is True.

    Returns
    -------
    dmssim : float
        The mean structural DISsimilarity index over the image.
    dgrad : ndarray
        The gradient of the structural DISsimilarity between im1 and im2.
        This is only returned if `gradient` is set to True.
    DS : ndarray
        The full DSSIM image.  This is only returned if `full` is set to True.

    Notes
    -----
    If `data_range` is not specified, the range is automatically guessed
    based on the image data type. However for floating-point image data, this
    estimate yields a result double the value of the desired range, as the
    `dtype_range` in `skimage.util.dtype.py` has defined intervals from -1 to
    +1. This yields an estimate of 2, instead of 1, which is most often
    required when working with image data (as negative light intensities are
    nonsensical). In case of working with YCbCr-like color data, note that
    these ranges are different per channel (Cb and Cr have double the range
    of Y), so one cannot calculate a channel-averaged SSIM with a single call
    to this function, as identical ranges are assumed for each channel.
    
    """

    if gradient is True and full is True:
        mssim, grad, S = SSIM(
            im1, im2, 
            win_size=win_size, 
            gradient=gradient,
            data_range=data_range,
            channel_axis=channel_axis,
            gaussian_weights=gaussian_weights,
            full=full,
            **kwargs
        )
        dmssim = (1 - mssim) / 2
        dgrad  = - grad / 2
        DS     = (1 -     S) / 2
        return dmssim, dgrad, DS
    
    elif gradient is True and full is False:
        mssim, grad = SSIM(
            im1, im2, 
            win_size=win_size, 
            gradient=gradient,
            data_range=data_range,
            channel_axis=channel_axis,
            gaussian_weights=gaussian_weights,
            full=full,
            **kwargs
        )
        dmssim = (1 - mssim) / 2
        dgrad  = - grad / 2
        return dmssim, dgrad
    
    elif gradient is False and full is True:
        mssim, S = SSIM(
            im1, im2, 
            win_size=win_size, 
            gradient=gradient,
            data_range=data_range,
            channel_axis=channel_axis,
            gaussian_weights=gaussian_weights,
            full=full,
            **kwargs
        )
        dmssim = (1 - mssim) / 2
        DS     = (1 -     S) / 2
        return dmssim, DS
    
    mssim = SSIM(
        im1, im2, 
        win_size=win_size, 
        gradient=gradient,
        data_range=data_range,
        channel_axis=channel_axis,
        gaussian_weights=gaussian_weights,
        full=full,
        **kwargs
    )
    dmssim = (1 - mssim) / 2
    return dmssim


#%%
# Permutation functions: 
# 1. permute_to_GT_M (permute M_hat and A_hat over GT endmembers M_gt), 
# 2. permute_to_GT_A (permute M_hat and A_hat over GT abundances A_gt), 
# 3. reorder_C (re-order classification labels over GT labels).
###

# Permute (A_hat, M_hat) to align with (_, M_gt)
def permute_to_GT_M(
        M_hat: np.ndarray, 
        A_hat: Optional[np.ndarray], 
        M_gt: np.ndarray, 
        method: Literal['all_arrangements','linear_sum'] = 'linear_sum'
) -> Tuple[np.ndarray, np.ndarray]|np.ndarray:
    """
    Permute the rows of M_hat and A_hat so that M_hat's rows are aligned with the ones of M_gt, minimizing SAD.\n
    * Endmember matrix shape: (n_endmembers, n_bands); 
    * Abundance matrix shape: (..., n_endmembers).\n
    If A_hat is None, then only permuted M_hat is returned. Otherwise, permuted M_hat and A_hat are both returned.
    """
    if M_hat is None or M_gt is None:
        raise ValueError("M_hat and M_gt arguments must both be given")

    if method.lower() == 'all_arrangements':
        n_endm = M_gt.shape[0]
        
        permutations = np.asarray(list(itertools.permutations(np.arange(n_endm), n_endm))).tolist()
    
        idx = permutations[0]
        val = SAD(M_gt, M_hat[idx], individual=False)
        for i in range(1,len(permutations)):
            new_idx = permutations[i]
            new_val = SAD(M_gt, M_hat[new_idx], individual=False)
            if new_val < val:
                idx = new_idx
                val = new_val
    
    elif method.lower() == 'linear_sum':
        exp_M_gt  = np.expand_dims(normed(M_gt ), axis=1)
        exp_M_hat = np.expand_dims(normed(M_hat), axis=0)
        matrix = np.arccos(scalar(exp_M_gt, exp_M_hat))
        del(exp_M_gt, exp_M_hat)

        _, idx = linear_sum_assignment(matrix, maximize=False)
    
    else:
        raise ValueError("Argument 'method' must be either 'all_arrangements' or 'linear_sum'")

    M_hat = M_hat[idx]
    if A_hat is None:
        return M_hat

    A_hat = A_hat[...,idx]
    return M_hat, A_hat

# Permute (A_hat, M_hat) to align with (A_gt, _)
def permute_to_GT_A(
        M_hat: Optional[np.ndarray], 
        A_hat: np.ndarray, 
        A_gt: np.ndarray, 
        method: Literal['all_arrangements','linear_sum'] = 'linear_sum'
) -> Tuple[np.ndarray, np.ndarray]|np.ndarray:
    """
    Permute the columns of M_hat and A_hat so that A_hat's columns are aligned with the ones of A_gt, minimizing RMSE.\n
    * Endmember matrix shape: (n_endmembers, n_bands); 
    * Abundance matrix shape: (..., n_endmembers).\n
    If M_hat is None, then only permuted A_hat is returned. Otherwise, permuted M_hat and A_hat are both returned.
    """
    if A_hat is None or A_gt is None:
        raise ValueError("M_hat and M_gt arguments must both be given")

    if method.lower() == 'all_arrangements':
        n_endm = A_gt.shape[-1]
        
        permutations = np.asarray(list(itertools.permutations(np.arange(n_endm), n_endm))).tolist()
    
        idx = permutations[0]
        val = RMSE(A_gt, A_hat[...,idx], individual=False)
        for i in range(1,len(permutations)):
            new_idx = permutations[i]
            new_val = RMSE(A_gt, A_hat[...,new_idx], individual=False)
            if new_val < val:
                idx = new_idx
                val = new_val

    elif method.lower() == 'linear_sum':
        diff_A = np.expand_dims(A_gt, axis=-1) - np.expand_dims(A_hat, axis=-2)
        matrix = np.sqrt(np.mean(np.square(diff_A), axis=tuple(range(A_gt.ndim-1))))
        del(diff_A)

        _, idx = linear_sum_assignment(matrix, maximize=False)
    
    else:
        raise ValueError("Argument 'method' must be either 'all_arrangements' or 'linear_sum'")

    A_hat = A_hat[...,idx]
    if M_hat is None:
        return A_hat
    
    M_hat = M_hat[idx]
    return M_hat, A_hat

# Compute classification map accuracy
def _map_accuracy(y_pred:np.ndarray, y_labl:np.ndarray) -> np.ndarray:
    """Accuracy for binary arrays of shape (..., n_classes)"""
    yp = np.zeros(y_pred.shape[:-1], np.int32)
    yl = np.zeros(y_labl.shape[:-1], np.int32)
    for val in range(y_pred.shape[-1]):
        yp[y_pred[...,val].astype(bool)] = val
    for val in range(y_labl.shape[-1]):
        yl[y_labl[...,val].astype(bool)] = val
    return accuracy_score(y_true=yl.flatten(), y_pred=yp.flatten(), normalize=True)

# Re-order predicted class labels over map accuracy
def reorder_C(
        y_pred: np.ndarray, 
        y_labl: np.ndarray, 
        adapt_values: bool = True, 
        method: Literal['all_arrangements','linear_sum'] = 'linear_sum'
) -> np.ndarray:
    """Re-order labels of predicted 2D classification map y_pred to align with y_labl (Ground-Truth)."""
    u_pred = np.unique(y_pred)
    u_labl = np.unique(y_labl)

    if method.lower() == 'all_arrangements':
        im_pred = np.asarray([y_pred.flatten()==val for val in u_pred], dtype=bool).T
        im_labl = np.asarray([y_labl.flatten()==val for val in u_labl], dtype=bool).T
        
        len_u = min(len(u_labl), len(u_pred))
        permutations = np.asarray(list(itertools.permutations(np.arange(len_u), len_u))).tolist()
        
        idx = permutations[0]
        acc = _map_accuracy(im_labl, im_pred[:,idx])
        for i in range(1,len(permutations)):
            new_idx = permutations[i]
            new_acc = _map_accuracy(im_labl, im_pred[:,new_idx])
            if new_acc > acc:
                idx = new_idx
                acc = new_acc
        
        new_y_pred = np.zeros_like(y_pred)
        for i in range(len(idx)): 
            new_val_i = u_labl[i] if adapt_values else u_pred[i]
            new_y_pred[y_pred==u_pred[idx[i]]] = new_val_i
        
        if len(u_pred) > len(u_labl):
            for k in set(range(len(u_pred))).difference(idx):
                matrix_k = [precision_score(y_true=(y_labl==val_j).flatten(), y_pred=(y_pred==k).flatten()) for val_j in u_labl]
                new_val_k = u_labl[np.argmax(matrix_k).item()] if adapt_values else u_pred[k]
                new_y_pred[y_pred==u_pred[k]] = new_val_k
        
        return new_y_pred.reshape(*y_pred.shape)

    elif method.lower() == 'linear_sum':
        score = precision_score if len(u_pred) != len(u_labl) else accuracy_score
    
        matrix = []
        for val_i in u_pred:
            mat_i = []
            for val_j in u_labl:
                val_ij = score(y_true=(y_labl==val_j).flatten(), y_pred=(y_pred==val_i).flatten())
                mat_i.append(val_ij)
            matrix.append(mat_i)
        
        idx, idy = linear_sum_assignment(matrix, maximize=True)
        
        new_y_pred = np.zeros_like(y_pred)
        
        if not adapt_values:
            idz = np.zeros_like(idx)
            if len(u_pred) <= len(u_labl):
                idz[np.argsort(idy)] = np.arange(len(idx))
            else:
                idz[np.argsort(idx)] = np.arange(len(idy))
        
        for k in range(len(idx)):
            if len(u_pred) <= len(u_labl):
                id_i = idx[k] # here, idx[k] == k
                id_j = idy[k] if adapt_values else idz[k]
            else:
                id_i = idx[k] if adapt_values else idz[k]
                id_j = idy[k]
            new_val_k = u_labl[id_j] if adapt_values else u_pred[id_j]
            new_y_pred[y_pred==u_pred[id_i]] = new_val_k
        
        if len(u_pred) > len(u_labl):
            for k in set(range(len(u_pred))).difference(idx if adapt_values else idz):
                new_val_k = u_labl[np.argmax(matrix[k]).item()] if adapt_values else u_pred[k]
                new_y_pred[y_pred==u_pred[k]] = new_val_k
        
        return new_y_pred

    else:
        raise ValueError("Argument 'method' must be either 'all_arrangements' or 'linear_sum'")
