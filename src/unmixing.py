import inspect
import warnings
import numpy as np

from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.svm import LinearSVC
from sklearn.multiclass import OneVsOneClassifier

from typing import Tuple, Literal, Optional, Callable, Protocol, Any

from src.polyset import scalar, norm, normed
from src.polyset import to_half_space_pairs, distribute_halfspaces, keep_only_necessary_pairs

from src.min_norm_point_PYTHON import minimum_norm_points_to_polyhedra_PYTHON # PYTHON VERSION
try: from min_norm_point import minimum_norm_points_to_polyhedra # C VERSION 
except Exception: warnings.warn("Failed loading the C version of the minimum-norm point function.")


#%%
# Model protocol for pixel clustering method
###

class ModelProtocol(Protocol):
    """Model protocol for pixel clustering method."""
    def __init__(self, n: int, init: Optional[Any]) -> None:
        """
        Parameters
        ----------
        n: int
            Number of classes / clusters.
        init: Any
            Initialization parameters.
        """
        ...
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray]):
        """
        Parameters
        ----------
        X: ndarray
            Fitting instances to classify / cluster.
        y: ignored
            Here for consistency with usual fitting functions.
        """
        ...
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X: ndarray
            Evaluation instances to classify / cluster.
        """
        ...


#%%
# Function for change of basis in the same vector space
###

def change_of_basis(data:np.ndarray, new_basis:np.ndarray) -> np.ndarray:
    """
    Parameters :
    * data: array of n data points of size m, with shape (n, m) ;
    * new_basis: array of m reference vectors of size m, with shape (m, m): must have full rank!\n
    Return data matrix expressed in new basis.
    """

    # get float resolution of new_basis
    if np.issubdtype(new_basis.dtype, np.floating):
        res = np.finfo(new_basis.dtype).resolution
    else:
        res = np.finfo(np.floating).resolution
    
    # if new_basis is NOT invertible, add a little shift
    Mat_det = np.linalg.det(new_basis)
    if np.abs(Mat_det) < res:
        add_res = 1e-3 * (new_basis.max() - new_basis.min())

        # shift data and new_basis by add_res
        data = data + 1e1 * add_res
        new_basis = new_basis + 1e1 * add_res

        # if new_basis is STILL NOT invertible, raise Error
        Mat_det = np.linalg.det(new_basis)
        if np.abs(Mat_det) < res:
            raise ValueError("New basis must have full rank!")

    # if new_basis is invertible, compute its inverse
    Mat_inv = np.linalg.inv(new_basis)

    return (Mat_inv.T @ data.T).T


#%%
# Functions to get reference distance vectors for change of basis by computing extremal (pure) vectors for each class. 
# Three implemented methods: 
# 1. extrema_minIn: extract extremal vectors among the observations by taking the vector of minimal distance to class i, for every i;
# 2. extrema_maxOut: compute extremal distance vectors by computing, for every i, its j-th component as the maximal distance to class j, for every j != i. 
#    The i-th component is the negated version of the minimal (positive) distances among the computed distances for vector i. 
# 3. extrema_maxOutPeak: same as extrema_maxOut, but with peak-distance point selection applied in each polyhedral set before extrema_maxOut. 
###

# Argument computation for the first method
def extrema_minIn_arg(distances:np.ndarray, n_elements_per_class:int=1) -> np.ndarray:
    """
    Parameters :
    * distances: ndarray of shape (n_data, n_classes) ;
    * n_elements_per_class: int = 1.
    """
    if n_elements_per_class == 1:
        return np.argmin(distances, axis=0)[np.newaxis]
    return np.argsort(distances, axis=0)[:n_elements_per_class]

# 1. First method: minimum-inside
def extrema_minIn(distances:np.ndarray, data:Optional[np.ndarray]=None, n_elements_per_class:int=1, element_mixing_func:Callable=np.mean) -> np.ndarray:
    """
    Extract extremal distance vectors among the observations by taking, for each class (polyhedra) i, 
    the distance vector x \in \mathbb{R}^{n_classes} of **minumum** signed distance (component) x_i **in** class i.\n
    Parameters :
    * distances: ndarray of shape (n_data, n_classes) ;
    * data (Optional): ndarray of shape (n_data, ndim) ;
    * n_elements_per_class: int = 1 ;
    * element_mixing_func: Callable with arguments (array, axis).\n
    If data is None, returns distance vectors, otherwise returns data vectors.
    """
    arg_extrema = extrema_minIn_arg(distances, n_elements_per_class)
    if data is None:
        return element_mixing_func(distances[arg_extrema], 0)
    return element_mixing_func(data[arg_extrema], 0)

# 2. Second method: maximum outside
def extrema_maxOut(distances:np.ndarray) -> np.ndarray:
    """
    Construct new extremal distance vectors not necessarily in the observations by taking, for each class 
    (polyhedra) i, the j-th component of vector x \in \mathbb{R}^{n_classes}, j != i, as the **maximum** 
    signed distance (component) to polyhedron j among the data **out** of i-complement (i.e., **in** class i).\n
    Parameters :
    * distances: ndarray of shape (n_data, n_classes).\n
    Returns constructed extremal distance vectors.
    """
    dis_extrema = []
    for i in range(distances.shape[-1]):
        in_polyhedron = distances[:,i] <= 0
        if in_polyhedron.sum() == 0:
            # Exception! => take the distance vector among all the data with the smallest distance to class i
            dis_i = distances[int(np.argmin(distances[:,i]))]
        else: # Take the maximal distances to every class j \neq i among the data in class i
            data_dis_i = distances[in_polyhedron] # take only the distance vectors that are inside polyhedron i
            dis_i = np.max(data_dis_i, axis=0)
            dis_i[i] = - np.min(dis_i[list(set(range(dis_i.shape[0])).difference({i}))])
        dis_extrema.append(dis_i)
    return np.asarray(dis_extrema)

# Peak vector selection for the third method
def peakSelection(vectors:np.ndarray, selection_prop:float=0.20, selection_mode:Literal['quantity','proximity']='proximity') -> list:
    """
    Peak vector selection function.\n
    Parameters :
    * vectors: ndarray of shape (n_data, ndim).
    * selection_prop: either (i) proportion of selected distance vectors if selection_mode=='quantity', 
      or (ii) ratio of the maximal projected distance over the unit vector that vectors must satisfy to be 
      selected if selection_mode=='proximity'.
    * selection_mode: mode for distance vectors' selection.\n
    Returns indices of selected vectors.
    """
    # ===== Peak selection ======
    weights = np.max(vectors, axis=0)
    weighted = vectors / weights
    unit_projections = np.sum(weighted, axis=1) / np.sqrt(weighted.shape[1]) # scalar(weighted, normed((1,1,...,1)))
    if selection_mode == 'quantity':
        threshold = max(1, int(selection_prop * unit_projections.shape[0]))
        arg_vec = np.argsort(unit_projections)[-threshold:]
    else:
        threshold = np.max(unit_projections) * (1 - selection_prop)
        arg_vec = np.argwhere(unit_projections >= threshold).flatten()
    # ===========================
    return list(arg_vec)

# 3. Third method: maximum outside with peak vector selection
def extrema_maxOutPeak(distances:np.ndarray, selection_prop:float=0.20, selection_mode:Literal['quantity','proximity']='proximity') -> np.ndarray:
    """
    Peak vector selection over the maxOut method.\n
    Construct new extremal distance vectors not necessarily in the observations by taking, for each class 
    (polyhedra) i, the j-th component of vector x \in \mathbb{R}^{n_classes}, j != i, as the **maximum** 
    signed distance (component) to polyhedron j among the data **out** of i-complement (i.e., **in** class i).\n
    Parameters :
    * distances: ndarray of shape (n_data, n_classes).
    * selection_prop: either (i) proportion of selected distance vectors if selection_mode=='quantity', 
      or (ii) ratio of the maximal projected distance over the unit vector that vectors must satisfy to be 
      selected if selection_mode=='proximity'.
    * selection_mode: mode for distance vectors' selection.\n
    Returns constructed extremal distance vectors.
    """
    dis_extrema = []
    for i in range(distances.shape[-1]):
        in_polyhedron = distances[:,i] <= 0
        if in_polyhedron.sum() == 0:
            # Exception! => take the distance vector among all the data with the smallest distance to class i
            dis_i = distances[int(np.argmin(distances[:,i]))]
        else: # Take the maximal distances to every class j \neq i among the data in class i
            data_dis_i = distances[in_polyhedron] # take only the distance vectors that are inside polyhedron i
            # ===== Peak selection ======
            data_vec_i = np.concatenate([data_dis_i[:,:i], data_dis_i[:,i+1:]], axis=-1) # remove (negative) distance to polyhedron i
            arg_vec = peakSelection(data_vec_i, selection_prop=selection_prop, selection_mode=selection_mode)
            selected_dis_i = data_dis_i[arg_vec]
            # ===========================
            dis_i = np.max(selected_dis_i, axis=0)
            dis_i[i] = - np.min(dis_i[list(set(range(dis_i.shape[0])).difference({i}))])
        dis_extrema.append(dis_i)
    return np.asarray(dis_extrema)

# Main function for extremal (pure) distance vector computation
def main_extrema(distances:np.ndarray, method:Literal['minIn','maxOut','maxOutPeak']='maxOut') -> np.ndarray:
    """
    Main function for extremal (pure) distance vector computation. 
    Three methods are implemented:
    1. 'minIn': extract extremal vectors among the observations by taking the vector of minimal distance to class i, for every i.
    2. 'maxOut': compute extremal distance vectors by computing, for every i, its j-th component as the maximal distance to class j, for every j != i.
    3. 'maxOutPeak': same as maxOut, but with peak-distance point selection applied in each polyhedral set before maxOut.

    Parameters
    ----------
    * distances: ndarray of shape (n_data, n_classes).
    * method: str, in {'minIn','maxOut','maxOutPeak'}.\n
    See additional parameters for a more personalized 'minIn' usage.
    
    Reurns
    ------
    extrema: ndarray of shape (n_classes, n_classes)
    """
    if method.lower() not in {'minin','maxout','maxoutpeak'}:
        raise ValueError("Main extrema method must be either 'minIn', 'maxOut' or 'maxOutPeak'")
    if method.lower() == 'minin':
        return extrema_minIn(distances)
    elif method.lower() == 'maxout':
        return extrema_maxOut(distances)
    return extrema_maxOutPeak(distances)


#%%
# Linear SVM for polyhedral partition fitting
###

# OvO multiclass unbiased linear-SVC (linear SVM classifier) solver
def ovo_linear_SVC(X:np.ndarray, y:np.ndarray, biased:bool=True) -> Tuple[np.ndarray, np.ndarray]:
    """
    OvO linear SVC solver. 
    * X: (n_samples, n_features), floats;
    * y: (n_sample,), only integers;
    * biased: whether to fit the intercept (bias).\n
    Returns non-normed (w, b). 
    """
    ## To be more precise:
    #   -> smaller 'tol' value (usually in [1e-5, 1e-2])
    #   -> larger 'max_iter' value (usually in [1,000, 10,000])
    ## To be faster:
    #   -> larger 'tol' value
    #   -> smaller 'max_iter' value
    #   -> dual = False (to solve the primal SVM problem)
    #   -> loss = 'squared_hinge' (+ is more stable in terms of convergence)
    ## To be more robust against noise/outliers:
    #   -> loss = 'hinge' (Warning: only when dual=True)
    ## To be closer to a standard SVM behaviour:
    #   -> dual = True
    #   -> loss = 'hinge'
    #   -> tol = 1e-3
    base = LinearSVC(
        fit_intercept = biased, 
        dual = True, 
        penalty = 'l2', 
        loss = 'hinge', 
        max_iter = 10000, 
        tol = 1e-3, 
        C = 1.0, 
    )
    clf = OneVsOneClassifier(base, n_jobs=1).fit(X=X, y=y)
    v, s = [], []
    for est in clf.estimators_:
        w = est.coef_.ravel()
        b = est.intercept_
        v.append(w)
        s.append(b)
    v = np.asarray(v)
    s = np.asarray(s)
    return v, s

# Main function to fit a polyhedral(-cone) partition of the spectral space over labeled data using a (unbiased) linear SVM
def fit_polyhedral_partition(
        data: np.ndarray, 
        labels: np.ndarray, 
        method: Literal['biased_SVM','unbiased_SVM','auto'] = 'auto', 
        sample_prop: float = 1.0, 
        sample_arg_priority: Optional[np.ndarray] = None, 
        min_H_description: bool = True
) -> list[np.ndarray]:
    """
    Compute a polyhedral(-cone) partitioning of the spectral space using a (unbiased) 
    linear Support Vector Machine to pairwise separate the labeled class clusters.
    * data: array of observed spectral data, of shape (n, d).
    * labels: array of data class labels, of shape (n, d).
    * method: whether to fit a biased or unbiased linear SVM.
    * sample_prop: proportion of the random sample to extract from the data. 
    * sample_arg_priority: array of shape (n,) of data indices to extract first for data sampling. 
    * min_H_description: whether to compute the minimal H-description of polyhedra.\n
    Returns the list of class polyhedra expressed as intersections of halfspaces.
    """

    # Check whether the SVM is biased
    if method.lower() == 'auto'.lower():
        ndim = data.shape[-1]
        n_classes = len(np.unique(labels))
        # biased iff 
        # (i) unmixing problem is underdetermined (n_bands < n_classes) or 
        # (ii) data matrix is ill-conditioned (rank data ~<~ n_classes)
        biased = ndim < n_classes or np.linalg.matrix_rank(data) < n_classes # rank tol = ?
    else:
        biased = 'unbiased' not in method.lower()

    # Uniform random sample extraction
    SVM_data, SVM_idx = extract_random_sample(
        data = data, 
        prop = sample_prop, 
        return_indices = True, 
        labels = labels, 
        weight = None, 
        arg_priority = sample_arg_priority)
    SVM_labels = labels[SVM_idx]

    # Mean point computation for each class (for half-space distribution, as regions are convex)
    class_means = np.asarray([np.mean(SVM_data[SVM_labels==i], axis=0) for i in np.unique(SVM_labels)])

    # (Un)biased One-VS-One linear Support Vector Classifier (SVC)
    v, s = ovo_linear_SVC(X=SVM_data, y=SVM_labels, biased=biased)

    # From separation hyperplanes to half-spaces
    h_hyperplanes = to_half_space_pairs(v, s)
    h_half_spaces = list(distribute_halfspaces(h_hyperplanes, class_means))
    
    # Minimal H-description computation for convex polyhedra
    if min_H_description:
        for _ in range(len(h_half_spaces)):
            h0 = h_half_spaces.pop(0)
            hp = keep_only_necessary_pairs(h0)
            h_half_spaces.append(hp)
    
    return h_half_spaces


#%%
# Function to extract random samples. It can be weighted by the Mahalanobis distances to cluster centers
###

# Replace eigenvalues of value 0 by val (to give volume to the covariance ellipsoid)
def inflate_null_eigs(Sigma:np.ndarray, val:float=1.0, tol:Optional[float]=None) -> np.ndarray:
    Sigma = (Sigma + Sigma.T) / 2
    lam, Q = np.linalg.eigh(Sigma)
    lam_max = np.max(lam) if lam.size else 0.0
    if tol is None:
        eps = np.finfo(lam.dtype).eps
        tol = eps * len(lam) * max(lam_max, 1.0)
    mask_zero = lam <= tol
    lam_new = lam.copy()
    lam_new[mask_zero] = val
    Sigma_prime = (Q * lam_new) @ Q.T
    return (Sigma_prime + Sigma_prime.T) / 2

# Mahalanobis distance function
def mahalanobis(data:np.ndarray, mu:Optional[np.ndarray]=None, sigma:Optional[np.ndarray]=None, inflate:bool=True) -> np.ndarray:
    """
    Compute Mahalanobis distances.
    * data: (n_data, ndim)
    * mu: (ndim,)
    * sigma: (ndim, ndim)
    """
    if mu is None:
        mu = np.mean(data, axis=0)
    if sigma is None:
        sigma = np.cov(data.T)
    if inflate:
        sigma = inflate_null_eigs(sigma)
    elif np.isclose(np.linalg.det(sigma),0):
        raise ValueError("Covariance matrix 'sigma' is not invertible")
    x = data - mu
    Sinv = np.linalg.inv(sigma)
    return np.sqrt(scalar(x,np.matmul(Sinv, x.T).T))

# Probability weight function based on Mahalanobis distances
def distance_weights(
        data:np.ndarray, 
        mu:Optional[np.ndarray]=None, 
        sigma:Optional[np.ndarray]=None, 
        dis_min:float=1.0, 
        dis_std:float=100.0, 
        power:float=4.0
) -> np.ndarray:
    """
    Compute data weights using Mahalanobis distances.
    * data: (n_data, ndim)
    * mu: (ndim,)
    * sigma: (ndim, ndim)
    * dis_min: float
    * dis_std: float
    * power: float
    """
    if power == 0:
        return np.ones(data.shape[0]) / data.shape[0]
    distances = mahalanobis(data, mu, sigma)
    distances_pow = np.power(distances, power)
    weights = distances_pow * dis_std + dis_min
    return weights / weights.sum()

# Function to extract a random sample from data (used to compute separation hyperplanes)
def extract_random_sample(
        data:np.ndarray, 
        prop:float, 
        return_indices:bool=False, 
        labels:Optional[np.ndarray]=None, 
        weight:Optional[float|np.ndarray]=None, 
        arg_priority:Optional[np.ndarray]=None
) -> np.ndarray|Tuple[np.ndarray, np.ndarray]:
    """
    Function to extract a random sample from data.
    * data: (n_data, ndim)
    * prop: float
    * return_indices: bool
    * labels: (n_data,) of ints
    * weight: float, or (n_data,) of floats
    * arg_priority: array of shape (n_data,) of data indices to extract first for data sampling.\n
    If arg_priority is not given, extract uniformly over all the data, with at least one pixel per 
    class if labels are given. Otherwise, extract uniformly over data[arg_priority] as a priority, 
    and complete by extracting elsewhere if the extracted proportion is not sufficient yet. 
    Total number of pixels in sample: |prop| * n.
    """
    
    n = data.shape[0]
    if prop is None or np.abs(prop) >= 1.0:
        if return_indices:
            return data, np.arange(n)
        return data
    prop = np.abs(prop)
    
    if arg_priority is not None:
        
        n_sample_pixels = int(n*prop)
        relative_prop = n_sample_pixels / len(arg_priority)
        arg_relative = list(arg_priority)
        
        data_in, idx_iin = extract_random_sample(
            data = data[arg_relative], 
            prop = relative_prop, 
            return_indices = True, 
            labels = labels[arg_relative] if labels is not None else None, 
            weight = weight[arg_relative] if weight is not None else None
        )
        indices_in = np.asarray(arg_relative)[list(idx_iin)]
        
        if n_sample_pixels > len(arg_priority):
            
            remain_prop = (n_sample_pixels - len(arg_priority)) / (n - len(arg_priority))
            arg_remain = list(set(np.arange(n)).difference(set(arg_priority)))
            
            data_out, idx_oout = extract_random_sample(
                data = data[arg_remain], 
                prop = remain_prop, 
                return_indices = True, 
                labels = labels[arg_remain] if labels is not None else None, 
                weight = weight[arg_remain] if weight is not None else None
            )
            indices_out = np.asarray(arg_remain)[list(idx_oout)]

            data_in = np.concatenate([data_in, data_out], axis=0, dtype=data_in.dtype)
            indices_in = np.concatenate([indices_in, indices_out], axis=0, dtype=indices_in.dtype)
        
        if return_indices:
            return data_in, indices_in
        return data_in
    
    if labels is None:

        if weight is None:
            indices = np.random.choice(n, size=max(int(n*prop),1), replace=False)
        
        elif np.iterable(weight):
            indices = np.random.choice(n, size=max(int(n*prop),1), replace=False, p=weight)
        
        else:
            weights = distance_weights(data, power=float(weight))
            indices = np.random.choice(n, size=max(int(n*prop),1), replace=False, p=weights)
        
    else:
        
        l_indices = []
        cls = np.unique(labels) # is already sorted!
        
        for i in range(len(cls)):
            boo = labels==cls[i]
            idx = np.argwhere(boo).flatten()
            n_i = len(idx)
            
            if weight is None:
                ide = np.random.choice(n_i, size=max(int(n_i*prop),1), replace=False)
            
            elif np.iterable(weight):
                ide = np.random.choice(n_i, size=max(int(n_i*prop),1), replace=False, p=weight[boo])
            
            else:
                weights_i = distance_weights(data[boo], power=float(weight))
                ide = np.random.choice(n_i, size=max(int(n_i*prop),1), replace=False, p=weights_i)
            
            l_indices.append(idx[list(ide)])
        
        indices = np.concatenate(l_indices, axis=0)
    
    if return_indices:
        return data[indices], indices
    return data[indices]


#%%
# Functions to compute the nearest points in convex polyhedra, and the associated signed distances
###

# Computes the Euclidean distance between points and polyhedral sets
def distance_to_polyhedra(data:np.ndarray, h:list[np.ndarray], python:bool=True, infos:bool=True) -> np.ndarray:
    """
    * data: ndarray of points in ndim-dimensional real vector space,
     with shape (n_samples, ndim) or (ndim,) ;
    * h: list of polyhedra represented as intersection of half_spaces described by couples (c,v),
     with shape n_classes * (n_half_spaces, 2, ndim) or (n_half_spaces, 2, ndim).\n
    Returns ndarray of Euclidean distances from data to each class polyhedra, 
    with shape (n_samples, n_classes) or (n_classes,) or (n_samples,) or scalar.
    """
    if python:
        min_n_pts = minimum_norm_points_to_polyhedra_PYTHON(
            data = data, 
            h = h, 
            infos = infos
        ) # PYTHON VERSION
    else:
        try: minimum_norm_points_to_polyhedra
        except: raise Exception("The C function 'minimum_norm_points_to_polyhedra' has not been properly imported")
        min_n_pts, _ = minimum_norm_points_to_polyhedra(
            V = [h[i][:,1] for i in range(len(h))], 
            s = [scalar(h[i][:,0],h[i][:,1]) for i in range(len(h))], 
            points = data, 
            infos = infos
        ) # C VERSION
    distances = norm(min_n_pts - data[:,np.newaxis], keepdims=False)
    return distances

# Compute negative distances from points to polyhedral sets
def add_negative_distance(data:np.ndarray, h:np.ndarray, distances:Optional[np.ndarray]=None) -> np.ndarray:
    """
    Adds negative parts of signed distances to 'distance' array. If such array is not given, a new one is created.
    """
    n_samples = data.shape[0] # = distances.shape[0]
    n_classes = len(h) # = distances.shape[1]
    if distances is None:
        new_distances = np.zeros(shape=(n_samples, n_classes), dtype=data.dtype)
    else:
        new_distances = distances.copy()
    eps = np.finfo(data.dtype).resolution
    for c in range(n_classes):
        h_class = h[c]
        max_dist = np.max(scalar(data[np.newaxis] - h_class[:,0,np.newaxis], h_class[:,1,np.newaxis]), axis=0)
        neg_data = max_dist < - eps
        new_distances[neg_data,c] = max_dist[neg_data]
    return new_distances

# Compute signed distances (positive and negative) from points to polyhedra
def signed_distances_to_polyhedra(data:np.ndarray, h:list[np.ndarray], python:bool=True, infos:bool=True) -> np.ndarray:
    """
    * data: ndarray of points in ndim-dimensional real vector space,
     with shape (n_samples, ndim) or (ndim,) ;
    * h: list of polyhedra represented as intersection of half_spaces described by couples (c,v),
     with shape n_classes * (n_half_spaces, 2, ndim) or (n_half_spaces, 2, ndim).\n
    Returns ndarray of signed Euclidean distances from data to each class polyhedra, 
    with shape (n_samples, n_classes) or (n_classes,) or (n_samples,) or scalar.
    """
    try: arr2D = np.asarray(h[0])
    except: raise ValueError("Cannot convert sub-arrays of h into numpy arrays")
    if arr2D.ndim == 2: h = [h]
    elif arr2D.ndim != 3: raise ValueError("h must be of dimension 3 or 4")
    distances = distance_to_polyhedra(data, h, python, infos)
    distances = add_negative_distance(data, h, distances)
    return distances


#%%
# Supervised endmember and abundance estimation by regularized matrix pseudo-inversion.
###

# Endmember M estimation from Y and A
def estimate_endmembers_ridge(Y:np.ndarray, A:np.ndarray, lambd:float=1e-5) -> np.ndarray:
    """
    Estimates M by solving: min_M ||Y - MA||_F^2 + lambda * ||M||_F^2

    Parameters:
        Y : ndarray (n, d) - observed spectra
        A : ndarray (n, m) - abundances
        lambd : float - regularization coefficient (λ)

    Returns:
        M_hat : ndarray (m, d) - endmembers estimés
    """
    pseudo_inverse = A @ np.linalg.inv(A.T @ A + lambd * np.eye(A.shape[1]))
    M_hat = Y.T @ pseudo_inverse
    return M_hat.T

# Abundance A estimation from Y and M
def estimate_abundances_ridge(Y:np.ndarray, M:np.ndarray, lambd:float=1e-5) -> np.ndarray:
    """
    Estimates A by solving: min_A ||Y - MA||_F^2 + lambda * ||A||_F^2

    Parameters:
        Y : ndarray (n, d) - observed spectra
        M : ndarray (m, d) - endmembers
        lambd : float - regularization coefficient (λ)

    Returns:
        A_hat : ndarray (n, m) - endmembers estimés
    """
    pseudo_inverse = np.linalg.inv(M @ M.T + lambd * np.eye(M.shape[0])) @ M
    A_hat = pseudo_inverse @ Y.T
    return A_hat.T


#%%
# Function to turn signed distances into an abundance map via projection on the probability simplex
###

def simplex_frontier_hyperplanes(m:int, pdis:float=1.0) -> np.ndarray:
    v = (np.eye(m, m) - np.ones(m) / m) / np.sqrt((m - 1) / m)
    b = np.full((m, 1), fill_value = -1 / np.sqrt((m - 1) * m))
    return - np.append(v, pdis * b, axis = -1)

def from_Vb_to_CV(Vb:np.ndarray) -> np.ndarray:
    V = Vb[:, np.newaxis, :-1]
    C = V * Vb[:, np.newaxis, -1:]
    return np.append(C, V, axis = 1)

def simplex_projection(x:np.ndarray, pdis:float=1.0, python:bool=True) -> np.ndarray:
    m = x.shape[-1]
    y = x + (pdis - np.sum(x, axis=-1, keepdims=True)) * np.ones(m) / m
    hVb = simplex_frontier_hyperplanes(m, pdis)
    if python:
        p = minimum_norm_points_to_polyhedra_PYTHON(y, from_Vb_to_CV(hVb), infos=False)
    else:
        p,_ = minimum_norm_points_to_polyhedra(hVb[:,:-1], hVb[:,-1], y, infos=False)
    return p / pdis

def to_probability(x:np.ndarray, saturation:float=1.0, python:bool=True) -> np.ndarray:
    """
    Function that projects x data onto the probability simplex.
    """
    exp_x = simplex_projection(x * saturation, python=python)

    exp_sum = np.sum(exp_x, axis=-1, keepdims=True)
    exp_sum_inf  = (exp_sum == np.inf).reshape(exp_sum.shape[:-1])
    exp_sum_zero = (exp_sum == 0     ).reshape(exp_sum.shape[:-1])
    
    exp_x_max = exp_x == exp_x.max(axis=-1, keepdims=True)
    exp_x[exp_sum_inf] = exp_x_max[exp_sum_inf]#.astype(exp_x)
    exp_sum[exp_sum_inf] = np.sum(exp_x_max[exp_sum_inf], axis=-1, keepdims=True)

    exp_x[exp_sum_zero] = 1
    exp_sum[exp_sum_zero] = x.shape[-1]

    return exp_x / exp_sum


#%%
# Main Polyhedral Unmixing Model class
###

class PolyhedralUnmixingModel:
    """
    Polyhedral unmixing model.

    Given hyperspectral data, the model determines both the endmembers M and abundances A 
    from any given clustering or classification method of the input data via a 
    polyhedral(-cone) partitioning of the spectral space via a (unbiased) linear SVM.
    """
    
    def __init__(
            self, 
            normalize: bool = True, 
            PCA_ndim: Optional[int] = None, 
            keep_normalized: bool = True, 
            only_python: bool = True, 
            saturation: float = 1.0, 
            verbose: bool = True
    ) -> None:
        """
        Polyhedral Unmixing Model initialization.

        Parameters
        ----------
        * normalize : bool, default: True
            (Pre-processing) Whether to normalize spectral luminance by projecting 
            the data onto the unit sphere.
        * PCA_ndim : int, optional, default: None
            (Pre-processing) If given, dimension of the lower-dimensional spectral 
            subspace determined via PCA.
        * keep_normalized : bool, default: True
            Whether to apply matrix pseudo-inversions over the original or 
            normalized version of the input.
        * only_python : bool, default: True
            Whether to use the Python or C code for the nearest-point problem. 
            The C version is about 100x faster.
        * saturation : float, default: 1.0
            Saturation scale hyperparameter. 
            It pushes the distance points closer to the simplex edges.
        * verbose : bool, default: True
            Whether to print progress and computation details.
        """

        self.normalize = normalize
        self.PCA_ndim: int = PCA_ndim
        self.PCA: PCA = None

        self.additional_preprocessing: Optional[Callable] = None
        
        self.clustering_method: Literal['GMM','k-means','__given__'] = None
        self.clustering_prop: Optional[float] = None
        self.clustering_classes: Optional[int|np.ndarray] = None

        self.clustering_model: ModelProtocol | Callable | Optional[Callable] = None

        self.arg_clustered_pixels: Optional[np.ndarray] = None
        
        self.polyhedral_method: Literal['biased_SVM','unbiased_SVM','auto'] = None
        self.polyhedral_prop: float = None

        self.only_python: bool = only_python

        self.saturation: float = saturation
        self.keep_normalized: bool = keep_normalized

        self.polyhedra: list = None
        self.pure_distances: np.ndarray = None
        
        self.labels: np.ndarray = None
        self.n_endmembers: int = None
        self.endmembers: np.ndarray = None

        self.verbose: bool = verbose
    
    def preprocess(
            self, 
            data: np.ndarray, 
            normalize: bool = None, 
            PCA_ndim: int = None, 
            additional: Callable = None,
            mode: Literal['fit','predict','indep'] = 'indep'
    ) -> np.ndarray:
        """
        Function to pre-process spectral data, via: \n
        (i) luminance normalization, \n
        (ii) dimension reduction, and \n
        (iii) additional pre-processing.
        
        Parameters
        ----------
        * data : ndarray
            Spectral data of shape (..., n_bands).
        * normalize : bool or None, default: None
            Whether to normalize spectral luminance (projection on the unit sphere).
        * PCA_ndim : int or None, default: None
            Dimension of the new spectral space given by PCA on the data.
        * additional : Callable or None, default: None
            Additional pre-processing function applied at the end of the chain.
        * mode : str in {'fit','predict','indep'}, default: 'indep'
            Whether to fit or predict the pre-processing model, or use it independently.

        Returns
        -------
        new_data: ndarray
            Pre-processed data.
        """

        try: data = np.asarray(data)
        except: raise ValueError("Parameter 'data' must be a ndarray")
        
        if data.ndim == 1: data = data[..., np.newaxis]
        elif data.ndim == 0: raise ValueError("Parameter 'data' must be of non-zero dimension")

        try: mode = str(mode)
        except: raise ValueError("Parameter 'mode' must be a string")
        
        if mode.lower() not in {'fit','predict','indep'}:
            raise ValueError("Parameter 'mode' must be either 'fit', 'predict' or 'indep'")
        
        predict_mode: bool = bool(mode.lower() == 'predict')

        if predict_mode:
            if normalize is not None or PCA_ndim is not None or additional is not None:
                warnings.warn("Pre-processing function is in 'predict' mode. Given parameters ignored.")
            if self.PCA is not None and type(self.PCA) is not PCA:
                raise ValueError("Parameter 'self.PCA' is of unknown type. Must be either None or PCA.")
            
            normalize  = self.normalize
            PCA_ndim   = True if self.PCA is not None else None
            additional = self.additional_preprocessing
        
        try: normalize = bool(normalize) if normalize is not None else None
        except: raise ValueError("Parameter '" + predict_mode * "self." + "normalize' must be a boolean|None")

        try: PCA_ndim = int(PCA_ndim) if PCA_ndim is not None else None
        except: raise ValueError("Parameter '" + predict_mode * "self." + "PCA_ndim' must be an integer|None")

        if additional is not None and not callable(additional):
            raise ValueError("Parameter '" + ("self.additional_preprocessing" if predict_mode else "additional") + "' must be callable|None")

        new_data = data.copy()

        # 1. luminance normalization
        if normalize is not None and normalize:
            new_data = normed(new_data)

        # 2. dimensionality reduction via PCA
        if PCA_ndim is not None and PCA_ndim > 0:
            tdata = new_data.reshape(int(np.prod(data.shape[:-1])), data.shape[-1])
            if predict_mode:
                pca = self.PCA
            else:
                pca = PCA(n_components=PCA_ndim)
                pca.fit(tdata)
            new_tdata = pca.transform(tdata)
            new_data = new_tdata.reshape(*data.shape[:-1], new_tdata.shape[-1])

        # 3. additional pre-processing
        if additional is not None:
            new_data = additional(new_data)

        if mode.lower() == 'fit':
            self.normalize = normalize
            self.PCA_ndim = PCA_ndim
            self.PCA = pca if PCA_ndim is not None and PCA_ndim > 0 else None
            self.additional_preprocessing = additional
        
        return new_data

    def compute_labels(
            self, 
            data: np.ndarray, 
            method: ModelProtocol | Callable | Literal['k-means','GMM'] = None, 
            classes: int | np.ndarray = None, 
            sample_prop: float = None, 
            mode: Literal['fit','predict','indep'] = 'indep'
    ) -> np.ndarray:
        """
        Function to compute labels given a clustering method.
        
        Parameters
        ----------
        * data : ndarray
            Spectral data of shape (..., n_bands).
        * method : ModelProtocol, Callable or in {'k-means','GMM'}, default: None
            Method for data clustering. 
            If ModelProtocol, it must respect the construction of ModelProtocol.
        * classes : int (> 0) or ndarray, default: None
            If int, number of classes to consider for the clustering. 
            Otherwise, list of either n class spectra or n coordinates in data. 
        * sample_prop : float, in (0,1], default: None
            Proportion of the random sample to extract from input data 
            on which is fit the chosen clustering method.
        * mode : str in {'fit','predict','indep'}, default: 'indep'
            Whether to fit or predict the labelling model, or use it independently.

        Returns
        -------
        labels: ndarray
            Computed class labels.
        """

        try: data = np.asarray(data)
        except: raise ValueError("Parameter 'data' must be a ndarray")
        
        if data.ndim == 1: data = data[..., np.newaxis]
        elif data.ndim == 0: raise ValueError("Parameter 'data' must be of non-zero dimension")

        try: mode = str(mode)
        except: raise ValueError("Parameter 'mode' must be a string")
        
        if mode.lower() not in {'fit','predict','indep'}:
            raise ValueError("Parameter 'mode' must be either 'fit', 'predict' or 'indep'")
        
        predict_mode: bool = bool(mode.lower() == 'predict')

        if predict_mode:
            if method is not None or classes is not None or sample_prop is not None:
                warnings.warn("Labelling function is in 'predict' mode. Given parameters ignored.")
            if self.clustering_method != '__given__' and self.clustering_model is None:
                warnings.warn("Labelling function cannot be predicted, as it has not been fitted yet. None returned.")
                return None
            if self.clustering_method == '__given__':
                warnings.warn("Labelling function cannot be predicted, as data labels have directly been given during fitting. None returned.")
                return None
            if not(inspect.isclass(self.clustering_model) or callable(self.clustering_model)):
                raise ValueError("Parameter 'self.clustering_model' is of unknown type. Must be either a ModelProtocol or Callable.")
            
            method  = self.clustering_method
            classes = self.clustering_classes
            sample_prop = self.clustering_prop

        elif method is None or classes is None: raise ValueError("Parameters 'method' and 'classes' must both be given!")
        
        if not(inspect.isclass(method) or callable(method)):
            try: method = str(method) if method is not None else None
            except: raise ValueError("Parameter '" + predict_mode * "self." + "method' must be a string|None")
            if method not in {'k-means','GMM', None}: 
                raise ValueError("Parameter '" + ("self.clustering_method" if predict_mode else "method") + "' must be callable or in \{'k-means','GMM'\}")

        try: classes = np.asarray(classes) if np.iterable(classes) else int(classes) if classes is not None else None
        except: raise ValueError("Parameter '" + ("self.clustering_classes" if predict_mode else "classes") + "' must be a ndarray or an integer|None")

        try: sample_prop = float(sample_prop) if sample_prop is not None else 1.0
        except: raise ValueError("Parameter '" + ("self.clustering_prop" if predict_mode else "sample_prop") + "' must be a float|None")
        
        if sample_prop <= 0 or sample_prop > 1: raise ValueError("Parameter '" + ("self.clustering_prop" if predict_mode else "sample_prop") + "' must be in (0,1]")

        # Compute type of classes
        if type(classes) is np.ndarray:
            if classes.ndim > 2: 
                raise ValueError("Iterable parameter '" + ("self.clustering_classes" if predict_mode else "classes") + "' must be of dimension 2, with shape (n_classes, n_bands) floats, or (n_classes, data.ndim-1) ints")
            elif classes.ndim == 1:
                if data.shape[-1] == 1: classes = classes[:,np.newaxis] # n grayscale classes
                elif classes.shape[0] == data.shape[-1]: classes = classes[np.newaxis] # 1 spectrum
                elif classes.shape[0] == data.ndim - 1 : classes = classes[np.newaxis] # 1 coordinate
                else: raise ValueError("Iterable parameter '" + ("self.clustering_classes" if predict_mode else "classes") + "' must be of dimension 2, with shape (n_classes, n_bands) floats, or (n_classes, data.ndim-1) ints")
            if classes.shape[-1] == data.shape[-1]: # (probably) spectra or (less probably) coordinates
                if classes.shape[-1] == data.ndim - 1 and np.issubdtype(classes.dtype, np.integer) and classes.dtype != data.dtype: # coordinates
                    class_type = 2
                else: # spectra
                    if classes.dtype != data.dtype:
                        try: classes = classes.astype(data.dtype)
                        except: raise ValueError("Iterable parameter '" + ("self.clustering_classes" if predict_mode else "classes") + "', if list of spectral signatures, must be of same dtype as input data")
                    class_type = 1
            elif classes.shape[-1] == data.ndim - 1: # coordinates
                if not np.issubdtype(classes.dtype, np.integer):
                    classes = classes.astype(np.int32)
                class_type = 2
            else: raise ValueError("Iterable parameter '" + ("self.clustering_classes" if predict_mode else "classes") + "' must either respresent spectral signatures or coordinates on the input spectral data")
        else: class_type = 0

        # Clustering: label the pixels into m clusters
        tdata = data.reshape(int(np.prod(data.shape[:-1])), data.shape[-1])

        if predict_mode:
            model = self.clustering_model
        else:
            # Deduce ('init','n_classes') parameters from ('classes','class_type')
            if class_type == 0: # [0] 'classes' is integer: number of classes
                init = classes
                n_classes = classes
            elif class_type == 1: # [1] 'classes' is list of spectra: ndarray of same dtype as input image
                init = classes
                n_classes = classes.shape[0]
            else: # [2] 'classes' is list of coordinates (on input image): ndarray of integers
                init = np.asarray([data[tuple(coord)] for coord in classes])
                n_classes = classes.shape[0]
            
            # Method: either in {'k-means', 'GMM'} or personal ModelProtocol
            if type(method) is str:
                
                if method == 'k-means':
                    center_init = init if type(init) is np.ndarray else 'k-means++'
                    model = KMeans(n_clusters=n_classes, init=center_init, n_init='auto')

                elif method == 'GMM':
                    means_init = init if type(init) is np.ndarray else None
                    model = GaussianMixture(n_components=n_classes, init_params='k-means++', n_init=10, means_init=means_init)
                
                else:
                    raise ValueError("Parameter '" + ("self.clustering_method" if predict_mode else "method") + "', if string, must be either 'k-means' or 'GMM'.")
            
            else: 
                if inspect.isclass(method):
                    try: model = method(n=n_classes, init=init)
                    except:
                        try: model = method(n_classes, init)
                        except:
                            try: model = method(n=n_classes)
                            except:
                                try: model = method(n_classes)
                                except: 
                                    try: model = method()
                                    except: raise ValueError("Cannot initialize clustering model. Please check ModelProtocol class functions and arguments.")
                else: model = method
            
            sample, sample_idx = extract_random_sample(tdata, prop=sample_prop, return_indices=True)

            if hasattr(model, "fit") and callable(getattr(model, "fit")):
                try: model.fit(X=sample)
                except: 
                    try: model.fit(sample)
                    except: raise TypeError("Cannot fit the clustering model using its 'fit' function. Please check ModelProtocol functions and arguments.")

        if hasattr(model, "predict") and callable(getattr(model, "predict")):
            try: labels = model.predict(X=tdata)
            except: 
                try: labels = model.predict(tdata)
                except: raise TypeError("Cannot predict data with the clustering model using its 'predict' function. Please check ModelProtocol functions and arguments.")
        else: # is instance or function
            try: labels = model(tdata)
            except: raise ValueError("Cannot call '" + ("self.clustering_model" if predict_mode else "method") + "' instance or function on given data.")

        labels = labels.reshape(*data.shape[:-1])

        if mode.lower() == 'fit':
            self.clustering_method = method
            self.clustering_classes = classes
            self.clustering_prop = sample_prop
            self.clustering_model = model
            self.arg_clustered_pixels = sample_idx
            self.labels = labels

        return labels
    
    def get_labels(
            self, 
    ) -> np.ndarray:
        return self.labels

    def fit(
            self, 
            data: np.ndarray, 
            labels: Optional[np.ndarray] = None, 
            clustering_method: Optional[ModelProtocol | Callable | Literal['GMM','k-means']] = None, 
            clustering_classes: Optional[int | np.ndarray] = None, 
            clustering_prop: Optional[float] = None, 
            polyhedral_method: Literal['biased_SVM','unbiased_SVM','auto'] = 'auto', 
            polyhedral_prop: Optional[float] = None, 
            pure_distance_method:Literal['minIn','maxOut','maxOutPeak'] = 'maxOut'
    ):
        """
        Main function for **polyhedral unmixing** fitting.

        Given a hyperspectral image, the algorithm determines both the endmembers and abundances 
        from a given classification map (i.e., semantic segmentation of the input spectral image) 
        using a polyhedral(-cone) partitioning of the spectral space via a (unbiased) linear SVM.

        Parameters
        ----------
        * data : ndarray
            Hyperspectral image Y to unmix of shape (n_pixels, n_bands).
        * labels : ndarray, optional, default: None
            Class labels of the input data, of shape (n_pixels,). 
        * clustering_method : ModelProtocol or Callable or {'GMM','k-means'}, optional, default: None
            Method for data clustering or classification 
            (if labels are not directly given). 
        * clustering_classes : int (> 0) or ndarray, optional, default: None
            If int, number of classes to consider for the segmentation. 
            Otherwise, list of n class spectra or coordinates.
        * clustering_prop : float, in (0,1], optional, default: None
            Proportion of the sample to extract from input image 
            on which is fit the chosen segmentation method.
        * polyhedral_method : {'biased_SVM','unbiased_SVM','auto'}, default: 'auto'
            Method used to construct the polyhedral partitioning of the space. 
            SVM can be either biased or unbiased.
        * polyhedral_prop : float, in [-1,1]\\\{0}, optional, default: None
            Proportion of the input sample on which is built the partitioning. 
            If negative, prioritize the sample used for clustering.
        * pure_distance_method : {'minIn','maxOut','maxOutPeak'}, default: 'maxOut'
            Method for computing pure / extremal distance vectors for change of basis. 
            Either extract or compute new vectors.

        Returns
        -------
        self: PolyhedralUnmixingModel.
        """
        
        ### Check: input data
        if type(data) is not np.ndarray:
            if np.iterable(data):
                try: data = np.asarray(data)
                except: raise ValueError("Input data cannot be converted into a ndarray")
            else: raise ValueError("Input data must be iterable")
        dshape = data.shape
        if data.ndim > 2:
            data = data.reshape(int(np.prod(data.shape[:-1])), data.shape[-1])
        if self.verbose:
            print(f"Spectral data - size: {dshape[:-1]}; bands: {data.shape[-1]}")
        
        ### Check: input labels
        if labels is None and clustering_method is None:
            raise ValueError("Either 'labels' or 'clustering method' must be given")
        if labels is not None:
            if type(labels) is not np.ndarray:
                if np.iterable(labels):
                    try: labels = np.asarray(labels)
                    except: raise ValueError("Input labels cannot be converted into a ndarray")
                else: raise ValueError("Input labels must be iterable")
            try: labels = labels.reshape(*data.shape[:-1])
            except: raise ValueError("Input labels must be of shape data.shape[:-1] = (n_pixels,)")
            if clustering_method is not None or clustering_classes is not None or clustering_prop is not None:
                warnings.warn("Labels are already given: given clustering parameters ignored")
        
        # A -> B: Pre-processing: dimensionality reduction via PCA + luminance normalization
        if self.verbose: print("* [1/7] (A -> B) Pre-processing input data...", end=' ')
        new_data = self.preprocess(
            data = data, 
            normalize = self.normalize, 
            PCA_ndim = self.PCA_ndim, 
            additional = self.additional_preprocessing, 
            mode = 'fit')
        if self.verbose: print("Done!")

        # B -> C: Semantic segmentation: classify the pixels into m classes
        if labels is None:
            if self.verbose: print("* [2/7] (B -> C) Clustering of input data...", end=' ')
            labels = self.compute_labels(
                data = new_data, 
                method = clustering_method, 
                classes = clustering_classes, 
                sample_prop = clustering_prop, 
                mode = 'fit')
            if self.verbose: print("Done!")
        else:
            if self.verbose: print("* [2/7] (B -> C) Pixel labels given as input!", end='\n')
            self.clustering_method = '__given__'
        self.labels = labels

        # C -> D: Polyhedral(-cone) partitioning of the spectral space using (unbiased) SVM
        if self.verbose: print("* [3/7] (C -> D) Polyhedral" + "-cone"  * ('unbiased' in polyhedral_method.lower()) + " partitioning...", end=' ')
        sample_arg_priority = self.arg_clustered_pixels if polyhedral_prop < 0 else None
        polyhedra = fit_polyhedral_partition(
            data = new_data, 
            labels = labels, 
            method = polyhedral_method, 
            sample_prop = np.abs(polyhedral_prop), 
            sample_arg_priority = sample_arg_priority)
        if self.verbose: print("Done!")
        self.polyhedra = polyhedra
        self.n_endmembers = len(polyhedra)

        # D -> E: Computation of the signed distances to the convex polyhedral sets
        if self.verbose: print("* [4/7] (D -> E) Signed distances computation (" + ("Python" if self.only_python else "C") + ")...", end=' ')
        distances = signed_distances_to_polyhedra(
            data = new_data, 
            h = polyhedra, 
            python = self.only_python, 
            infos = False)
        if self.verbose: print("Done!")
        
        # E -> F: Change of basis in distance space
        if self.verbose: print("* [5/7] (E -> F) Change of basis in distance space...", end=' ')
        pure_distances = main_extrema(distances, method=pure_distance_method)
        new_distances = change_of_basis(
            data = distances, 
            new_basis = pure_distances)
        if self.verbose: print("Done!")
        self.pure_distances = pure_distances

        # F -> G: Projection of new distance vectors onto the probaility simplex
        if self.verbose: print("* [6/7] (F -> G) Projection onto the probaility simplex...", end=' ')
        saturation = self.saturation / np.std(new_distances)
        A_estimate = to_probability(
            x = new_distances, 
            saturation = saturation, 
            python = self.only_python)
        if self.verbose: print("Done!")

        # FINAL: Endmember and abundance recovery
        if self.verbose: print("* [7/7] (FINAL!) Deducing endmembers...", end=' ')
        observations = self.preprocess(
            data = data, 
            normalize = self.keep_normalized, 
            PCA_ndim = None, 
            mode = 'indep')
        endmembers = estimate_endmembers_ridge(
            Y = observations, 
            A = A_estimate
        )
        if self.verbose: print("Done!")
        self.endmembers = endmembers

        return self

    def predict_raw(
            self, 
            data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function for polyhedral unmixing **raw** prediction.

        Predict both endmembers and abundances by passing the data through the whole polyhedral processing chain.

        Parameters
        ----------
        data : ndarray
            Hyperspectral image Y to unmix of shape (n_pixels, n_bands).

        Returns
        -------
        endmembers : ndarray, M
            Spectral signatures associated with each of the n classes.
        abundances : ndarray, A
            Abundance maps representing endmember fractions in the pixels.
        """

        ### Check: input data
        if type(data) is not np.ndarray:
            if np.iterable(data):
                try: data = np.asarray(data)
                except: raise ValueError("Input data cannot be converted into a ndarray")
            else: raise ValueError("Input data must be iterable")
        dshape = data.shape
        if data.ndim > 2:
            data = data.reshape(int(np.prod(data.shape[:-1])), data.shape[-1])
        if self.verbose:
            print(f"Spectral data - size: {dshape[:-1]}; bands: {data.shape[-1]}")
        
        ### Check whether model parameters have been fitted
        if self.polyhedra is None or self.pure_distances is None:
            raise Exception("Model parameters have not been fitted yet! Please use either .fit_init() or .fit() to fit them")

        # A -> B: Pre-processing: dimensionality reduction via PCA + luminance normalization
        if self.verbose: print("* [1/5] (A -> B) Pre-processing input data...", end=' ')
        new_data = self.preprocess(data = data, mode = 'predict')
        if self.verbose: print("Done!")

        # D -> E: Computation of the signed distances to the convex polyhedral sets
        if self.verbose: print("* [2/5] (D -> E) Signed distances computation (" + ("Python" if self.only_python else "C") + ")...", end=' ')
        distances = signed_distances_to_polyhedra(
            data = new_data, 
            h = self.polyhedra, 
            python = self.only_python, 
            infos = False)
        if self.verbose: print("Done!")
        
        # E -> F: Change of basis in distance space
        if self.verbose: print("* [3/5] (E -> F) Change of basis in distance space...", end=' ')
        new_distances = change_of_basis(
            data = distances, 
            new_basis = self.pure_distances)
        if self.verbose: print("Done!")

        # F -> G: Projection of new distance vectors onto the probaility simplex
        if self.verbose: print("* [4/5] (F -> G) Projection onto the probaility simplex...", end=' ')
        saturation = self.saturation / np.std(new_distances)
        A_estimate = to_probability(
            x = new_distances, 
            saturation = saturation, 
            python = self.only_python)
        if self.verbose: print("Done!")

        # FINAL: Endmember and abundance recovery
        if self.verbose: print("* [5/5] (FINAL!) Deducing endmembers and abundances...", end=' ')
        observations = self.preprocess(
            data = data, 
            normalize = self.keep_normalized, 
            PCA_ndim = None, 
            mode = 'indep')
        endmembers = estimate_endmembers_ridge(
            Y = observations, 
            A = A_estimate
        )
        abundances = estimate_abundances_ridge(
            Y = observations, 
            M = endmembers
        )
        if self.verbose: print("Done!")

        abundances[abundances<=0] = 0
        abundances = abundances / abundances.sum(axis=-1, keepdims=True)

        return endmembers, abundances.reshape(*dshape[:-1], abundances.shape[-1])

    def predict(
            self, 
            data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Function for **polyhedral unmixing** prediction. 

        Predict abundances using Ridge matrix pseudo-inversion over the endmembers fitted during model fitting.

        Parameters
        ----------
        data : ndarray
            Hyperspectral image Y to unmix of shape (n_pixels, n_bands).

        Returns
        -------
        endmembers : ndarray, M
            Spectral signatures associated with each of the n classes.
        abundances : ndarray, A
            Abundance maps representing endmember fractions in the pixels.
        """

        ### Check: input data
        if type(data) is not np.ndarray:
            if np.iterable(data):
                try: data = np.asarray(data)
                except: raise ValueError("Input data cannot be converted into a ndarray")
            else: raise ValueError("Input data must be iterable")
        dshape = data.shape
        if data.ndim > 2:
            data = data.reshape(int(np.prod(data.shape[:-1])), data.shape[-1])
        if self.verbose:
            print(f"Spectral data - size: {dshape[:-1]}; bands: {data.shape[-1]}")

        ### Check whether endmembers have been fitted
        if self.endmembers is None:
            raise Exception("Endmembers have not been fitted yet! Please use .fit() to fit them")

        # FINAL: Endmember and abundance recovery
        if self.verbose: print("Recovering endmembers and deducing abundances...", end=' ')
        observations = self.preprocess(
            data = data, 
            normalize = self.keep_normalized, 
            PCA_ndim = None, 
            mode = 'indep')
        abundances = estimate_abundances_ridge(
            Y = observations, 
            M = self.endmembers
        )
        if self.verbose: print("Done!")

        abundances[abundances<=0] = 0
        abundances = abundances / abundances.sum(axis=-1, keepdims=True)

        return self.endmembers, abundances.reshape(*dshape[:-1], abundances.shape[-1])

    def fit_predict(
            self, 
            data: np.ndarray, 
            labels: Optional[np.ndarray] = None, 
            clustering_method: Optional[ModelProtocol | Callable | Literal['GMM','k-means']] = None, 
            clustering_classes: Optional[int | np.ndarray] = None, 
            clustering_prop: Optional[float] = None, 
            polyhedral_method: Literal['biased_SVM','unbiased_SVM','auto'] = 'auto', 
            polyhedral_prop: Optional[float] = None, 
            pure_distance_method:Literal['minIn','maxOut','maxOutPeak'] = 'maxOut'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Main function for **polyhedral unmixing** fitting and prediction.

        Given a hyperspectral image, the algorithm determines both the endmembers and abundances 
        from a given classification map (i.e., semantic segmentation of the input spectral image) 
        using a polyhedral(-cone) partitioning of the spectral space via a (unbiased) linear SVM.

        Parameters
        ----------
        * data : ndarray
            Hyperspectral image Y to unmix of shape (n_pixels, n_bands).
        * labels : ndarray, optional, default: None
            Class labels of the input data, of shape (n_pixels,). 
        * clustering_method : ModelProtocol or Callable or {'GMM','k-means'}, optional, default: None
            Method for data clustering or classification 
            (if labels are not directly given). 
        * clustering_classes : int (> 0) or ndarray, optional, default: None
            If int, number of classes to consider for the segmentation. 
            Otherwise, list of n class spectra or coordinates.
        * clustering_prop : float, in (0,1], optional, default: None
            Proportion of the sample to extract from input image 
            on which is fit the chosen segmentation method.
        * polyhedral_method : {'biased_SVM','unbiased_SVM','auto'}, default: 'auto'
            Method used to construct the polyhedral partitioning of the space. 
            SVM can be either biased or unbiased.
        * polyhedral_prop : float, in (0,1], optional, default: None
            Proportion of the sample to extract from input image 
            on which is built the polyhedral partitioning.
        * pure_distance_method : {'minIn','maxOut','maxOutPeak'}, default: 'maxOut'
            Method for computing pure / extremal distance vectors for change of basis. 
            Either extract or compute new vectors.

        Returns
        -------
        endmembers : ndarray, M
            Spectral signatures associated with each of the n classes.
        abundances : ndarray, A
            Abundance maps representing endmember fractions in the pixels.
        """

        # 1. fit model
        self.fit(
            data = data, 
            labels = labels, 
            clustering_method = clustering_method, 
            clustering_classes = clustering_classes, 
            clustering_prop = clustering_prop, 
            polyhedral_method = polyhedral_method, 
            polyhedral_prop = polyhedral_prop, 
            pure_distance_method = pure_distance_method
        )

        # 2. predict
        verbose = self.verbose
        self.verbose = False
        endmembers, abundances = self.predict(data = data)
        self.verbose = verbose

        return endmembers, abundances

    def get_endmembers(self) -> np.ndarray:
        """
        Function to return endmembers fitted during model fitting.

        Returns
        -------
        endmembers : ndarray, M
            Spectral signatures associated with each of the n classes.
        """
        return self.endmembers
    
    def predict_abundances(self, data: np.ndarray) -> np.ndarray:
        """
        Function for abundance prediction. 

        Predict abundances using Ridge matrix pseudo-inversion over the endmembers fitted during model fitting.

        Parameters
        ----------
        data : ndarray
            Hyperspectral image Y to unmix of shape (n_pixels, n_bands).

        Returns
        -------
        abundances : ndarray, A
            Abundance maps representing endmember fractions in the pixels.
        """
        return self.predict(data)[1]

    def fit_initial(
            self, 
            data: np.ndarray, 
            labels: Optional[np.ndarray] = None, 
            clustering_method: Optional[ModelProtocol | Callable | Literal['GMM','k-means']] = None, 
            clustering_classes: Optional[int | np.ndarray] = None, 
            clustering_prop: Optional[float] = None, 
            polyhedral_method: Literal['biased_SVM','unbiased_SVM','auto'] = 'auto', 
            polyhedral_prop: Optional[float] = None, 
            pure_distance_method:Literal['minIn','maxOut','maxOutPeak'] = 'maxOut'
    ):
        """
        Function for **initial** fitting.

        Given a hyperspectral image, the algorithm determines an initial abundance estimate 
        from a given classification map (i.e., semantic segmentation of the input spectral image) 
        using a polyhedral(-cone) partitioning of the spectral space via a (unbiased) linear SVM.

        No endmbember estimation nor Ridge matrix pseudo-inversion is performed.

        Parameters
        ----------
        * data : ndarray
            Hyperspectral image Y to unmix of shape (n_pixels, n_bands).
        * labels : ndarray, optional, default: None
            Class labels of the input data, of shape (n_pixels,). 
        * clustering_method : ModelProtocol or Callable or {'GMM','k-means'}, optional, default: None
            Method for data clustering or classification 
            (if labels are not directly given). 
        * clustering_classes : int (> 0) or ndarray, optional, default: None
            If int, number of classes to consider for the segmentation. 
            Otherwise, list of n class spectra or coordinates.
        * clustering_prop : float, in (0,1], optional, default: None
            Proportion of the sample to extract from input image 
            on which is fit the chosen segmentation method.
        * polyhedral_method : {'biased_SVM','unbiased_SVM','auto'}, default: 'auto'
            Method used to construct the polyhedral partitioning of the space. 
            SVM can be either biased or unbiased.
        * polyhedral_prop : float, in [-1,1]\\\{0}, optional, default: None
            Proportion of the input sample on which is built the partitioning. 
            If negative, prioritize the sample used for clustering.
        * pure_distance_method : {'minIn','maxOut','maxOutPeak'}, default: 'maxOut'
            Method for computing pure / extremal distance vectors for change of basis. 
            Either extract or compute new vectors.

        Returns
        -------
        self: PolyhedralUnmixingModel.
        """
        
        ### Check: input data
        if type(data) is not np.ndarray:
            if np.iterable(data):
                try: data = np.asarray(data)
                except: raise ValueError("Input data cannot be converted into a ndarray")
            else: raise ValueError("Input data must be iterable")
        dshape = data.shape
        if data.ndim > 2:
            data = data.reshape(int(np.prod(data.shape[:-1])), data.shape[-1])
        if self.verbose:
            print(f"Spectral data - size: {dshape[:-1]}; bands: {data.shape[-1]}")
        
        ### Check: input labels
        if labels is None and clustering_method is None:
            raise ValueError("Either 'labels' or 'clustering method' must be given")
        if labels is not None:
            if type(labels) is not np.ndarray:
                if np.iterable(labels):
                    try: labels = np.asarray(labels)
                    except: raise ValueError("Input labels cannot be converted into a ndarray")
                else: raise ValueError("Input labels must be iterable")
            try: labels = labels.reshape(*data.shape[:-1])
            except: raise ValueError("Input labels must be of shape data.shape[:-1] = (n_pixels,)")
            if clustering_method is not None or clustering_classes is not None or clustering_prop is not None:
                warnings.warn("Labels are already given: given clustering parameters ignored")
        
        # A -> B: Pre-processing: dimensionality reduction via PCA + luminance normalization
        if self.verbose: print("* [1/5] (A -> B) Pre-processing input data...", end=' ')
        new_data = self.preprocess(
            data = data, 
            normalize = self.normalize, 
            PCA_ndim = self.PCA_ndim, 
            additional = self.additional_preprocessing, 
            mode = 'fit')
        if self.verbose: print("Done!")

        # B -> C: Semantic segmentation: classify the pixels into m classes
        if labels is None:
            if self.verbose: print("* [2/5] (B -> C) Clustering of input data...", end=' ')
            labels = self.compute_labels(
                data = new_data, 
                method = clustering_method, 
                classes = clustering_classes, 
                sample_prop = clustering_prop, 
                mode = 'fit')
            if self.verbose: print("Done!")
        else:
            if self.verbose: print("* [2/5] (B -> C) Pixel labels given as input!", end='\n')
            self.clustering_method = '__given__'
        self.labels = labels

        # C -> D: Polyhedral(-cone) partitioning of the spectral space using (unbiased) SVM
        if self.verbose: print("* [3/5] (C -> D) Polyhedral" + "-cone"  * ('unbiased' in polyhedral_method.lower()) + " partitioning...", end=' ')
        sample_arg_priority = self.arg_clustered_pixels if polyhedral_prop < 0 else None
        polyhedra = fit_polyhedral_partition(
            data = new_data, 
            labels = labels, 
            method = polyhedral_method, 
            sample_prop = np.abs(polyhedral_prop), 
            sample_arg_priority = sample_arg_priority)
        if self.verbose: print("Done!")
        self.polyhedra = polyhedra
        self.n_endmembers = len(polyhedra)

        # D -> E: Computation of the signed distances to the convex polyhedral sets
        if self.verbose: print("* [4/5] (D -> E) Signed distances computation (" + ("Python" if self.only_python else "C") + ")...", end=' ')
        distances = signed_distances_to_polyhedra(
            data = new_data, 
            h = polyhedra, 
            python = self.only_python, 
            infos = False)
        if self.verbose: print("Done!")
        
        # E -> F: Change of basis in distance space
        if self.verbose: print("* [5/5] (E -> F) Pure / extremal distances extraction...", end=' ')
        pure_distances = main_extrema(distances, method=pure_distance_method)
        if self.verbose: print("Done!")
        self.pure_distances = pure_distances

        return self

    def fit_predict_initial_abundances(
            self, 
            data: np.ndarray, 
            labels: Optional[np.ndarray] = None, 
            clustering_method: Optional[ModelProtocol | Callable | Literal['GMM','k-means']] = None, 
            clustering_classes: Optional[int | np.ndarray] = None, 
            clustering_prop: Optional[float] = None, 
            polyhedral_method: Literal['biased_SVM','unbiased_SVM','auto'] = 'auto', 
            polyhedral_prop: Optional[float] = None, 
            pure_distance_method:Literal['minIn','maxOut','maxOutPeak'] = 'maxOut'
    ) -> np.ndarray:
        """
        Function for **initial** fitting and **initial abundance** prediction.

        Given a hyperspectral image, the algorithm determines an initial abundance estimate 
        from a given classification map (i.e., semantic segmentation of the input spectral image) 
        using a polyhedral(-cone) partitioning of the spectral space via a (unbiased) linear SVM.

        No endmbember estimation nor Ridge matrix pseudo-inversion is performed.

        Parameters
        ----------
        * data : ndarray
            Hyperspectral image Y to unmix of shape (n_pixels, n_bands).
        * labels : ndarray, optional, default: None
            Class labels of the input data, of shape (n_pixels,). 
        * clustering_method : ModelProtocol or Callable or {'GMM','k-means'}, optional, default: None
            Method for data clustering or classification 
            (if labels are not directly given). 
        * clustering_classes : int (> 0) or ndarray, optional, default: None
            If int, number of classes to consider for the segmentation. 
            Otherwise, list of n class spectra or coordinates.
        * clustering_prop : float, in (0,1], optional, default: None
            Proportion of the sample to extract from input image 
            on which is fit the chosen segmentation method.
        * polyhedral_method : {'biased_SVM','unbiased_SVM','auto'}, default: 'auto'
            Method used to construct the polyhedral partitioning of the space. 
            SVM can be either biased or unbiased.
        * polyhedral_prop : float, in [-1,1]\\\{0}, optional, default: None
            Proportion of the input sample on which is built the partitioning. 
            If negative, prioritize the sample used for clustering.
        * pure_distance_method : {'minIn','maxOut','maxOutPeak'}, default: 'maxOut'
            Method for computing pure / extremal distance vectors for change of basis. 
            Either extract or compute new vectors.

        Returns
        -------
        init_abundances : ndarray, A'
            Abundance maps representing endmember fractions in the pixels.
        """
        
        ### Check: input data
        if type(data) is not np.ndarray:
            if np.iterable(data):
                try: data = np.asarray(data)
                except: raise ValueError("Input data cannot be converted into a ndarray")
            else: raise ValueError("Input data must be iterable")
        dshape = data.shape
        if data.ndim > 2:
            data = data.reshape(int(np.prod(data.shape[:-1])), data.shape[-1])
        if self.verbose:
            print(f"Spectral data - size: {dshape[:-1]}; bands: {data.shape[-1]}")
        
        ### Check: input labels
        if labels is None and clustering_method is None:
            raise ValueError("Either 'labels' or 'clustering method' must be given")
        if labels is not None:
            if type(labels) is not np.ndarray:
                if np.iterable(labels):
                    try: labels = np.asarray(labels)
                    except: raise ValueError("Input labels cannot be converted into a ndarray")
                else: raise ValueError("Input labels must be iterable")
            try: labels = labels.reshape(*data.shape[:-1])
            except: raise ValueError("Input labels must be of shape data.shape[:-1] = (n_pixels,)")
            if clustering_method is not None or clustering_classes is not None or clustering_prop is not None:
                warnings.warn("Labels are already given: given clustering parameters ignored")
        
        # A -> B: Pre-processing: dimensionality reduction via PCA + luminance normalization
        if self.verbose: print("* [1/6] (A -> B) Pre-processing input data...", end=' ')
        new_data = self.preprocess(
            data = data, 
            normalize = self.normalize, 
            PCA_ndim = self.PCA_ndim, 
            additional = self.additional_preprocessing, 
            mode = 'fit')
        if self.verbose: print("Done!")

        # B -> C: Semantic segmentation: classify the pixels into m classes
        if labels is None:
            if self.verbose: print("* [2/6] (B -> C) Clustering of input data...", end=' ')
            labels = self.compute_labels(
                data = new_data, 
                method = clustering_method, 
                classes = clustering_classes, 
                sample_prop = clustering_prop, 
                mode = 'fit')
            if self.verbose: print("Done!")
        else:
            if self.verbose: print("* [2/6] (B -> C) Pixel labels given as input!", end='\n')
            self.clustering_method = '__given__'
        self.labels = labels

        # C -> D: Polyhedral(-cone) partitioning of the spectral space using (unbiased) SVM
        if self.verbose: print("* [3/6] (C -> D) Polyhedral" + "-cone"  * ('unbiased' in polyhedral_method.lower()) + " partitioning...", end=' ')
        sample_arg_priority = self.arg_clustered_pixels if polyhedral_prop < 0 else None
        polyhedra = fit_polyhedral_partition(
            data = new_data, 
            labels = labels, 
            method = polyhedral_method, 
            sample_prop = np.abs(polyhedral_prop), 
            sample_arg_priority = sample_arg_priority)
        if self.verbose: print("Done!")
        self.polyhedra = polyhedra
        self.n_endmembers = len(polyhedra)

        # D -> E: Computation of the signed distances to the convex polyhedral sets
        if self.verbose: print("* [4/6] (D -> E) Signed distances computation (" + ("Python" if self.only_python else "C") + ")...", end=' ')
        distances = signed_distances_to_polyhedra(
            data = new_data, 
            h = polyhedra, 
            python = self.only_python, 
            infos = False)
        if self.verbose: print("Done!")
        
        # E -> F: Change of basis in distance space
        if self.verbose: print("* [5/6] (E -> F) Change of basis in distance space...", end=' ')
        pure_distances = main_extrema(distances, method=pure_distance_method)
        new_distances = change_of_basis(
            data = distances, 
            new_basis = pure_distances)
        if self.verbose: print("Done!")
        self.pure_distances = pure_distances

        # F -> G: Projection of new distance vectors onto the probaility simplex
        if self.verbose: print("* [6/6] (F -> G) Projection onto the probaility simplex...", end=' ')
        saturation = self.saturation / np.std(new_distances)
        A_estimate = to_probability(
            x = new_distances, 
            saturation = saturation, 
            python = self.only_python)
        if self.verbose: print("Done!")

        return A_estimate.reshape(*dshape[:-1], A_estimate.shape[-1])

    def predict_initial_abundances(
            self, 
            data: np.ndarray
    ) -> np.ndarray:
        """
        Function for **initial abundance** prediction.

        Predict the initial abundance estimate by passing the data through the whole polyhedral 
        processing chain without endmember estimation nor Ridge matrix pseudo-inversion.

        Parameters
        ----------
        data : ndarray
            Hyperspectral image Y to unmix of shape (n_pixels, n_bands).

        Returns
        -------
        init_abundances : ndarray, A'
            Abundance maps representing endmember fractions in the pixels.
        """

        ### Check: input data
        if type(data) is not np.ndarray:
            if np.iterable(data):
                try: data = np.asarray(data)
                except: raise ValueError("Input data cannot be converted into a ndarray")
            else: raise ValueError("Input data must be iterable")
        dshape = data.shape
        if data.ndim > 2:
            data = data.reshape(int(np.prod(data.shape[:-1])), data.shape[-1])
        if self.verbose:
            print(f"Spectral data - size: {dshape[:-1]}; bands: {data.shape[-1]}")

        ### Check whether model parameters have been fitted
        if self.polyhedra is None or self.pure_distances is None:
            raise Exception("Model parameters have not been fitted yet! Please use either .fit_init() or .fit() to fit them")

        # A -> B: Pre-processing: dimensionality reduction via PCA + luminance normalization
        if self.verbose: print("* [1/4] (A -> B) Pre-processing input data...", end=' ')
        new_data = self.preprocess(data = data, mode = 'predict')
        if self.verbose: print("Done!")

        # D -> E: Computation of the signed distances to the convex polyhedral sets
        if self.verbose: print("* [2/4] (D -> E) Signed distances computation (" + ("Python" if self.only_python else "C") + ")...", end=' ')
        distances = signed_distances_to_polyhedra(
            data = new_data, 
            h = self.polyhedra, 
            python = self.only_python, 
            infos = False)
        if self.verbose: print("Done!")
        
        # E -> F: Change of basis in distance space
        if self.verbose: print("* [3/4] (E -> F) Change of basis in distance space...", end=' ')
        new_distances = change_of_basis(
            data = distances, 
            new_basis = self.pure_distances)
        if self.verbose: print("Done!")

        # F -> G: Projection of new distance vectors onto the probaility simplex
        if self.verbose: print("* [4/4] (F -> G) Projection onto the probaility simplex...", end=' ')
        saturation = self.saturation / np.std(new_distances)
        A_estimate = to_probability(
            x = new_distances, 
            saturation = saturation, 
            python = self.only_python)
        if self.verbose: print("Done!")

        return A_estimate.reshape(*dshape[:-1], A_estimate.shape[-1])

