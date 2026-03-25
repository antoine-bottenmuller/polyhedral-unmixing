import warnings
import itertools
import numpy as np

from typing import Tuple, Optional, Literal

from scipy.optimize import linprog


#%%
# Basic linear algebra functions
###

def scalar(a:np.ndarray, b:np.ndarray, keepdims:bool=False) -> np.ndarray:
    if keepdims:
        return np.einsum('...i,...i->...', a, b)[..., np.newaxis]
    return np.einsum('...i,...i->...', a, b)

def norm(v:np.ndarray, keepdims:bool=False) -> np.ndarray:
    return np.linalg.norm(v, axis=-1, keepdims=keepdims)

def normed(v:np.ndarray) -> np.ndarray:
    res = np.finfo(v.dtype).resolution if np.issubdtype(v.dtype, np.floating) else 1
    v_norm = norm(v, keepdims=True)
    v_zero = v_norm < res
    return v * ~v_zero / (v_norm * ~v_zero + v_zero)


#%%
# Function to check whether data points are (individually) inside a given polyhedral set
###

def in_polyhedron(data:np.ndarray, polyhedron:np.ndarray) -> np.ndarray:
    """
    Check whether data are inside a polyhedron defined as an intersection of finitely-many half-spaces. 
    Return boolean array of shape data.shape[:-1] where True means inside, False outside.
    """
    eps = np.finfo(data.dtype).resolution if np.issubdtype(data.dtype, np.floating) else np.finfo(np.floating).resolution
    inside = np.max(scalar(data[np.newaxis] - polyhedron[:,0,np.newaxis], polyhedron[:,1,np.newaxis]), axis=0) < eps
    return inside


#%%
# Function to distribute n=(m-1)*m/2 separation hyperplanes into (m, m-1) halfspaces to form m classes' polyhedral sets
###

def distribute_halfspaces(h:np.ndarray, means:np.ndarray) -> np.ndarray:
    """
    Distribute class-separating hyperplanes to form classes' polyhedral sets.
    * h: (n_hyperplanes:=(n_classes-1)*n_classes/2, 2, ndim)
    * means: (n_classes, ndim)\n
    Return h: (n_classes, n_classes-1, 2, ndim)
    """
    n_classes = means.shape[0]
    if len(h) != int(np.round(n_classes*(n_classes-1)/2)):
        message = "Means (class references) and hyperplanes do not fit!"
        message+=" There must be n_classes*(n_classes-1)/2 hyperplanes."
        warnings.warn(message)
    pair_id = list(itertools.combinations(np.arange(n_classes), 2))
    new_h = np.zeros(shape=(n_classes,n_classes-1)+h.shape[1:], dtype=h.dtype)
    for k in range(h.shape[0]):
        i, j = pair_id[k]
        c, v = h[k]
        v_ij = means[j] - means[i]
        v_i = v * np.sign(scalar(v,+v_ij))
        v_j = v * np.sign(scalar(v,-v_ij))
        h_i = np.asarray((c, v_i))
        h_j = np.asarray((c, v_j))
        new_h[i,j-1] = h_i
        new_h[j,i] = h_j
    return new_h


#%%
# Functions to convert parameters (a,b) in <a,x> + b >= 0, to parameters (c,v), v unit, in <x-c,v> <= 0 ; and reciprocally!
###

def max_indicator_array(a:np.ndarray) -> np.ndarray:
    """
    Return boolean array of same shape as a, indicating where the (unique) maximum values along the last axis are located.
    """
    max_indices = np.argmax(a, axis=-1)
    b = np.zeros(shape=a.shape, dtype=bool)
    grid = np.ogrid[tuple(slice(dim) for dim in a.shape[:-1])]
    b[(*grid, max_indices)] = True
    return b

# from inequalities to half-space ordered pairs
def to_half_space_pairs(w:np.ndarray, b:np.ndarray) -> np.ndarray:
    """
    Finds a (non-unique) ordered pair of vectors (c,v), with v unite vector, 
    such that the inequality <x-c,v> <= 0 is equivalent to <w,x>+b >= 0.
    These are two ways of expressing an half-space inequality.
    We have v=-w. c exists and is unique iif w!=0 ; 
    c exists and can be any vector iif w==0 and b>=0 (c=0 by default) ; 
    c does not exist iif w==0 and b<0 (c,v are then set to np.nan).\n
    Input:
    * w: vector or matrix of shape (..., ndim) ;
    * b: scalar or vector of shape (...,) or (..., 1).\n
    Return: array h of ordered pairs (c,v), with shape (..., 2, ndim).
    """
    if type(w) is not np.ndarray:
        try:
            w = np.asarray(w)
        except:
            raise ValueError("Parameter 'w' must be an array-like.")
    if type(b) is not np.ndarray:
        try:
            b = np.asarray(b)
        except:
            raise ValueError("Parameter 'b' must be an array-like or a scalar.")
    if b.ndim == w.ndim-1:
        b = np.expand_dims(b, axis=-1)
    elif b.ndim != w.ndim:
        raise ValueError("Array 'b' must be of dimension w.ndim-1 or w.ndim.")
    
    sign_cv_ineq = - 1 # '-1' because inequality <x-c,v> <= 0 is negative inequality
    sign_wb_ineq = + 1 # '+1' because inequality <w,x>+b >= 0 is positive inequality
    inverted = sign_cv_ineq * sign_wb_ineq # '-1' if these two inequalities have opposite sign!

    w_max = w * max_indicator_array(np.abs(w))
    w_max_zero = w_max == 0

    w_norm = norm(w, keepdims=True)
    w_norm_zero = w_norm == 0

    c_all_b = inverted * np.ones_like(w) * b
    c = c_all_b * ~w_max_zero / (w_max + w_max_zero)

    v = inverted * w / (w_norm + w_norm_zero)

    # Cases where w is the zero vector (i.e. if max(|w_i|)==0 or ||w||==0)
    w_zero = np.prod(w_max_zero, axis=-1, dtype=bool, keepdims=True) + w_norm_zero # max(|w_i|)==0 or ||w||==0
    opposite_b = sign_wb_ineq * b < 0 # 0*x+b >= 0 has no solution iif b < 0 (inverted if inequality is negative)
    ispossible = (w_zero *~opposite_b)[...,0] # consider cases where the wb inequality is possible!
    impossible = (w_zero * opposite_b)[...,0] # consider cases where the wb inequality is impossible!
    c[ispossible], v[ispossible] = 0, 0
    c[impossible], v[impossible] = np.nan, np.nan

    h = np.asarray([c,v], dtype=v.dtype)
    h = np.transpose(h, axes=tuple(np.arange(1,v.ndim))+(0,)+(v.ndim,))

    return h

# from half-space ordered pairs to inequalities
def to_half_space_inequality(h:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds a (non-unique) vector/matrix w and scalar/vector b, 
    such that the inequality <w,x>+b >= 0 is equivalent to <x-c,v> <= 0.
    These are two ways of expressing an half-space inequality.
    We have w=-v and b=<v,c> (both are always well defined).\n
    Input: array h of ordered pairs of vectors (c,v), with shape (..., 2, ndim).\n
    Return:
    * w: vector or matrix of shape (..., ndim) ;
    * b: scalar or vector of shape (...,).
    """
    if type(h) is not np.ndarray:
        try:
            h = np.asarray(h)
        except:
            raise ValueError("Parameter 'h' must be an array-like.")
    
    sign_cv_ineq = - 1 # '-1' because inequality <x-c,v> <= 0 is negative inequality
    sign_wb_ineq = + 1 # '+1' because inequality <w,x>+b >= 0 is positive inequality
    inverted = sign_cv_ineq * sign_wb_ineq # '-1' if these two inequalities have opposite sign!

    h = np.transpose(h, axes=(-2,)+tuple(np.arange(h.ndim-2))+(-1,))
    c = h[0] # shape: (..., ndim)
    v = h[1] # shape: (..., ndim)

    w = inverted * v
    b = - inverted * np.sum(v * c, axis=-1)

    return w, b


#%%
# Function to compute the I-rank of a matrix. 
# Ref: Lloyd L. Dines, "Systems of Linear Inequalities", Annals of Mathematics, Vol. 20, No. 3, March 1919, pp. 191-199
###

def mres(array:np.ndarray, upmul:int=0) -> float:
    dtype = array.dtype
    if np.issubdtype(dtype, np.floating):
        finfo = np.finfo(dtype)
        decim = int(finfo.precision - upmul)
        nfres = finfo.resolution * np.power(10,upmul)
        return np.round(nfres, decimals=decim)
    return dtype.type(1)

def I_minor(M:np.ndarray, r:int) -> np.ndarray:
    pos = M[:,r] > 0
    neg = M[:,r] < 0
    zer = M[:,r] ==0
    P,N = pos.sum(), neg.sum()
    new_M = np.delete(M,r,axis=1)
    pairs = np.asarray(np.meshgrid(M[pos,r], M[neg,r])).T.reshape(P*N,2)
    mat = np.asarray([np.tile(new_M[pos],(N,1,1)), np.tile(new_M[neg],(P,1,1)).transpose(1,0,2)]).T.reshape(new_M.shape[1],P*N,2)
    prod_pairs_mat = (pairs * mat[:,:,::-1]).T
    I_complement_PN = prod_pairs_mat[0] - prod_pairs_mat[1]
    I_complement_Z  = new_M[zer]
    I_complement = np.concatenate((I_complement_PN, I_complement_Z), axis=0)
    return I_complement

def I_rank(M:np.ndarray, res:Optional[float]=None) -> int:
    if res is None:
        res = mres(M, 1 + max(0, int(np.log10(np.max(np.abs(M)))))) # error committed (>0)
    else:
        res = np.abs(res)
    m = M.shape[0]
    M = M*(np.abs(M)>res)
    sgn = np.sign(M)
    ssm = sgn.sum(0)
    tpe = (ssm==m)+(ssm==-m)
    if tpe.sum()>0:
        return M.shape[1]
    matrices = [M]
    level = M.shape[1]-1
    while level>0:
        I_minors = []
        for M in matrices:
            for r in range(level+1):
                I_complement = I_minor(M,r)
                m = I_complement.shape[0]
                I_complement = I_complement*(np.abs(I_complement)>res)
                sgn = np.sign(I_complement)
                ssm = sgn.sum(0)
                tpe = (ssm==m)+(ssm==-m)
                if tpe.sum()>0:
                    return level
                I_minors.append(I_complement)
        matrices = I_minors
        level -= 1
    return level


#%%
# Functions to check full-dimensionality of polyhedral sets using either matrix I-rank or Linear Programming
###

def exists_a_solution_using_I_rank_Computation_Method(A:np.ndarray, b:Optional[np.ndarray]=None, res:Optional[float]=None) -> bool:
    """
    Parameters:
    * A: 2D matrix, as ndarray of shape (m,n) ;
    * b: 1D vector, as ndarray of shape (m,) ;
    * res: precision float >=0 to solve A * x + b > 0 (Psi open) ; if 0, ~should solve A * x + b > 0.\n
    => Returns True iif a solution x exists in the system of linear inequalities: A * x + b > 0.
    """
    if b is not None:
        A = np.concatenate([A, b[:,np.newaxis]], axis=1, dtype=A.dtype)
        A = np.concatenate([A, [(0,)*(A.shape[1]-1)+(1,)]], axis=0, dtype=A.dtype)
    return I_rank(A, res) > 0

def exists_a_solution_using_Linear_Programming_Feasibility_Method(A:np.ndarray, b:Optional[np.ndarray]=None, res:Optional[float]=None) -> bool:
    """
    Parameters:
    * A: 2D matrix, as ndarray of shape (m,n) ;
    * b: 1D vector, as ndarray of shape (m,) ;
    * res: precision float >0 to solve A * x + b > 0 (Psi open) ; float <= 0 to solve A * x + b >= 0.\n
    => Returns True iif a solution x exists in the system of linear inequalities: A * x + b > 0.
    """
    if res is None:
        res = mres(A, 1 + max(0, int(np.log10(np.max(np.abs(A)))))) # error committed (>0)
    res = 1e-6 * np.sign(res) # TODO: better adapt this parameter! ???
    return linprog(np.zeros(A.shape[1]), A_ub=-A, b_ub=b-res, bounds=(None,None), method='highs', options={'presolve':False}).success

def exists_a_solution(A:np.ndarray, b:Optional[np.ndarray]=None, res:Optional[float]=None, method:Literal['LP','IR']='LP') -> bool:
    """
    Parameters:
    * A: 2D matrix, as ndarray of shape (m,n) ;
    * b: 1D vector, as ndarray of shape (m,) ;
    * method: to check consistency of linear system, either 'LP' for Linear Programming or 'IR' for I-rank ;
    * res: precision float >=0 to solve A * x + b > 0 (Psi open) ; if 0, ~should solve A * x + b > 0.\n
    => Returns True iif a solution x exists in the system of linear inequalities: A * x + b > 0.
    """
    if method == 'LP':
        return exists_a_solution_using_Linear_Programming_Feasibility_Method(A, b, res)
    elif method == 'IR':
        return exists_a_solution_using_I_rank_Computation_Method(A, b, res)
    raise ValueError("Parameter 'method' must be either 'LP' or 'IR'.")

def polyhedron_is_full_dimensional(V:np.ndarray, P:np.ndarray, res:Optional[float]=None) -> bool:
    """
    Parameters:
    * V: family of m vectors of size n directing the m hyperplans (HP) and pointing outward the half-spaces (HS), as ndarray of shape (m,n) ;
    * P: family of m points of size n belonging to the m hyperplans, as ndarray of shape (m,n) ;
    * res: precision float >=0 to solve A * x + b > 0 (Psi open) ; if 0, ~should solve A * x + b > 0.\n
    => Returns True iif the interior of polyhedron Psi_h, i.e. the intersection of all half-spaces defined by ordered pairs h=(P,V), is not empty.
    """
    #V = normed(V) # important to norm direction vectors V ?
    #P = global_standardized(P) # important to standardize points P ?
    A = - V
    b = scalar(P,V)
    return exists_a_solution(A, b, res)


#%%
# Functions to check whether a half-space is necessary and compute the minimal H-description of a polyhedral set (keep_only_necessary_pairs)
###

def pair_is_necessary(i:int, V:np.ndarray, P:np.ndarray, res:Optional[float]=None) -> bool:
    """
    Parameters:
    * V: family of m vectors of size n directing the m hyperplans (HP) and pointing outward the half-spaces (HS), as ndarray of shape (m,n) ;
    * P: family of m points of size n belonging to the m hyperplans, as ndarray of shape (m,n) ;
    * res: precision float >=0 to solve A * x + b > 0 (Psi open) ; if 0, ~should solve A * x + b > 0.\n
    => Returns True iif the i-th ordered pair in h=(P,V) is necessary for the construction of polyhedron Psi_h.
    """
    V_prime = V.copy()
    V_prime[i] = - V_prime[i]
    return polyhedron_is_full_dimensional(V_prime, P, res)

def keep_only_necessary_pairs(h:np.ndarray, eps:Optional[float]=None) -> np.ndarray:
    """
    Compute the minimal H-description of the polyhedron described by h. 
    Parameter h is a list of halfspaces, defined by ordered pairs of vectors (c,v).
    """
    i = h.shape[0] - 1
    while i >= 0:
        if not pair_is_necessary(i, h[:,1], h[:,0], eps):
            h = np.delete(h, i, axis=0)
        i -= 1
    return h

def keep_only_necessary_pairs_idx(h:np.ndarray, eps:Optional[float]=None) -> np.ndarray:
    """
    Compute the halfspace indices in h that form the minimal H-description of its associated polyhedron.
    Parameter h is a list of halfspaces, defined by ordered pairs of vectors (c,v).
    """
    idx = np.ones(shape=h.shape[0], dtype=bool)
    i = h.shape[0] - 1
    while i >= 0:
        if not pair_is_necessary(i, h[:,1], h[:,0], eps):
            h = np.delete(h, i, axis=0)
            idx[i] = False
        i -= 1
    return idx

