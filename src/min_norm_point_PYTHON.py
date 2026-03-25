import time
import numpy as np

from typing import Tuple, List, Optional

from src.polyset import scalar, norm, mres
from src.polyset import keep_only_necessary_pairs, in_polyhedron
from src.polyset import polyhedron_is_full_dimensional


#%%
# Function to check whether a point q in a polyhedral set P(h) is of minimal distance to a reference point p not in P(h)
###

def q_is_not_minor(q:np.ndarray, h:np.ndarray, p:Optional[np.ndarray]=None, res:Optional[float]=None) -> bool:
    """
    Parameters:
    * q: any point in space, different from p, with shape (n,) ;
    * h: family of vector couples (c,v) defining half-spaces whose intersection is the polyhedron Psi_h to consider, with shape (m,2,n) ;
    * p: reference point outside Psi_h, with shape (n,) [zero by default];
    * res: precision float >=0 to solve A * x + b > 0 (Psi open) ; if 0, ~should solve A * x + b > 0.\n
    => Returns True iif the i-th couple in h=(P,V) is necessary for the construction of polyhedron Psi_h.
    """
    #p_q = normed(q-p) # not necessary (costs time), but better for precision!?
    if p is None:
        h_0 = np.array([q, q])[np.newaxis]
    else:
        h_0 = np.array([q, q-p])[np.newaxis]
    new_h = np.concatenate((h_0, h), axis=0, dtype=h.dtype)
    return polyhedron_is_full_dimensional(new_h[:,1], new_h[:,0], res)


#%%
# Main function to compute distance from point 'p' to polyhedron 'P_h'. 
# Version 0.
##

# Recursive function for nearest-point algorithm
def __f0(h_original:np.ndarray, q:np.ndarray, h:np.ndarray, U:np.ndarray, eps:float=1e-6, steps:List=[], step:int=0) -> Tuple[np.ndarray, bool]:
    """Recursive function for nearest-point algorithm (V0)."""
    # p_original = 0
    # NO condition on family (v_i)_i
    steps.append(step)

    q_not_in_psi = np.max(scalar(q - h_original[:,0], h_original[:,1])) > eps
    U_not_fullsize = U.shape[0] < q.shape[0]
    h_not_empty = h.shape[0] > 0

    if q_not_in_psi and U_not_fullsize and h_not_empty: # we can go further in the projections!
        
        proj = scalar(q - h[:,0], h[:,1])
        positive = proj > eps # there is at least one True element, as max(proj) > 0

        v_prime = h[:,1] - scalar(scalar(h[:,1,np.newaxis], U, keepdims=True).transpose(0,2,1), U.T) # (|h|, n)
        v_prime_norm = norm(v_prime) # (|h|,)
        v_independant = v_prime_norm > eps # are the couples (c,v) in h for which v is linearly independant from U

        positive_and_independant = positive * v_independant

        if positive_and_independant.sum() == 0: # no couple for which proj is positive and v is linearly independant from U: we turn back!
            return q, False

        v_prime = v_prime[positive_and_independant] / v_prime_norm[positive_and_independant, np.newaxis] # (nbi, n)
        distances = proj[positive_and_independant] / scalar(h[positive_and_independant,1], v_prime) # == <q-c',v'>
        q_p_list = q - distances[..., np.newaxis] * v_prime # q projected on each of the positive AND independant hyperplanes

        IDX_order = list(np.argsort(distances)[::-1]) ### ARG on h_p
        IDX_pos_and_indep = np.argwhere(positive_and_independant) ### ARG on h

        q_p = q
        state = False

        while state is False and len(IDX_order) > 0:
            
            i = IDX_order.pop(0)
            v_independant[IDX_pos_and_indep[i]] = False

            q_p = q_p_list[i]
            h_p = h[v_independant]
            U_p = np.concatenate((U, [v_prime[i]]), axis=0)
            
            q_p, state = __f0(h_original, q_p, h_p, U_p, eps, steps, step+1)

        if state:
            return q_p, state
        return q, state
    
    elif q_not_in_psi:
        return q, False
    
    elif U.shape[0] <= 1: # q is necessarily minor, because only projected on one hyperplane at max!
        return q, True

    elif q_is_not_minor(q, h_original, res = 1e1 * eps): # only 1e1 (* ...) ?
        return q, False
    
    return q, True # q is in Psi AND is minor


# Main function for nearest-point algorithm
def algo_0(h:np.ndarray, p:Optional[np.ndarray]=None, eps:Optional[float]=None, verify_h:bool=False) -> np.ndarray:
    """
    Nearest-point algorithm (V0).
    """
    # precision: must be non-negative
    # verify_h: if False, no checking on the state of h and no reduction of h (Ps_i is supposed to be full-dimensional)

    n = h.shape[-1]
    
    if p is None:
        p = np.zeros(n)

    if np.prod(h.shape) == 0: # h is empty
        return p, []

    v_norm = norm(h[:,1])
    nonull = v_norm > mres(h)

    if nonull.sum() == 0: # all v are null
        return p, []

    h = h[nonull]
    h[:,0] = h[:,0] - p
    h[:,1] = h[:,1] / v_norm[nonull,np.newaxis]
    
    if eps is None:
        eps = min(1e-6, mres(h, 1 + max(0, int(np.log10(n * np.max(np.abs(h[:,0]))))))) # is it a good 'eps' ???
    
    if np.min(scalar(h[:,0], h[:,1])) >= -eps:
        return p, []

    if not verify_h:

        q = np.zeros_like(p)
        U = np.zeros(shape=(0,n), dtype=p.dtype)

        steps = []

        q, state = __f0(h, q, h, U, eps, steps)

        if state is False:
            print("FAILURE: No point found! Please chose a better-adapted precision value (higher!).","\n")
            print("h:")
            print(h,"\n")
            print("p:")
            print(p,"\n")

        return q + p, steps
    
    elif polyhedron_is_full_dimensional(h[:,1], h[:,0], eps): # polyhedron_is_full_dimensional: costs time!
        
        h = keep_only_necessary_pairs(h, eps) # keep_only_necessary_pairs: costs time!

        q = np.zeros_like(p)
        U = np.zeros(shape=(0,n), dtype=p.dtype)

        steps = []

        q, state = __f0(h, q, h, U, eps, steps)

        if state is False:
            print("FAILURE: No point found! Please chose a better-adapted precision value (higher!).","\n")
            print("h:")
            print(h,"\n")
            print("p:")
            print(p,"\n")

        return q + p, steps

    elif True:
        print("Warning: polyhedron is not full-dimensional!")
        return None, []
    
    else:
        print("Warning: polyhedron is empty!")
        return np.full(shape=p.shape, fill_value=np.nan), []


#%%
# MAIN FUNCTION TO COMPUTE THE MINIMUM-NORM POINTS TO CONVEX POLYHEDRA (Python version -> very slow!)
###

def minimum_norm_points_to_polyhedra_PYTHON(data:np.ndarray, h:list[np.ndarray], infos:bool=True) -> np.ndarray:
    """
    Main Python function to compute the minimum-norm points to convex polyhedra. 
    * data: ndarray of points in ndim-dimensional real vector space,
     with shape (n_samples, ndim) or (ndim,) ;
    * h: list of polyhedra represented as intersection of half_spaces described by couples (c,v),
     with shape n_classes * (n_half_spaces, 2, ndim) or (n_half_spaces, 2, ndim).\n
    Returns ndarray of minimum-norm points from data to each class polyhedra, 
    with shape (n_samples, n_classes, ndim) or (n_classes, ndim) or (n_samples, ndim) or (ndim,).
    """
    data_is_1D = data.ndim == 1
    h_is_3D = type(h) is np.ndarray and h.ndim == 3

    if data_is_1D:
        data = data[np.newaxis]
    if h_is_3D:
        h = h[np.newaxis]
    
    n_samples, ndim = data.shape
    n_classes = len(h)

    min_n_pts = np.empty(shape=(n_samples, n_classes, ndim), dtype=data.dtype)

    estimated = False
    start = time.time()

    # For each polyhedron in h, compute the minimum-norm points for all points in data
    for c in range(n_classes):

        if infos:
            print(f"* Processing class: {c+1} / {n_classes}...", end=' ')
        
        # Get half-space vector couples forming the polyhedron related to class c
        h_class = h[c]

        # Directly associate p themselves to points p which belong to h_class
        p_in_h = in_polyhedron(data=data, polyhedron=h_class)
        min_n_pts[p_in_h,c] = data[p_in_h]
        arg_data_not_in_h = np.argwhere(~p_in_h)[:,0]
        n_samples_not_in_h = arg_data_not_in_h.shape[0]

        # Compute corresponding minimum-norm points q to the other points p
        for i in range(n_samples_not_in_h):
            
            # Estimate the computation time from the already-computed sample
            if infos and not estimated and time.time() - start > 1 and c * n_samples + i > 0:
                delta_time = time.time() - start
                total_iter = n_samples * n_classes
                n_iter_now = c * n_samples + i
                total_time = total_iter / n_iter_now * delta_time
                if total_time > 60:
                    total_time = total_time / 60
                    if total_time > 60:
                        total_time = total_time / 60
                        unit = "hours"
                    else:
                        unit = "minutes"
                else:
                    unit = "seconds"
                decimals = 2
                total_time = np.round(total_time, decimals)
                print(f"\nEstimated computation time: {total_time} {unit}")
                estimated = True
            
            # Compute the minimum-norm point q from p to h_class with algo
            arg = arg_data_not_in_h[i]
            p = data[arg]
            q, _ = algo_0(h_class, p, eps=1e-8)
            min_n_pts[arg,c] = q

        if infos:
            print("Done!")
    
    if data_is_1D:
        min_n_pts = min_n_pts[0]
        if h_is_3D:
            min_n_pts = min_n_pts[0]
    elif h_is_3D:
        min_n_pts = min_n_pts[:,0]

    return min_n_pts

