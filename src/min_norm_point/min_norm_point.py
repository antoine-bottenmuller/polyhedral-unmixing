import os
import sys
import time
import numpy as np
from typing import Optional, Literal

# Si le fichier .so n'est pas dans le même répertoire
repository = os.path.dirname(os.path.abspath(__file__))
if repository not in sys.path:
    sys.path.insert(1, repository)

# Importer mon module Python
import min_norm_point_module

# Definition d'une fonction de calcul de resolution
def mres(array:np.ndarray, upmul:int=0) -> float:
    dtype = array.dtype
    if np.issubdtype(dtype, np.floating):
        finfo = np.finfo(dtype)
        decim = int(finfo.precision - upmul)
        nfres = finfo.resolution * np.power(10,upmul)
        return np.round(nfres, decimals=decim)
    return dtype.type(1)

# Definition de la fonction principale en Python
def minimum_norm_points_to_polyhedra(
        points:np.ndarray, 
        V:list[np.ndarray]|np.ndarray, 
        s:list[np.ndarray]|np.ndarray, 
        res:Optional[float]=None, 
        method:Literal["0","1","2","3"]="0", 
        infos:bool=True
) -> np.ndarray:
    """
    * (V,s): list of arrays of vector-scalar couples (v,s) defining a list of polyhedra,
     with shape: n_classes * (n_half_spaces, n + 1) or (n_half_spaces, n + 1) ;
    * points: ndarray of points in n-dimensional real vector space,
     with shape: (n_points, n) or (n,).\n
    Returns ndarray of minimum-norm points from points to each class polyhedra, 
    with shape: (n_points, n_classes, n) or (n_classes, n) or (n_points, n) or (n,).
    """
    # Calcul de res si None
    if res is None:
        res = min(1e-6, mres(s[0], 1 + max(0, int(np.ceil(np.log10(V[0].shape[-1] * np.max([np.max(np.abs(s[i])) for i in range(len(s))])))))))

    # Redimensionnement des points
    points_is_1D = points.ndim == 1
    if points_is_1D:
        n_points = 1
    else: # points.ndim == 2
        n_points = points.shape[0]
    points = points.flatten()
    
    # Redimensionnement de V et C
    V_is_2D = type(V) is np.ndarray and V.ndim == 2
    if type(V) is not np.ndarray: # type(C) is not np.ndarray
        n_polyhedra = len(V)
        cols = V[0].shape[1]
        l_rows = []
        for i in range(n_polyhedra):
            l_rows.append(V[i].shape[0])
        new_V = np.empty(shape=sum(l_rows)*cols, dtype=V[0].dtype)
        new_s = np.empty(shape=sum(l_rows), dtype=s[0].dtype)
        for i in range(n_polyhedra):
            new_V[sum(l_rows[:i])*cols:sum(l_rows[:i+1])*cols] = V[i].flatten()
            new_s[sum(l_rows[:i]):sum(l_rows[:i+1])] = s[i].flatten()
    else:
        if V.ndim == 2:
            n_polyhedra = 1
            rows = V.shape[0]
            cols = V.shape[1]
        else: # V.ndim == 3
            n_polyhedra = V.shape[0]
            rows = V.shape[1]
            cols = V.shape[2]
        l_rows = [rows] * n_polyhedra
        new_V = V.flatten()
        new_s = s.flatten()

    # Conversion des donnees contiguees pour C
    l_V = new_V.astype(np.double)
    l_s = new_s.astype(np.double)
    l_rows = np.asarray(l_rows, dtype=np.int32)
    points = points.astype(np.double)

    l_V = np.ascontiguousarray(l_V)
    l_s = np.ascontiguousarray(l_s)
    l_rows = np.ascontiguousarray(l_rows)
    points = np.ascontiguousarray(points)

    st = time.time()

    # Fonction principale C avec conversion des donnees pour C
    min_n_pts = min_norm_point_module.minimum_norm_points_to_polyhedra(
        l_V, l_s, points, l_rows, 
        n_polyhedra, n_points, cols, 
        res, method
    )

    dt = time.time() - st
    if infos:
        print("Computation time:", dt, "seconds.")

    # Redimensionnement de la sortie
    min_n_pts = min_n_pts.reshape(n_points, n_polyhedra, cols)
    if points_is_1D:
        min_n_pts = min_n_pts[0]
        if V_is_2D:
            min_n_pts = min_n_pts[0]
    elif V_is_2D:
        min_n_pts = min_n_pts[:,0]

    return min_n_pts, dt
