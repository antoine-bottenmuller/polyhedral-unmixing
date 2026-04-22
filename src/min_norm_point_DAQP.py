import numpy as np
from qpsolvers import solve_qp

def _normalize_halfspaces(h: np.ndarray, eps: float = 1e-15):
    """
    h: array shape (m, 2, ndim)
       h[:, 0] = c_i (point on hyperplane)
       h[:, 1] = v_i (outward normal)

    Returns
    -------
    V : (m, ndim)
        normalized normals
    alpha : (m,)
        right-hand side so that V x <= alpha
    """
    h = np.asarray(h)
    C = np.asarray(h[:, 0], dtype=np.float64)
    V = np.asarray(h[:, 1], dtype=np.float64)

    vn = np.linalg.norm(V, axis=1)
    keep = vn > eps
    if not np.any(keep):
        return np.empty((0, h.shape[-1]), dtype=np.float64), np.empty((0,), dtype=np.float64)

    C = C[keep]
    V = V[keep] / vn[keep, None]

    alpha = np.einsum("ij,ij->i", C, V)
    return V, alpha


def in_polyhedron_qp(data: np.ndarray, V: np.ndarray, alpha: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """
    data: (n_samples, ndim) or (ndim,)
    returns boolean mask of shape (n_samples,) or scalar-bool-like array
    """
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return np.max(V @ data - alpha) <= tol
    return np.max(data @ V.T - alpha[None, :], axis=1) <= tol


def project_point_to_polyhedron_daqp(
    p: np.ndarray,
    V: np.ndarray,
    alpha: np.ndarray,
    initvals: np.ndarray | None = None,
    primal_tol: float = 1e-8,
    dual_tol: float = 1e-8,
    iter_limit: int | None = None,
    time_limit: float | None = None
) -> np.ndarray:
    """
    Solve:
        min_x 0.5 * ||x - p||^2
        s.t.  V x <= alpha
    """
    p = np.asarray(p, dtype=np.float64)
    ndim = p.shape[0]

    P = np.eye(ndim, dtype=np.float64)
    q = -p

    kwargs = {
        "solver": "daqp", 
        "initvals": initvals, 
        "primal_tol": primal_tol, 
        "dual_tol": dual_tol, 
        "verbose": False
    }
    if iter_limit is not None:
        kwargs["iter_limit"] = iter_limit
    if time_limit is not None:
        kwargs["time_limit"] = time_limit

    x = solve_qp(P=P, q=q, G=V, h=alpha, **kwargs)

    if x is None:
        raise RuntimeError("DAQP failed to solve the projection QP.")
    return x


def minimum_norm_points_to_polyhedra_DAQP(
    data: np.ndarray,
    h,
    infos: bool = True,
    primal_tol: float = 1e-8,
    dual_tol: float = 1e-8,
    membership_tol: float = 1e-10,
    use_warm_start: bool = True
) -> np.ndarray:
    """
    data: (n_samples, ndim) or (ndim,)
    h: list of polyhedra, each of shape (m_c, 2, ndim), or one array (m, 2, ndim)

    Returns:
        (n_samples, n_classes, ndim), or squeezed version matching your old API.
    """
    data = np.asarray(data, dtype=np.float64)

    data_is_1D = (data.ndim == 1)
    h_is_3D = isinstance(h, np.ndarray) and h.ndim == 3

    if data_is_1D:
        data = data[None, :]
    if h_is_3D:
        h = h[None]

    n_samples, ndim = data.shape
    n_classes = len(h)

    out = np.empty((n_samples, n_classes, ndim), dtype=np.float64)

    for c in range(n_classes):
        if infos:
            print(f"* Processing class {c+1}/{n_classes}...")

        V, alpha = _normalize_halfspaces(h[c])

        # Degenerated case: no useful constraint
        if V.shape[0] == 0:
            out[:, c, :] = data
            continue

        inside = in_polyhedron_qp(data, V, alpha, tol=membership_tol)
        out[inside, c, :] = data[inside]

        outside_idx = np.flatnonzero(~inside)

        # Warm start: we use the previous solution as starting point
        x_prev = None

        for idx in outside_idx:
            p = data[idx]
            x0 = x_prev if use_warm_start else None
            x = project_point_to_polyhedron_daqp(
                p, V, alpha,
                initvals=x0,
                primal_tol=primal_tol,
                dual_tol=dual_tol,
            )
            out[idx, c, :] = x
            x_prev = x

    if data_is_1D:
        out = out[0]
        if h_is_3D:
            out = out[0]
    elif h_is_3D:
        out = out[:, 0]

    return out

