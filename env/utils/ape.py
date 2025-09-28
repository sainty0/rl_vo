# env/utils/ape.py
import numpy as np

# ---------------------------
# TUM I/O
# ---------------------------
def load_tum(path: str):
    """
    Returns times (N,), positions (N,3), quats (N,4) with rows sorted by time.
    TUM: timestamp tx ty tz qx qy qz qw
    """
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    arr = arr[np.argsort(arr[:, 0])]
    t = arr[:, 0].astype(np.float64)
    p = arr[:, 1:4].astype(np.float64)
    q = arr[:, 4:8].astype(np.float64)
    return t, p, q

# ---------------------------
# Time association
# ---------------------------
def _associate_nearest(t_ref, t_query, max_dt=0.05):
    """
    For each t_query, pick nearest t_ref within max_dt.
    Returns indices (i_ref, i_query) of matches.
    """
    i_ref = []
    i_query = []
    j = 0
    for i, tq in enumerate(t_query):
        j = np.searchsorted(t_ref, tq, side="left")
        candidates = []
        if j > 0:
            candidates.append(j - 1)
        if j < len(t_ref):
            candidates.append(j)
        if not candidates:
            continue
        # choose nearest
        jj = min(candidates, key=lambda k: abs(t_ref[k] - tq))
        if abs(t_ref[jj] - tq) <= max_dt:
            i_ref.append(jj)
            i_query.append(i)
    return np.asarray(i_ref, dtype=np.int64), np.asarray(i_query, dtype=np.int64)

def _slerp(q0, q1, u):
    """Quaternion slerp; q0,q1 (4,), u in [0,1] -> (4,)"""
    q0 = q0 / (np.linalg.norm(q0) + 1e-12)
    q1 = q1 / (np.linalg.norm(q1) + 1e-12)
    dot = np.clip(np.dot(q0, q1), -1.0, 1.0)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        # almost colinear -> lerp
        out = q0 + u * (q1 - q0)
        return out / (np.linalg.norm(out) + 1e-12)
    theta_0 = np.arccos(dot)
    sin_0 = np.sin(theta_0)
    theta = theta_0 * u
    s0 = np.sin(theta_0 - theta) / (sin_0 + 1e-12)
    s1 = np.sin(theta) / (sin_0 + 1e-12)
    out = s0 * q0 + s1 * q1
    return out / (np.linalg.norm(out) + 1e-12)

def _interp_at(times_src, pos_src, quat_src, t_query):
    """
    Linear interpolate position and slerp quaternion at t_query.
    Assumes times_src sorted ascending.
    Returns pos_q (M,3), quat_q (M,4)
    """
    idx = np.searchsorted(times_src, t_query, side="left")
    idx0 = np.clip(idx - 1, 0, len(times_src) - 1)
    idx1 = np.clip(idx,       0, len(times_src) - 1)
    t0 = times_src[idx0]; t1 = times_src[idx1]
    # avoid div by 0: if identical, u=0
    denom = (t1 - t0)
    denom[denom == 0.0] = 1.0
    u = np.clip((t_query - t0) / denom, 0.0, 1.0)

    p0 = pos_src[idx0]; p1 = pos_src[idx1]
    q0 = quat_src[idx0]; q1 = quat_src[idx1]
    p = p0 + (p1 - p0) * u[:, None]
    q = np.vstack([_slerp(q0[i], q1[i], float(u[i])) for i in range(len(u))])
    return p, q

# ---------------------------
# Alignment (SE3)
# ---------------------------
def umeyama_se3(P, Q):
    """
    Find R,t that minimizes || R*P + t - Q ||^2 (no scale).
    P,Q are (N,3) matched points.
    Returns R(3x3), t(3,)
    """
    assert P.shape == Q.shape and P.shape[1] == 3
    mu_P = P.mean(axis=0)
    mu_Q = Q.mean(axis=0)
    X = P - mu_P
    Y = Q - mu_Q
    Sigma = X.T @ Y / P.shape[0]
    U, S, Vt = np.linalg.svd(Sigma)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    t = mu_Q - R @ mu_P
    return R, t

# ---------------------------
# Windows & RMSE
# ---------------------------
def _window_last_seconds(t, seconds):
    if seconds is None or seconds <= 0:
        return np.ones_like(t, dtype=bool)
    t_end = t[-1]
    return t >= (t_end - seconds)

def ape_rmse(
    est_tum: str,
    gt_tum: str,
    max_dt: float = 0.05,
    use_interpolation: bool = True,
    align: str = "se3",
    score_last_seconds: float = None,
) -> float:
    """
    Compute APE RMSE with timestamp association and SE3 alignment.
    - If use_interpolation: interpolate GT at est times (best when rates differ)
    - Else: nearest neighbor association within max_dt
    - score_last_seconds: if set, compute RMSE over the last N seconds
    """
    try:
        tE, PE, qE = load_tum(est_tum)
        tG, PG, qG = load_tum(gt_tum)
        if len(tE) < 2 or len(tG) < 2:
            return 20.0

        if use_interpolation:
            # Interpolate GT at estimator timestamps
            mask_win = _window_last_seconds(tE, score_last_seconds)
            tEval = tE[mask_win]
            P_eval = PE[mask_win]
            P_ref, _ = _interp_at(tG, PG, qG, tEval)
        else:
            # Nearest neighbor association
            mask_win = _window_last_seconds(tE, score_last_seconds)
            tE_win = tE[mask_win]
            idxG, idxE_rel = _associate_nearest(tG, tE_win, max_dt=max_dt)
            if len(idxG) < 5:
                return 20.0
            P_ref = PG[idxG]
            P_eval = PE[mask_win][idxE_rel]

        # Align eval trajectory to reference (GT) in SE3 (no scale)
        if align.lower() == "se3":
            R, t = umeyama_se3(P_eval, P_ref)
            P_aligned = (R @ P_eval.T).T + t
        elif align.lower() in ("none", "off"):
            P_aligned = P_eval
        else:
            # For monocular VO you could add Sim3 with scale; not needed for LIO-SAM.
            R, t = umeyama_se3(P_eval, P_ref)
            P_aligned = (R @ P_eval.T).T + t

        err = np.linalg.norm(P_aligned - P_ref, axis=1)
        if len(err) < 5:
            return 20.0
        rmse = float(np.sqrt(np.mean(err ** 2)))
        return rmse
    except Exception:
        return 20.0

