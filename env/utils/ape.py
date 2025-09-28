import numpy as np

def _load_tum(path: str):
    # TUM: timestamp tx ty tz qx qy qz qw
    arr = np.loadtxt(path)
    if arr.ndim == 1:
        arr = arr[None, :]
    t = arr[:, 0]
    p = arr[:, 1:4]
    return t, p

def ape_rmse(est_tum: str, gt_tum: str) -> float:
    try:
        tE, PE = _load_tum(est_tum)
        tG, PG = _load_tum(gt_tum)
        n = min(len(PE), len(PG))
        if n < 5:
            return 20.0
        # Direct positional RMSE without mean-centering (constant offset counts as error)
        e = PE[:n] - PG[:n]
        return float(np.sqrt(np.mean(np.sum(e * e, axis=1))))
    except Exception:
        return 20.0
