import numpy as np, os, tempfile
from env.utils.ape import ape_rmse

def test_ape_runs():
    with tempfile.TemporaryDirectory() as td:
        p = lambda name: os.path.join(td, name)
        t = np.column_stack([np.arange(10)*0.1,
                             np.zeros((10,3)),
                             np.tile([0,0,0,1], (10,1))])
        np.savetxt(p("gt.tum"), t, fmt="%.6f")
        est = t.copy(); est[:,1] = 1.0
        np.savetxt(p("est.tum"), est, fmt="%.6f")
        rmse = ape_rmse(p("est.tum"), p("gt.tum"))
        assert rmse > 0.0
