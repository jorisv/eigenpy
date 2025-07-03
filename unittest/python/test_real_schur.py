import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

cs = eigenpy.RealSchur(A)

U = cs.matrixU()
T = cs.matrixT()

assert eigenpy.is_approx(A.real, (U @ T @ U.T).real)
