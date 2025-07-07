import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

tri = eigenpy.Tridiagonalization(A)
