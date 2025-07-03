import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))
B = rng.random((dim, dim))

realqz = eigenpy.RealQZ(A, B)

Q = realqz.matrixQ()
S = realqz.matrixS()
Z = realqz.matrixZ()
T = realqz.matrixT()

assert eigenpy.is_approx(A, Q @ S @ Z)
assert eigenpy.is_approx(B, Q @ T @ Z)
