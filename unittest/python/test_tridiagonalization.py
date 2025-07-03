import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

tri = eigenpy.Tridiagonalization(A)

# Q = tri.matrixQ()
# print("Q")
# print(Q)
# Q_conj = Q.conj().T

# T = tri.matrixT()
# print("T")
# print(T)

# assert eigenpy.is_approx(A, Q @ T @ Q_conj)
