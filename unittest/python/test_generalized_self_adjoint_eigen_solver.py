import numpy as np

import eigenpy

dim = 5
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5

B = rng.random((dim, dim))
B = B @ B.T + 0.1 * np.eye(dim)

gsaes = eigenpy.GeneralizedSelfAdjointEigenSolver(A, B)

V = gsaes.eigenvectors()
D = gsaes.eigenvalues()

for i in range(dim):
    v = V[:, i]
    lam = D[i]

    Av = A @ v
    lam_Bv = lam * (B @ v)

    assert np.allclose(Av, lam_Bv, atol=1e-6)
