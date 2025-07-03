import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))
B = rng.random((dim, dim))

ges = eigenpy.GeneralizedEigenSolver(A, B)

alphas = ges.alphas()
betas = ges.betas()
V = ges.eigenvectors()

eigenvalues = alphas / betas
if np.all(np.abs(betas) >= 1e-15):
    for i in range(dim):
        v = V[:, i]
        lam = eigenvalues[i]

        Av = A @ v
        lam_Bv = lam * (B @ v)

        assert eigenpy.is_approx(Av.real, lam_Bv.real)
        assert eigenpy.is_approx(Av.imag, lam_Bv.imag)
