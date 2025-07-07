import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))
B = rng.random((dim, dim))
B = (B + B.T) * 0.5 + np.diag(10.0 + rng.random(dim))  # Make B not singular

ges = eigenpy.GeneralizedEigenSolver(A, B)

assert ges.info() == eigenpy.ComputationInfo.Success

alphas = ges.alphas()
betas = ges.betas()

vec = ges.eigenvectors()

val_est = alphas / betas
for i in range(dim):
    v = vec[:, i]
    lam = val_est[i]

    Av = A @ v
    lam_Bv = lam * (B @ v)

    assert eigenpy.is_approx(Av.real, lam_Bv.real)
    assert eigenpy.is_approx(Av.imag, lam_Bv.imag)
