import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))
B = rng.random((dim, dim))
B = (B + B.T) * 0.5 + np.diag(10.0 + rng.random(dim))

ges = eigenpy.GeneralizedEigenSolver()
ges_size = eigenpy.GeneralizedEigenSolver(dim)
ges_matrices = eigenpy.GeneralizedEigenSolver(A, B)
assert ges_matrices.info() == eigenpy.ComputationInfo.Success

ges_with_vectors = eigenpy.GeneralizedEigenSolver(A, B, True)
assert ges_with_vectors.info() == eigenpy.ComputationInfo.Success

ges_without_vectors = eigenpy.GeneralizedEigenSolver(A, B, False)
assert ges_without_vectors.info() == eigenpy.ComputationInfo.Success

alphas = ges_matrices.alphas()
betas = ges_matrices.betas()
eigenvectors = ges_matrices.eigenvectors()
eigenvalues = ges_matrices.eigenvalues()

assert alphas.shape == (dim,)
assert betas.shape == (dim,)
assert eigenvectors.shape == (dim, dim)
assert eigenvalues.shape == (dim,)

for k in range(dim):
    v = eigenvectors[:, k]
    lambda_k = eigenvalues[k]

    Av = A @ v
    lambda_Bv = lambda_k * (B @ v)
    assert eigenpy.is_approx(Av.real, lambda_Bv.real, 1e-10)
    assert eigenpy.is_approx(Av.imag, lambda_Bv.imag, 1e-10)

for k in range(dim):
    v = eigenvectors[:, k]
    alpha = alphas[k]
    beta = betas[k]

    alpha_Bv = alpha * (B @ v)
    beta_Av = beta * (A @ v)
    assert eigenpy.is_approx(alpha_Bv.real, beta_Av.real, 1e-10)
    assert eigenpy.is_approx(alpha_Bv.imag, beta_Av.imag, 1e-10)

for k in range(dim):
    if abs(betas[k]) > 1e-12:
        expected_eigenvalue = alphas[k] / betas[k]
        assert abs(eigenvalues[k] - expected_eigenvalue) < 1e-12

ges_compute = eigenpy.GeneralizedEigenSolver()
result = ges_compute.compute(A, B, False)
assert result.info() == eigenpy.ComputationInfo.Success
alphas_only = ges_compute.alphas()
betas_only = ges_compute.betas()
eigenvalues_only = ges_compute.eigenvalues()

result_with_vectors = ges_compute.compute(A, B, True)
assert result_with_vectors.info() == eigenpy.ComputationInfo.Success
eigenvectors_computed = ges_compute.eigenvectors()

ges_default = eigenpy.GeneralizedEigenSolver()
result_default = ges_default.compute(A, B)
assert result_default.info() == eigenpy.ComputationInfo.Success

ges_iter = eigenpy.GeneralizedEigenSolver(A, B)
ges_iter.setMaxIterations(100)
ges_iter.setMaxIterations(200)

A1 = rng.random((dim, dim))
B1 = rng.random((dim, dim))
spdA = A.T @ A + A1.T @ A1
spdB = B.T @ B + B1.T @ B1

ges_spd = eigenpy.GeneralizedEigenSolver(spdA, spdB)
assert ges_spd.info() == eigenpy.ComputationInfo.Success

spd_eigenvalues = ges_spd.eigenvalues()
max_imag = np.max(np.abs(spd_eigenvalues.imag))
assert max_imag < 1e-10

ges1 = eigenpy.GeneralizedEigenSolver()
ges2 = eigenpy.GeneralizedEigenSolver()
id1 = ges1.id()
id2 = ges2.id()
assert id1 != id2
assert id1 == ges1.id()
assert id2 == ges2.id()

ges3 = eigenpy.GeneralizedEigenSolver(dim)
ges4 = eigenpy.GeneralizedEigenSolver(dim)
id3 = ges3.id()
id4 = ges4.id()
assert id3 != id4
assert id3 == ges3.id()
assert id4 == ges4.id()

ges5 = eigenpy.GeneralizedEigenSolver(A, B)
ges6 = eigenpy.GeneralizedEigenSolver(A, B)
id5 = ges5.id()
id6 = ges6.id()
assert id5 != id6
assert id5 == ges5.id()
assert id6 == ges6.id()

ges7 = eigenpy.GeneralizedEigenSolver(A, B, True)
ges8 = eigenpy.GeneralizedEigenSolver(A, B, False)
id7 = ges7.id()
id8 = ges8.id()
assert id7 != id8
assert id7 == ges7.id()
assert id8 == ges8.id()
