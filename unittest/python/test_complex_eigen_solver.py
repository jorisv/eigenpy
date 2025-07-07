import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

ces = eigenpy.ComplexEigenSolver(A)

assert ces.info() == eigenpy.ComputationInfo.Success

V = ces.eigenvectors()
D = ces.eigenvalues()

assert eigenpy.is_approx(A.dot(V).real, V.dot(np.diag(D)).real)
assert eigenpy.is_approx(A.dot(V).imag, V.dot(np.diag(D)).imag)

ces.setMaxIterations(10)
assert ces.getMaxIterations() == 10
