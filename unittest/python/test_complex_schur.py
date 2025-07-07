import numpy as np

import eigenpy

dim = 5
rng = np.random.default_rng()
A = rng.random((dim, dim))

cs = eigenpy.ComplexSchur(A)

U = cs.matrixU()
T = cs.matrixT()
U_star = U.conj().T

assert eigenpy.is_approx(A.real, (U @ T @ U_star).real)
assert np.allclose(A.imag, (U @ T @ U_star).imag)

cs.setMaxIterations(10)
assert cs.getMaxIterations() == 10
