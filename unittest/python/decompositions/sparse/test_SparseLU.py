import numpy as np
import scipy.sparse as spa

import eigenpy

dim = 100
rng = np.random.default_rng()

A_fac = spa.random(dim, dim, density=0.25, random_state=rng)
A = A_fac.T @ A_fac
A += spa.diags(10.0 * rng.standard_normal(dim) ** 2)
A = A.tocsc(True)
A.check_format()

splu = eigenpy.SparseLU(A)

assert splu.info() == eigenpy.ComputationInfo.Success

X = rng.random((dim, 20))
B = A.dot(X)
X_est = splu.solve(B)
assert isinstance(X_est, np.ndarray)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)

splu.analyzePattern(A)
splu.factorize(A)

X_sparse = spa.random(dim, 10, random_state=rng)
B_sparse = A.dot(X_sparse)
B_sparse: spa.csc_matrix = B_sparse.tocsc(True)
if not B_sparse.has_sorted_indices:
    B_sparse.sort_indices()

X_est = splu.solve(B_sparse)
assert isinstance(X_est, spa.csc_matrix)
assert eigenpy.is_approx(X_est.toarray(), X_sparse.toarray())
assert eigenpy.is_approx(A.dot(X_est.toarray()), B_sparse.toarray())
