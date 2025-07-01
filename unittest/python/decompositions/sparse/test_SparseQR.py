import numpy as np
import scipy
from scipy.sparse import csc_matrix

import eigenpy

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))
A = csc_matrix(A)

sparseqr = eigenpy.SparseQR(A)

assert sparseqr.info() == eigenpy.ComputationInfo.Success

# Solve
X = rng.random((dim, 20))
B = A.dot(X)
X_est = sparseqr.solve(B)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)

X_sparse = scipy.sparse.random(dim, 10)
B_sparse = A.dot(X_sparse)
B_sparse = B_sparse.tocsc(True)

if not B_sparse.has_sorted_indices:
    B_sparse.sort_indices()

X_est = sparseqr.solve(B_sparse)
assert eigenpy.is_approx(X_est.toarray(), X_sparse.toarray())
assert eigenpy.is_approx(A.dot(X_est.toarray()), B_sparse.toarray())

# Others
cols = sparseqr.cols()
rows = sparseqr.rows()

sparseqr.analyzePattern(A)
sparseqr.factorize(A)

cols_permutation = sparseqr.colsPermutation()
rank = sparseqr.rank()

sparseqr.setPivotThreshold(1e-8)
