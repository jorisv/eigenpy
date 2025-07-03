import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

bdcsvd = eigenpy.BDCSVD(A, 24)

# Solve
X = rng.random((dim, 20))
B = A.dot(X)
X_est = bdcsvd.solve(B)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)

# Others
cols = bdcsvd.cols()
rows = bdcsvd.rows()

comp_U = bdcsvd.computeU()
comp_V = bdcsvd.computeV()

U = bdcsvd.matrixU()
V = bdcsvd.matrixV()

nonzerosingval = bdcsvd.nonzeroSingularValues()
singularvalues = bdcsvd.singularValues()

bdcsvd.setThreshold(1e-8)
threshold = bdcsvd.threshold()
rank = bdcsvd.rank()
