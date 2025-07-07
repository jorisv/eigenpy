import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

opt_U = eigenpy.DecompositionOptions.ComputeFullU
opt_V = eigenpy.DecompositionOptions.ComputeFullV

bdcsvd = eigenpy.BDCSVD(A, opt_U | opt_V)
assert bdcsvd.info() == eigenpy.ComputationInfo.Success

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

S = np.diag(singularvalues)
V_adj = V.conj().T
assert eigenpy.is_approx(A, U @ S @ V_adj)

bdcsvd.setThreshold(1e-8)
threshold = bdcsvd.threshold()

bdcsvd.setThreshold()
threshold = bdcsvd.threshold()

rank = bdcsvd.rank()
bdcsvd.setSwitchSize(10)
