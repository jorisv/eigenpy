import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

opt_U = eigenpy.DecompositionOptions.ComputeFullU
opt_V = eigenpy.DecompositionOptions.ComputeFullV

jacobisvd = eigenpy.JacobiSVD(A, opt_U | opt_V)
assert jacobisvd.info() == eigenpy.ComputationInfo.Success

# Solve
X = rng.random((dim, 20))
B = A.dot(X)
X_est = jacobisvd.solve(B)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)

# Others
cols = jacobisvd.cols()
rows = jacobisvd.rows()

comp_U = jacobisvd.computeU()
comp_V = jacobisvd.computeV()

U = jacobisvd.matrixU()
V = jacobisvd.matrixV()

nonzerosingval = jacobisvd.nonzeroSingularValues()
singularvalues = jacobisvd.singularValues()

S = np.diag(singularvalues)
V_adj = V.conj().T
assert eigenpy.is_approx(A, U @ S @ V_adj)

jacobisvd.setThreshold(1e-8)
threshold = jacobisvd.threshold()

jacobisvd.setThreshold()
threshold = jacobisvd.threshold()

rank = jacobisvd.rank()
