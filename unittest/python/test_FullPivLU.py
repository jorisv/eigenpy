import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

fullpivlu = eigenpy.FullPivLU(A)

# Solve
X = rng.random((dim, 20))
B = A.dot(X)
X_est = fullpivlu.solve(B)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)

# Others
cols = fullpivlu.cols()
rows = fullpivlu.rows()

det = fullpivlu.determinant()
dim_kernel = fullpivlu.dimensionOfKernel()
rank = fullpivlu.rank()
rcond = fullpivlu.rcond()
max_pivot = fullpivlu.maxPivot()
nonzero_pivots = fullpivlu.nonzeroPivots()

is_injective = fullpivlu.isInjective()
is_invertible = fullpivlu.isInvertible()
is_surjective = fullpivlu.isSurjective()

fullpivlu.setThreshold(1e-8)
threshold = fullpivlu.threshold()

fullpivlu.setThreshold()
threshold = fullpivlu.threshold()

image = fullpivlu.image(A)
inverse = fullpivlu.inverse()
kernel = fullpivlu.kernel()

LU = fullpivlu.matrixLU()
P = fullpivlu.permutationP()
Q = fullpivlu.permutationQ()

reconstructed_matrix = fullpivlu.reconstructedMatrix()
assert eigenpy.is_approx(reconstructed_matrix, A)
