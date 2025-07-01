import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()

A = rng.random((dim, dim))
A = (A + A.T) * 0.5 + np.diag(10.0 + rng.random(dim))

partialpivlu = eigenpy.PartialPivLU(A)

# Solve
X = rng.random((dim, 20))
B = A.dot(X)
X_est = partialpivlu.solve(B)
assert eigenpy.is_approx(X, X_est)
assert eigenpy.is_approx(A.dot(X_est), B)

# Others
cols = partialpivlu.cols()
rows = partialpivlu.rows()

det = partialpivlu.determinant()
rcond = partialpivlu.rcond()

matrixLU = partialpivlu.matrixLU()
permutationP = partialpivlu.permutationP()
reconstructed_matrix = partialpivlu.reconstructedMatrix()
