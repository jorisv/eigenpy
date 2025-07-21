import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))
B = rng.random((dim, dim))

realqz = eigenpy.RealQZ(A, B)
assert realqz.info() == eigenpy.ComputationInfo.Success

Q = realqz.matrixQ()
S = realqz.matrixS()
Z = realqz.matrixZ()
T = realqz.matrixT()

assert eigenpy.is_approx(A, Q @ S @ Z)
assert eigenpy.is_approx(B, Q @ T @ Z)

assert eigenpy.is_approx(Q @ Q.T, np.eye(dim))
assert eigenpy.is_approx(Z @ Z.T, np.eye(dim))

for i in range(dim):
    for j in range(i):
        assert abs(T[i, j]) < 1e-12

for i in range(dim):
    for j in range(i - 1):
        assert abs(S[i, j]) < 1e-12

realqz_no_qz = eigenpy.RealQZ(A, B, False)
assert realqz_no_qz.info() == eigenpy.ComputationInfo.Success
S_no_qz = realqz_no_qz.matrixS()
T_no_qz = realqz_no_qz.matrixT()

for i in range(dim):
    for j in range(i):
        assert abs(T_no_qz[i, j]) < 1e-12

for i in range(dim):
    for j in range(i - 1):
        assert abs(S_no_qz[i, j]) < 1e-12

realqz_compute_no_qz = eigenpy.RealQZ(dim)
result_no_qz = realqz_compute_no_qz.compute(A, B, False)
assert result_no_qz.info() == eigenpy.ComputationInfo.Success
S_compute_no_qz = realqz_compute_no_qz.matrixS()
T_compute_no_qz = realqz_compute_no_qz.matrixT()

realqz_with_qz = eigenpy.RealQZ(dim)
realqz_without_qz = eigenpy.RealQZ(dim)

result_with = realqz_with_qz.compute(A, B, True)
result_without = realqz_without_qz.compute(A, B, False)

assert result_with.info() == eigenpy.ComputationInfo.Success
assert result_without.info() == eigenpy.ComputationInfo.Success

S_with = realqz_with_qz.matrixS()
T_with = realqz_with_qz.matrixT()
S_without = realqz_without_qz.matrixS()
T_without = realqz_without_qz.matrixT()

assert eigenpy.is_approx(S_with, S_without)
assert eigenpy.is_approx(T_with, T_without)

iterations = realqz.iterations()
assert iterations >= 0

realqz_iter = eigenpy.RealQZ(dim)
realqz_iter.setMaxIterations(100)
realqz_iter.setMaxIterations(500)
result_iter = realqz_iter.compute(A, B)
assert result_iter.info() == eigenpy.ComputationInfo.Success

realqz1_id = eigenpy.RealQZ(dim)
realqz2_id = eigenpy.RealQZ(dim)
id1 = realqz1_id.id()
id2 = realqz2_id.id()
assert id1 != id2
assert id1 == realqz1_id.id()
assert id2 == realqz2_id.id()

realqz3_id = eigenpy.RealQZ(A, B)
realqz4_id = eigenpy.RealQZ(A, B)
id3 = realqz3_id.id()
id4 = realqz4_id.id()
assert id3 != id4
assert id3 == realqz3_id.id()
assert id4 == realqz4_id.id()

realqz5_id = eigenpy.RealQZ(A, B, True)
realqz6_id = eigenpy.RealQZ(A, B, False)
id5 = realqz5_id.id()
id6 = realqz6_id.id()
assert id5 != id6
assert id5 == realqz5_id.id()
assert id6 == realqz6_id.id()
