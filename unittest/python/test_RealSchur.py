import numpy as np

import eigenpy


def verify_is_quasi_triangular(T):
    size = T.shape[0]

    for row in range(2, size):
        for col in range(row - 1):
            assert abs(T[row, col]) < 1e-12

    for row in range(1, size):
        if abs(T[row, row - 1]) > 1e-12:
            if row < size - 1:
                assert abs(T[row + 1, row]) < 1e-12

            tr = T[row - 1, row - 1] + T[row, row]
            det = T[row - 1, row - 1] * T[row, row] - T[row - 1, row] * T[row, row - 1]
            assert 4 * det > tr * tr


dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

rs = eigenpy.RealSchur(A)
assert rs.info() == eigenpy.ComputationInfo.Success

U = rs.matrixU()
T = rs.matrixT()

assert eigenpy.is_approx(A, U @ T @ U.T, 1e-10)
assert eigenpy.is_approx(U @ U.T, np.eye(dim), 1e-10)

verify_is_quasi_triangular(T)

A_test = rng.random((dim, dim))
rs1 = eigenpy.RealSchur(dim)
rs1.compute(A_test)
rs2 = eigenpy.RealSchur(A_test)

assert rs1.info() == eigenpy.ComputationInfo.Success
assert rs2.info() == eigenpy.ComputationInfo.Success

T1 = rs1.matrixT()
U1 = rs1.matrixU()
T2 = rs2.matrixT()
U2 = rs2.matrixU()

assert eigenpy.is_approx(T1, T2, 1e-12)
assert eigenpy.is_approx(U1, U2, 1e-12)

rs_no_u = eigenpy.RealSchur(A, False)
assert rs_no_u.info() == eigenpy.ComputationInfo.Success
T_no_u = rs_no_u.matrixT()

assert eigenpy.is_approx(T, T_no_u, 1e-12)

rs_compute_no_u = eigenpy.RealSchur(dim)
result_no_u = rs_compute_no_u.compute(A, False)
assert result_no_u.info() == eigenpy.ComputationInfo.Success
T_compute_no_u = rs_compute_no_u.matrixT()
assert eigenpy.is_approx(T, T_compute_no_u, 1e-12)

rs_iter = eigenpy.RealSchur(dim)
rs_iter.setMaxIterations(40 * dim)  # m_maxIterationsPerRow * size
result_iter = rs_iter.compute(A)
assert result_iter.info() == eigenpy.ComputationInfo.Success
assert rs_iter.getMaxIterations() == 40 * dim

T_iter = rs_iter.matrixT()
U_iter = rs_iter.matrixU()
assert eigenpy.is_approx(T, T_iter, 1e-12)
assert eigenpy.is_approx(U, U_iter, 1e-12)

if dim > 2:
    rs_few_iter = eigenpy.RealSchur(dim)
    rs_few_iter.setMaxIterations(1)
    result_few = rs_few_iter.compute(A)
    assert rs_few_iter.getMaxIterations() == 1

A_triangular = np.triu(A)
rs_triangular = eigenpy.RealSchur(dim)
rs_triangular.setMaxIterations(1)
result_triangular = rs_triangular.compute(A_triangular)
assert result_triangular.info() == eigenpy.ComputationInfo.Success

T_triangular = rs_triangular.matrixT()
U_triangular = rs_triangular.matrixU()

assert eigenpy.is_approx(T_triangular, A_triangular, 1e-10)
assert eigenpy.is_approx(U_triangular, np.eye(dim), 1e-10)

hess = eigenpy.HessenbergDecomposition(A)
H = hess.matrixH()
Q_hess = hess.matrixQ()

rs_from_hess = eigenpy.RealSchur(dim)
result_from_hess = rs_from_hess.computeFromHessenberg(H, Q_hess, True)
assert result_from_hess.info() == eigenpy.ComputationInfo.Success

T_from_hess = rs_from_hess.matrixT()
U_from_hess = rs_from_hess.matrixU()

assert eigenpy.is_approx(A, U_from_hess @ T_from_hess @ U_from_hess.T, 1e-10)

rs1_id = eigenpy.RealSchur(dim)
rs2_id = eigenpy.RealSchur(dim)
id1 = rs1_id.id()
id2 = rs2_id.id()
assert id1 != id2
assert id1 == rs1_id.id()
assert id2 == rs2_id.id()

rs3_id = eigenpy.RealSchur(A)
rs4_id = eigenpy.RealSchur(A)
id3 = rs3_id.id()
id4 = rs4_id.id()
assert id3 != id4
assert id3 == rs3_id.id()
assert id4 == rs4_id.id()

rs5_id = eigenpy.RealSchur(A, True)
rs6_id = eigenpy.RealSchur(A, False)
id5 = rs5_id.id()
id6 = rs6_id.id()
assert id5 != id6
assert id5 == rs5_id.id()
assert id6 == rs6_id.id()
