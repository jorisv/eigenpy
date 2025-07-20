import numpy as np

import eigenpy

dim = 100
rng = np.random.default_rng()
A = rng.random((dim, dim))

ces = eigenpy.ComplexEigenSolver()
es = eigenpy.ComplexEigenSolver(dim)
es = eigenpy.ComplexEigenSolver(A)
assert es.info() == eigenpy.ComputationInfo.Success

es_with_vectors = eigenpy.ComplexEigenSolver(A, True)
assert es_with_vectors.info() == eigenpy.ComputationInfo.Success

es_without_vectors = eigenpy.ComplexEigenSolver(A, False)
assert es_without_vectors.info() == eigenpy.ComputationInfo.Success

V = es.eigenvectors()
D = es.eigenvalues()
assert V.shape == (dim, dim)
assert D.shape == (dim,)

AV = A @ V
VD = V @ np.diag(D)
assert eigenpy.is_approx(AV.real, VD.real)
assert eigenpy.is_approx(AV.imag, VD.imag)

es_compute = eigenpy.ComplexEigenSolver()
result = es_compute.compute(A, False)
assert result.info() == eigenpy.ComputationInfo.Success
D_only = es_compute.eigenvalues()
assert D_only.shape == (dim,)

result_with_vectors = es_compute.compute(A, True)
assert result_with_vectors.info() == eigenpy.ComputationInfo.Success
V_computed = es_compute.eigenvectors()
D_computed = es_compute.eigenvalues()
assert V_computed.shape == (dim, dim)
assert D_computed.shape == (dim,)

trace_A = np.trace(A)
trace_D = np.sum(D)
assert abs(trace_A - trace_D.real) < 1e-10
assert abs(trace_D.imag) < 1e-10

es_default = eigenpy.ComplexEigenSolver()
result_default = es_default.compute(A)
assert result_default.info() == eigenpy.ComputationInfo.Success
V_default = es_default.eigenvectors()
D_default = es_default.eigenvalues()

es_iter = eigenpy.ComplexEigenSolver(A)
default_iter = es_iter.getMaxIterations()
es_iter.setMaxIterations(100)
assert es_iter.getMaxIterations() == 100
es_iter.setMaxIterations(200)
assert es_iter.getMaxIterations() == 200

assert es.info() == eigenpy.ComputationInfo.Success

ces1 = eigenpy.ComplexEigenSolver()
ces2 = eigenpy.ComplexEigenSolver()
id1 = ces1.id()
id2 = ces2.id()
assert id1 != id2
assert id1 == ces1.id()
assert id2 == ces2.id()

dim_constructor = 3
ces3 = eigenpy.ComplexEigenSolver(dim_constructor)
ces4 = eigenpy.ComplexEigenSolver(dim_constructor)
id3 = ces3.id()
id4 = ces4.id()
assert id3 != id4
assert id3 == ces3.id()
assert id4 == ces4.id()

ces5 = eigenpy.ComplexEigenSolver(A)
ces6 = eigenpy.ComplexEigenSolver(A)
id5 = ces5.id()
id6 = ces6.id()
assert id5 != id6
assert id5 == ces5.id()
assert id6 == ces6.id()

ces7 = eigenpy.ComplexEigenSolver(A, True)
ces8 = eigenpy.ComplexEigenSolver(A, False)
id7 = ces7.id()
id8 = ces8.id()
assert id7 != id8
assert id7 == ces7.id()
assert id8 == ces8.id()
