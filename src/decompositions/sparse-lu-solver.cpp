/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/decompositions/sparse/LU.hpp"

namespace eigenpy {
void exposeSparseLUSolver() {
  using namespace Eigen;
  typedef SparseMatrix<double, ColMajor> ColMajorSparseMatrix;
  SparseLUVisitor<ColMajorSparseMatrix>::expose("SparseLU");
}
}  // namespace eigenpy
