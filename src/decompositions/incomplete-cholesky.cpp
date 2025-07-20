/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/sparse/IncompleteCholesky.hpp"

namespace eigenpy {
void exposeIncompleteCholesky() {
  using namespace Eigen;
  typedef SparseMatrix<double, ColMajor> ColMajorSparseMatrix;
  IncompleteCholeskyVisitor<ColMajorSparseMatrix>::expose("IncompleteCholesky");
}
}  // namespace eigenpy
