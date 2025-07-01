/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/sparse/QR.hpp"

namespace eigenpy {
void exposeSparseQRSolver() {
  using namespace Eigen;
  typedef SparseMatrix<double, ColMajor> ColMajorSparseMatrix;
  SparseQRVisitor<ColMajorSparseMatrix>::expose("SparseQR");
}
}  // namespace eigenpy
