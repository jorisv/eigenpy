/*
 * Copyright 2025 INRIA
 */

#include "eigenpy/decompositions/sparse/QR.hpp"

namespace eigenpy {
void exposeSparseQRSolver() {
  using namespace Eigen;

  typedef SparseMatrix<double, ColMajor> ColMajorSparseMatrix;
  typedef COLAMDOrdering<int> Ordering;
  typedef SparseQR<ColMajorSparseMatrix, Ordering> SparseQRType;

  SparseQRMatrixQTransposeReturnTypeVisitor<SparseQRType>::expose(
      "SparseQRMatrixQTransposeReturnType");
  SparseQRMatrixQReturnTypeVisitor<SparseQRType>::expose(
      "SparseQRMatrixQReturnType");
  SparseQRVisitor<SparseQRType>::expose("SparseQR");
}
}  // namespace eigenpy
