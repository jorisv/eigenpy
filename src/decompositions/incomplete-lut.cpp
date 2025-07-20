/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/sparse/IncompleteLUT.hpp"

namespace eigenpy {
void exposeIncompleteLUT() {
  using namespace Eigen;
  typedef SparseMatrix<double, ColMajor> ColMajorSparseMatrix;
  IncompleteLUTVisitor<ColMajorSparseMatrix>::expose("IncompleteLUT");
}
}  // namespace eigenpy
