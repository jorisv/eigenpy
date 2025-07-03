
/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/HessenbergDecomposition.hpp"

namespace eigenpy {
void exposeHessenbergDecomposition() {
  using namespace Eigen;
  HessenbergDecompositionVisitor<MatrixXd>::expose("HessenbergDecomposition");
}
}  // namespace eigenpy
