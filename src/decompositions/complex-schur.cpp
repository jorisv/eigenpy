
/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/ComplexSchur.hpp"

namespace eigenpy {
void exposeComplexSchur() {
  using namespace Eigen;
  ComplexSchurVisitor<MatrixXd>::expose("ComplexSchur");
}
}  // namespace eigenpy
