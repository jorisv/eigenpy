
/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/RealSchur.hpp"

namespace eigenpy {
void exposeRealSchur() {
  using namespace Eigen;
  RealSchurVisitor<MatrixXd>::expose("RealSchur");
}
}  // namespace eigenpy
