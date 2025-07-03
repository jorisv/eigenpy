
/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/RealQZ.hpp"

namespace eigenpy {
void exposeRealQZ() {
  using namespace Eigen;
  RealQZVisitor<MatrixXd>::expose("RealQZ");
}
}  // namespace eigenpy
