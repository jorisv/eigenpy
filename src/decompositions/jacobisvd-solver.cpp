/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/JacobiSVD.hpp"

namespace eigenpy {
void exposeJacobiSVDSolver() {
  using namespace Eigen;
  JacobiSVDVisitor<MatrixXd>::expose("JacobiSVD");
}
}  // namespace eigenpy
