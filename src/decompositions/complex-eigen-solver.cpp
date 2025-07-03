
/*
 * Copyright 2024 INRIA
 */

#include "eigenpy/decompositions/ComplexEigenSolver.hpp"

namespace eigenpy {
void exposeComplexEigenSolver() {
  using namespace Eigen;
  ComplexEigenSolverVisitor<MatrixXd>::expose("ComplexEigenSolver");
}
}  // namespace eigenpy
