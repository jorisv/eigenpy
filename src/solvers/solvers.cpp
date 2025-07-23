/*
 * Copyright 2017-2025 CNRS INRIA
 */

#include <Eigen/Core>

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)

#include "eigenpy/solvers/ConjugateGradient.hpp"
#include "eigenpy/solvers/solvers.hpp"

#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
#include "eigenpy/solvers/LeastSquaresConjugateGradient.hpp"
#endif

#include "eigenpy/solvers/BiCGSTAB.hpp"
#include "eigenpy/solvers/MINRES.hpp"

#include "eigenpy/solvers/IncompleteLUT.hpp"
#include "eigenpy/solvers/IncompleteCholesky.hpp"

namespace eigenpy {
void exposeSolvers() {
  using namespace Eigen;

  using Eigen::Lower;

  using Eigen::BiCGSTAB;
  using Eigen::ConjugateGradient;
  using Eigen::LeastSquaresConjugateGradient;

  using Eigen::DiagonalPreconditioner;
  using Eigen::IdentityPreconditioner;
  using Eigen::LeastSquareDiagonalPreconditioner;

  using IdentityBiCGSTAB = BiCGSTAB<MatrixXd, IdentityPreconditioner>;
  using IdentityConjugateGradient =
      ConjugateGradient<MatrixXd, Lower, IdentityPreconditioner>;
  using IdentityLeastSquaresConjugateGradient =
      LeastSquaresConjugateGradient<MatrixXd, IdentityPreconditioner>;
  using DiagonalLeastSquaresConjugateGradient = LeastSquaresConjugateGradient<
      MatrixXd, DiagonalPreconditioner<typename MatrixXd::Scalar>>;

  ConjugateGradientVisitor<ConjugateGradient<MatrixXd, Lower>>::expose(
      "ConjugateGradient");
  ConjugateGradientVisitor<IdentityConjugateGradient>::expose(
      "IdentityConjugateGradient");

#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
  LeastSquaresConjugateGradientVisitor<LeastSquaresConjugateGradient<
      MatrixXd, LeastSquareDiagonalPreconditioner<MatrixXd::Scalar>>>::
      expose("LeastSquaresConjugateGradient");
  LeastSquaresConjugateGradientVisitor<IdentityLeastSquaresConjugateGradient>::
      expose("IdentityLeastSquaresConjugateGradient");
  LeastSquaresConjugateGradientVisitor<DiagonalLeastSquaresConjugateGradient>::
      expose("DiagonalLeastSquaresConjugateGradient");
#endif

  BiCGSTABVisitor<BiCGSTAB<MatrixXd>>::expose("BiCGSTAB");
  BiCGSTABVisitor<IdentityBiCGSTAB>::expose("IdentityBiCGSTAB");

  MINRESSolverVisitor<MatrixXd>::expose("MINRES");

  typedef SparseMatrix<double, ColMajor> ColMajorSparseMatrix;
  IncompleteLUTVisitor<ColMajorSparseMatrix>::expose("IncompleteLUT");
  IncompleteCholeskyVisitor<ColMajorSparseMatrix>::expose("IncompleteCholesky");
}
}  // namespace eigenpy

#endif
