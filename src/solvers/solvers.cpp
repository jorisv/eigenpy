/*
 * Copyright 2017-2020 CNRS INRIA
 */

#include <Eigen/Core>

#if EIGEN_VERSION_AT_LEAST(3, 2, 0)

#include "eigenpy/solvers/ConjugateGradient.hpp"
#include "eigenpy/solvers/solvers.hpp"

#if EIGEN_VERSION_AT_LEAST(3, 3, 5)
#include "eigenpy/solvers/LeastSquaresConjugateGradient.hpp"
#endif

#include "eigenpy/solvers/BiCGSTAB.hpp"

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
}
}  // namespace eigenpy

#endif
