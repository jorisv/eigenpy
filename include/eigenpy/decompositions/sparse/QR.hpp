/*
 * Copyright 2024 INRIA
 */

#ifndef __eigenpy_decompositions_sparse_qr_hpp__
#define __eigenpy_decompositions_sparse_qr_hpp__

#include <Eigen/SparseQR>
#include <Eigen/Core>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/decompositions/sparse/SparseSolverBase.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType,
          typename _Ordering = Eigen::AMDOrdering<
              typename _MatrixType::StorageIndex>>
struct SparseQRVisitor : public boost::python::def_visitor<
                             SparseQRVisitor<_MatrixType, _Ordering>> {
  typedef SparseQRVisitor<_MatrixType, _Ordering> Visitor;
  typedef _MatrixType MatrixType;
  typedef _Ordering Ordering;

  typedef Eigen::SparseQR<MatrixType, Ordering> Solver;
  typedef typename MatrixType::Scalar Scalar;
  typedef typename MatrixType::RealScalar RealScalar;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1, MatrixType::Options>
      DenseVectorXs;
  typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic,
                        MatrixType::Options>
      DenseMatrixXs;

  typedef Eigen::SparseQRMatrixQReturnType<Solver> MatrixQType;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        .def(bp::init<MatrixType>(
            bp::args("self", "mat"),
            "Construct a QR factorization of the matrix mat."))

        .def("cols", &Solver::cols, bp::arg("self"),
             "Returns the number of columns of the represented matrix. ")
        .def("rows", &Solver::rows, bp::arg("self"),
             "Returns the number of rows of the represented matrix. ")

        .def("compute", &Solver::compute, bp::args("self", "matrix"),
             "Compute the symbolic and numeric factorization of the input "
             "sparse matrix. "
             "The input matrix should be in column-major storage. ")
        .def("analyzePattern", &Solver::analyzePattern, bp::args("self", "mat"),
             "Compute the column permutation to minimize the fill-in.")
        .def("factorize", &Solver::factorize, bp::args("self", "matrix"),
             "Performs a numeric decomposition of a given matrix.\n"
             "The given matrix must has the same sparcity than the matrix on "
             "which the symbolic decomposition has been performed.")

        // TODO: Expose so that the return type are convertible to np arrays
        // matrixQ
        // matrixR

        .def("colsPermutation", &Solver::colsPermutation, bp::arg("self"),
             "Returns a reference to the column matrix permutation PTc such "
             "that Pr A PTc = LU.",
             bp::return_value_policy<bp::copy_const_reference>())

        .def("info", &Solver::info, bp::arg("self"),
             "NumericalIssue if the input contains INF or NaN values or "
             "overflow occured. Returns Success otherwise.")
        .def("lastErrorMessage", &Solver::lastErrorMessage, bp::arg("self"),
             "Returns a string describing the type of error. ")

        .def("rank", &Solver::rank, bp::arg("self"),
             "Returns the number of non linearly dependent columns as "
             "determined "
             "by the pivoting threshold. ")

        .def("setPivotThreshold", &Solver::setPivotThreshold,
             bp::args("self", "thresh"),
             "Set the threshold used for a diagonal entry to be an acceptable "
             "pivot.")

        .def(SparseSolverBaseVisitor<Solver>());
  }

  static void expose() {
    static const std::string classname =
        "SparseQR_" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver, boost::noncopyable>(
        name.c_str(),
        "Sparse left-looking QR factorization with numerical column pivoting. "
        "This class implements a left-looking QR decomposition of sparse "
        "matrices "
        "with numerical column pivoting. When a column has a norm less than a "
        "given "
        "tolerance it is implicitly permuted to the end. The QR factorization "
        "thus "
        "obtained is given by A*P = Q*R where R is upper triangular or "
        "trapezoidal. \n\n"
        "P is the column permutation which is the product of the fill-reducing "
        "and the "
        "numerical permutations. \n\n"
        "Q is the orthogonal matrix represented as products of Householder "
        "reflectors. \n\n"
        "R is the sparse triangular or trapezoidal matrix. The later occurs "
        "when A is rank-deficient. \n\n",
        bp::no_init)
        .def(SparseQRVisitor())
        .def(IdVisitor<Solver>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_sparse_qr_hpp__
