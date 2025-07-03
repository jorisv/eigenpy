/*
 * Copyright 2020 INRIA
 */

#ifndef __eigenpy_decompositions_tridiagonalization_hpp__
#define __eigenpy_decompositions_tridiagonalization_hpp__

#include <Eigen/Core>
#include <Eigen/Eigenvalues>

#include "eigenpy/eigen-to-python.hpp"
#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct TridiagonalizationVisitor
    : public boost::python::def_visitor<TridiagonalizationVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef typename MatrixType::Scalar Scalar;
  typedef Eigen::Tridiagonalization<MatrixType> Solver;

  template <class PyClass>
  void visit(PyClass& cl) const {
    cl.def(
          bp::init<Eigen::DenseIndex>(bp::arg("size"), "Default constructor. "))
        .def(bp::init<MatrixType>(
            bp::arg("matrix"),
            "Constructor; computes tridiagonal decomposition of given matrix. "))

        .def("compute", &TridiagonalizationVisitor::compute_proxy<MatrixType>,
             bp::args("self", "matrix"),
             "Computes tridiagonal decomposition of given matrix. ",
             bp::return_self<>())
        .def("compute",
             (Solver &
              (Solver::*)(const Eigen::EigenBase<MatrixType>& matrix)) &
                 Solver::compute,
             bp::args("self", "matrix"),
             "Computes tridiagonal decomposition of given matrix. ", bp::return_self<>())

        .def("diagonal", &Solver::diagonal, bp::arg("self"),
             "Returns the diagonal of the tridiagonal matrix T in the decomposition. ")

        .def("householderCoefficients", &Solver::householderCoefficients,
             bp::arg("self"), "Returns the Householder coefficients. ")

        .def("matrixQ", &Solver::matrixQ,
             bp::arg("self"), "Returns the unitary matrix Q in the decomposition. ")
        .def("matrixT", &Solver::matrixT,
             bp::arg("self"), "Returns the unitary matrix T in the decomposition. ")

        .def("packedMatrix", &Solver::packedMatrix, bp::arg("self"),
             "Returns the internal representation of the decomposition. ",
             bp::return_value_policy<bp::copy_const_reference>())

        .def("subDiagonal", &Solver::subDiagonal, bp::arg("self"),
             "Returns the subdiagonal of the tridiagonal matrix T in the decomposition.");
  }

  static void expose() {
    static const std::string classname =
        "TridiagonalizationVisitor" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string& name) {
    bp::class_<Solver>(name.c_str(), bp::no_init)
        .def(TridiagonalizationVisitor())
        .def(IdVisitor<Solver>());
  }

 private:
  template <typename MatrixType>
  static Solver& compute_proxy(Solver& self, const Eigen::EigenBase<MatrixType>& matrix) {
    return self.compute(matrix);
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_tridiagonalization_hpp__
