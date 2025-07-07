/*
 * Copyright 2020-2024 INRIA
 */

#ifndef __eigenpy_decompositions_jacobisvd_hpp__
#define __eigenpy_decompositions_jacobisvd_hpp__

#include <Eigen/SVD>
#include <Eigen/Core>

#include "eigenpy/eigenpy.hpp"
#include "eigenpy/utils/scalar-name.hpp"
#include "eigenpy/eigen/EigenBase.hpp"
#include "eigenpy/decompositions/SVDBase.hpp"

namespace eigenpy {

template <typename _MatrixType>
struct JacobiSVDVisitor
    : public boost::python::def_visitor<JacobiSVDVisitor<_MatrixType>> {
  typedef _MatrixType MatrixType;
  typedef Eigen::JacobiSVD<MatrixType> Solver;
  typedef typename MatrixType::Scalar Scalar;

  template <class PyClass>
  void visit(PyClass &cl) const {
    cl.def(bp::init<>(bp::arg("self"), "Default constructor"))
        // TODO: Management of _Options, put default options for default
        // constructor
        .def(bp::init<Eigen::DenseIndex, Eigen::DenseIndex>(
            bp::args("self", "rows", "cols"),
            "Default Constructor with memory preallocation. "))
        .def(bp::init<Eigen::DenseIndex, Eigen::DenseIndex, unsigned int>(
            bp::args("self", "rows", "cols", "computationOptions "),
            "Default Constructor with memory preallocation. \n\n"
            "Like the default constructor but with preallocation of the "
            "internal "
            "data according to the specified problem size and the "
            "computationOptions. "))
        .def(bp::init<MatrixType>(
            bp::args("self", "matrix"),
            "Constructor performing the decomposition of given matrix. "))
        .def(bp::init<MatrixType, unsigned int>(
            bp::args("self", "matrix", "computationOptions "),
            "Constructor performing the decomposition of given matrix. \n\n"
            "One cannot request unitiaries using both the Options template "
            "parameter "
            "and the constructor. If possible, prefer using the Options "
            "template parameter."))

        .def("cols", &Solver::cols, bp::arg("self"),
             "Returns the number of columns. ")
        .def("compute",
             (Solver & (Solver::*)(const MatrixType &matrix)) & Solver::compute,
             bp::args("self", "matrix"),
             "Method performing the decomposition of given matrix. Computes "
             "Thin/Full "
             "unitaries U/V if specified using the Options template parameter "
             "or the class constructor. ",
             bp::return_self<>())
        .def("compute",
             (Solver & (Solver::*)(const MatrixType &matrix,
                                   unsigned int computationOptions)) &
                 Solver::compute,
             bp::args("self", "matrix"),
             "Method performing the decomposition of given matrix, as "
             "specified by the computationOptions parameter.  ",
             bp::return_self<>())
        .def("rows", &Solver::rows, bp::arg("self"),
             "Returns the number of rows. . ")

        .def(SVDBaseVisitor<Solver>());
  }

  static void expose() {
    static const std::string classname =
        "JacobiSVD_" + scalar_name<Scalar>::shortname();
    expose(classname);
  }

  static void expose(const std::string &name) {
    bp::class_<Solver, boost::noncopyable>(
        name.c_str(),
        "Two-sided Jacobi SVD decomposition of a rectangular matrix. \n\n"
        "SVD decomposition consists in decomposing any n-by-p matrix A as a "
        "product "
        "A=USV∗ where U is a n-by-n unitary, V is a p-by-p unitary, and S is a "
        "n-by-p r"
        "eal positive matrix which is zero outside of its main diagonal; the "
        "diagonal "
        "entries of S are known as the singular values of A and the columns of "
        "U and V "
        "are known as the left and right singular vectors of A respectively. "
        "\n\n"
        "Singular values are always sorted in decreasing order. \n\n "
        "This JacobiSVD decomposition computes only the singular values by "
        "default. "
        "If you want U or V, you need to ask for them explicitly. \n\n"
        "You can ask for only thin U or V to be computed, meaning the "
        "following. "
        "In case of a rectangular n-by-p matrix, letting m be the smaller "
        "value among "
        "n and p, there are only m singular vectors; the remaining columns of "
        "U and V "
        "do not correspond to actual singular vectors. Asking for thin U or V "
        "means asking "
        "for only their m first columns to be formed. So U is then a n-by-m "
        "matrix, and V "
        "is then a p-by-m matrix. Notice that thin U and V are all you need "
        "for (least squares) "
        "solving.",
        bp::no_init)
        .def(JacobiSVDVisitor())
        .def(IdVisitor<Solver>());
  }
};

}  // namespace eigenpy

#endif  // ifndef __eigenpy_decompositions_jacobisvd_hpp__
