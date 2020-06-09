#include "solvers/EigenSparseSolver.h"
#include "plugins/deformation_plugin/include/console_color.h"
#include <vector>
#include <iostream>

template <typename vectorTypeI, typename vectorTypeS>
EigenSparseSolver<vectorTypeI, vectorTypeS>::EigenSparseSolver()
{
}

template <typename vectorTypeI, typename vectorTypeS>
EigenSparseSolver<vectorTypeI, vectorTypeS>::~EigenSparseSolver()
{
}

template <typename vectorTypeI, typename vectorTypeS>
void EigenSparseSolver<vectorTypeI, vectorTypeS>::set_pattern(const vectorTypeI &II, const vectorTypeI &JJ, const vectorTypeS &SS)
{
	perpareMatrix(II, JJ, SS);
}

template <typename vectorTypeI, typename vectorTypeS>
void EigenSparseSolver<vectorTypeI, vectorTypeS>::analyze_pattern()
{
	solver.analyzePattern(full_A);
	//assert(solver.info() == Eigen::Success && "analyzePattern failed!");
}

template <typename vectorTypeI, typename vectorTypeS>
void EigenSparseSolver<vectorTypeI, vectorTypeS>::factorize(const vectorTypeI &II, const vectorTypeI &JJ, const vectorTypeS &SS)
{
	perpareMatrix(II,JJ,SS);

	solver.factorize(full_A);
	//// Launch MATLAB
	//igl::matlab::mlinit(&engine);
	//igl::matlab::mleval(&engine, "desktop");

	//// Send matrix to matlab
	//igl::matlab::mlsetmatrix(&engine, "a", full_A);
	
	assert(solver.info() == Eigen::Success && "factorization failed!");
}

template <typename vectorTypeI, typename vectorTypeS>
Eigen::VectorXd EigenSparseSolver<vectorTypeI, vectorTypeS>::solve(Eigen::VectorXd &rhs)
{
	Eigen::VectorXd x;
	x = solver.solve(rhs);
	MSE = (full_A * x - rhs).cwiseAbs2().sum();
	return x;
}

template <typename vectorTypeI, typename vectorTypeS>
void EigenSparseSolver<vectorTypeI, vectorTypeS>::perpareMatrix(const vectorTypeI &II, const vectorTypeI &JJ, const vectorTypeS &SS)
{
	double epsilon = 1e-2;
	Eigen::SparseMatrix<double> UpperTriangular_A = OptimizationUtils::BuildMatrix(II,JJ,SS);
	full_A = UpperTriangular_A.selfadjointView<Eigen::Upper>();

	if (CheckPositiveDefinite) {
		double min_eig_value = full_A.toDense().eigenvalues().real().minCoeff();
		if (min_eig_value < epsilon)
			std::cout << console_color::red;
		else
			std::cout << console_color::green;
		std::cout << "before: min_eig_value = " << min_eig_value << std::endl;
		if (min_eig_value < epsilon) {
			for (int i = 0; i < full_A.rows(); i++) {
				full_A.coeffRef(i, i) = full_A.coeff(i, i) + (-min_eig_value + epsilon);
			}
		}
		std::cout << console_color::white;
		std::cout << "after: full min_eig_value = " << full_A.toDense().eigenvalues().real().minCoeff() << std::endl;
	}
	
	full_A.makeCompressed();
}

template class EigenSparseSolver<std::vector<int, std::allocator<int> >, std::vector<double, std::allocator<double> > >;
