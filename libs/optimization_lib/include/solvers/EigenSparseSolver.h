#pragma once
#include "../OptimizationUtils.h"
#include <Eigen/Core>
#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>
#include <Eigen/SparseCholesky>
#include <igl/matlab_format.h>
#include <Eigen/PardisoSupport>

// TODO: Make this depend on a #define
// #include <igl/matlab/MatlabWorkspace.h>
// #include <igl/matlab/matlabinterface.h>

template <typename vectorTypeI, typename vectorTypeS>
class EigenSparseSolver
{
public:
	// TODO: Make this depend on a #define
	// Matlab instance
	// Engine *engine;
	bool CheckPositiveDefinite = true;

	EigenSparseSolver();
	~EigenSparseSolver();
	void set_pattern(const vectorTypeI &II, const vectorTypeI &JJ, const vectorTypeS &SS);
	void analyze_pattern();
	void factorize(const vectorTypeI &II, const vectorTypeI &JJ, const vectorTypeS &SS);
	void perpareMatrix(const vectorTypeI &II, const vectorTypeI &JJ, const vectorTypeS &SS);
	Eigen::VectorXd solve(Eigen::VectorXd &rhs);
    //Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;

	Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	//Eigen::PardisoLU<Eigen::SparseMatrix<double>> solver;
	//Eigen::ConjugateGradient<Eigen::SparseMatrix<double>, Eigen::Upper> solver;
	/*Eigen::SparseLU<Eigen::SparseMatrix<double>> solver;
	Eigen::SimplicialLLT<Eigen::SparseMatrix<double>> solver;
	Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;*/

    Eigen::SparseMatrix<double> full_A;
	double MSE;
};

