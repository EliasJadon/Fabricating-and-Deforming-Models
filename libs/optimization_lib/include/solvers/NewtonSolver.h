#pragma once

#include "libs/optimization_lib/include/solvers/solver.h"
#include "libs/optimization_lib/include/solvers/EigenSparseSolver.h"
//#include "libs/optimization_lib/include/solvers/PardisoSolver.h"

class NewtonSolver : public solver
{
public:
	NewtonSolver(const int solverID): solver(solverID) {}

	virtual void step() override {
		objective->updateX(X);
		currentEnergy = objective->value(true);
		objective->gradient(g, true);
		objective->hessian();
		linear_solver->factorize(objective->II, objective->JJ, objective->SS);
		Eigen::VectorXd rhs = -g;
		p = linear_solver->solve(rhs);
	}

	virtual bool test_progress() override {
		return true;
	}

	virtual void internal_init() override {
		bool needs_init = linear_solver == nullptr;
		if (needs_init) {
			linear_solver = std::make_unique<EigenSparseSolver<std::vector<int>, std::vector<double>>>();
		}
		objective->updateX(X);
		g.resize(X.size());
		objective->hessian();
		if (needs_init) {
			linear_solver->set_pattern(objective->II, objective->JJ, objective->SS);
			linear_solver->analyze_pattern();
		}
	}
	
	Eigen::SparseMatrix<double> get_Hessian() {
		return linear_solver->full_A;
	}

	double get_MSE() {
		return linear_solver->MSE;
	}

	bool getPositiveDefiniteChecker() {
		return linear_solver->CheckPositiveDefinite;
	}

	void SwitchPositiveDefiniteChecker(const bool PD) {
		linear_solver->CheckPositiveDefinite = PD;
	}
	
private:
	std::unique_ptr< EigenSparseSolver<std::vector<int>, std::vector<double>>> linear_solver = nullptr;
	//std::unique_ptr<PardisoSolver<std::vector<int>, std::vector<double>>> linear_solver = nullptr;
};