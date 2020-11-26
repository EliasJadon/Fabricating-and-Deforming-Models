#pragma once

#include "Minimizer.h"
#include "EigenSparseLinearEquationSolver.h"

class NewtonMinimizer : public Minimizer
{
public:
	NewtonMinimizer(const int solverID): Minimizer(solverID) {}

	virtual void step() override {
		/*objective->updateX(X);
		currentEnergy = objective->value(true);
		objective->gradient(g, true);
		objective->hessian();
		linear_solver->factorize(objective->II, objective->JJ, objective->SS);
		Eigen::VectorXd rhs = -g;
		p = linear_solver->solve(rhs);*/
	}
	virtual void internal_init() override {
		////pardisoSolver p;
		//bool needs_init = linear_solver == nullptr;
		//if (needs_init) {
		//	linear_solver = std::make_unique<EigenSparseLinearEquationSolver<std::vector<int>, std::vector<double>>>();
		//}
		//objective->updateX(X);
		//g.resize(X.size());
		//objective->hessian();
		//if (needs_init) {
		//	linear_solver->set_pattern(objective->II, objective->JJ, objective->SS);
		//	linear_solver->analyze_pattern();
		//}
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
	std::unique_ptr< EigenSparseLinearEquationSolver<std::vector<int>, std::vector<double>>> linear_solver = nullptr;
	//std::unique_ptr<PardisoLinearEquationSolver<std::vector<int>, std::vector<double>>> linear_solver = nullptr;
};