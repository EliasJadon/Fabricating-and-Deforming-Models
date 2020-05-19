#pragma once

#include "libs/optimization_lib/include/solvers/solver.h"

class AdamMinimizer : public solver
{
private:
	Eigen::VectorXd v_adam;
	Eigen::VectorXd s_adam;
	double alpha_adam, beta1_adam, beta2_adam;

public:
	AdamMinimizer(const int solverID, 
		double alpha_adam = 1,
		double beta1_adam = 0.90,
		double beta2_adam = 0.9990) :
		solver(solverID),
		alpha_adam(alpha_adam),
		beta1_adam(beta1_adam),
		beta2_adam(beta2_adam) {}
	
		
	virtual void step() override {
		objective->updateX(X);
		currentEnergy = objective->value(true);
		objective->gradient(g, true);
				
		v_adam = beta1_adam * v_adam + (1 - beta1_adam) * g;
		// TODO one could avoid casting if function->minimize(..) would be generic -> see Eigen Docs
		Eigen::VectorXd tmp = g.array().square(); // just for casting
		s_adam = beta2_adam * s_adam + (1 - beta2_adam) * tmp; // element-wise square
		
		// bias correction - note, how much do we trust first gradients?
		// v = v / (1-std::pow(beta1,t));
		// s = s / (1-std::pow(beta2,t));
		tmp = s_adam.array().sqrt() + 1e-8; // just for casting
		p = -v_adam.cwiseQuotient(tmp);
	}

	virtual bool test_progress() override {
		return true;
	}

	virtual void internal_init() override {
		objective->updateX(X);
		g.resize(X.size());
		v_adam = Eigen::VectorXd::Zero(X.size());
		s_adam = Eigen::VectorXd::Zero(X.size());
	}
};

