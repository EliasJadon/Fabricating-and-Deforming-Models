#pragma once

#include "Minimizer.h"

class AdamMinimizer : public Minimizer
{
private:
#ifndef USING_CUDA
	Eigen::VectorXd v_adam;
	Eigen::VectorXd s_adam;
#endif
	double alpha_adam, beta1_adam, beta2_adam;
public:
	AdamMinimizer(const int solverID, 
		double alpha_adam = 1,
		double beta1_adam = 0.90,
		double beta2_adam = 0.9990) :
		Minimizer(solverID),
		alpha_adam(alpha_adam),
		beta1_adam(beta1_adam),
		beta2_adam(beta2_adam) 
	{}	
	virtual void step() override {
#ifdef USING_CUDA
		Eigen::VectorXd _ = Eigen::VectorXd::Zero(1);
		Cuda::copyArrays(Cuda::Minimizer::curr_x, Cuda::Minimizer::X);
		objective->updateX(_);
		currentEnergy = objective->value(true);
		///////////objective->gradient(_, true);
		Cuda::AuxBendingNormal::gradient();

		Cuda::copyArrays(Cuda::Minimizer::g, Cuda::AuxBendingNormal::grad);
		Cuda::AdamMinimizer::step(alpha_adam, beta1_adam, beta2_adam);
#else
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
#endif
	}
	

	virtual void internal_init() override {
#ifdef USING_CUDA
		unsigned int size = 3 * V.rows() + 7 * F.rows();
		Cuda::AllocateMemory(Cuda::AdamMinimizer::v_adam, size);
		Cuda::AllocateMemory(Cuda::AdamMinimizer::s_adam, size);
		for (int i = 0; i < size; i++) {
			Cuda::AdamMinimizer::v_adam.host_arr[i] = 0;
			Cuda::AdamMinimizer::s_adam.host_arr[i] = 0;
		}
		Cuda::MemCpyHostToDevice(Cuda::AdamMinimizer::v_adam);
		Cuda::MemCpyHostToDevice(Cuda::AdamMinimizer::s_adam);
#else
		objective->updateX(X);
		g.resize(X.size());
		v_adam = Eigen::VectorXd::Zero(X.size());
		s_adam = Eigen::VectorXd::Zero(X.size());
#endif
	}
};

