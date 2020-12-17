#pragma once
#include "Minimizer.h"
#include "cuda_optimization_lib/Cuda_AdamMinimizer.cuh"
#include "cuda_optimization_lib/Cuda_AuxBendingNormal.cuh"

class AdamMinimizer : public Minimizer
{
private:
#ifndef USING_CUDA
	Eigen::VectorXd v_adam;
	Eigen::VectorXd s_adam;
#endif

public:
	AdamMinimizer(const int solverID) : Minimizer(solverID){}

	virtual void step() override {
#ifdef USING_CUDA
		Cuda::Minimizer::AdamStep();
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
		Cuda::AllocateMemory(Cuda::Minimizer::v_adam, size);
		Cuda::AllocateMemory(Cuda::Minimizer::s_adam, size);
		for (int i = 0; i < size; i++) {
			Cuda::Minimizer::v_adam.host_arr[i] = 0;
			Cuda::Minimizer::s_adam.host_arr[i] = 0;
		}
		Cuda::MemCpyHostToDevice(Cuda::Minimizer::v_adam);
		Cuda::MemCpyHostToDevice(Cuda::Minimizer::s_adam);
#else
		objective->updateX(X);
		g.resize(X.size());
		v_adam = Eigen::VectorXd::Zero(X.size());
		s_adam = Eigen::VectorXd::Zero(X.size());
#endif
	}
};

