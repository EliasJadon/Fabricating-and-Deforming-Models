#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/cuda_SDenergy.cuh"

class SDenergy : public ObjectiveFunction {
public:
	std::shared_ptr<Cuda_SDenergy> cuda_SD;
	SDenergy(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
	~SDenergy();
	
	virtual Cuda::Array<double>* getValue() override {
		return &(cuda_SD->EnergyAtomic);
	}
	virtual Cuda::Array<double>* getGradient() override {
		return &(cuda_SD->grad);
	}
	virtual void value(Cuda::Array<double>& curr_x);
	virtual void gradient(Cuda::Array<double>& X);
};