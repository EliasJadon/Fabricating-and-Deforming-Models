#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/cuda_STVK.cuh"

class STVK : public ObjectiveFunction {
private:	
	void setRestShapeFromCurrentConfiguration();
public:
	std::shared_ptr<Cuda_STVK> cuda_STVK;
	STVK(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
	~STVK();
	virtual Cuda::Array<double>* getValue() override {
		return &(cuda_STVK->EnergyAtomic);
	}
	virtual void value(Cuda::Array<double>& curr_x);
	virtual Cuda::Array<double>* gradient(Cuda::Array<double>& X, const bool update) override;
};