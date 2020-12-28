#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_FixChosenConstraints.cuh"
#include <mutex>

class FixChosenConstraints : public ObjectiveFunction
{
private:
	std::mutex m_value, m_gradient;
public:
	std::shared_ptr<Cuda_FixChosenConstraints> Cuda_FixChosConst;
	FixChosenConstraints(
		const unsigned int numF,
		const unsigned int numV,
		const ConstraintsType type);
	~FixChosenConstraints();
	virtual Cuda::Array<double>* getValue() override {
		return &(Cuda_FixChosConst->EnergyAtomic);
	}
	virtual Cuda::Array<double>* getGradient() override {
		return &(Cuda_FixChosConst->grad);
	}
	virtual void value(Cuda::Array<double>& curr_x) override;
	virtual void gradient(Cuda::Array<double>& X) override;
	void updateExtConstraints(std::vector<int>& CVInd, Eigen::MatrixX3d& CVPos);
};