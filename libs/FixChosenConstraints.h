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
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual Cuda::Array<double>* gradient(Cuda::Array<double>& X, const bool update) override;
	void updateExtConstraints(std::vector<int>& CVInd, Eigen::MatrixX3d& CVPos);
};