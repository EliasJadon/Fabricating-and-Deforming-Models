#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_Grouping.cuh"
#include <mutex>

class Grouping : public ObjectiveFunction
{
private:
	std::mutex m_value, m_gradient;
public:
	std::shared_ptr<Cuda_Grouping> cudaGrouping;
	Grouping(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F, const ConstraintsType type);
	~Grouping();
	virtual Cuda::Array<double>* getValue() override {
		return &(cudaGrouping->EnergyAtomic);
	}
	virtual void value(Cuda::Array<double>& curr_x) override;
	virtual Cuda::Array<double>* gradient(Cuda::Array<double>& X, const bool update) override;
	void updateExtConstraints(std::vector < std::vector<int>>& CInd);
};