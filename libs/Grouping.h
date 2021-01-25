#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_Grouping.cuh"
#include <mutex>

class Grouping : public ObjectiveFunction
{
private:
	std::vector < std::vector<int>> CInd;
	std::mutex m_value, m_gradient;
public:
	std::shared_ptr<Cuda_Grouping> cudaGrouping;
	Grouping(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F, const ConstraintsType type);
	~Grouping();
	virtual Cuda::Array<double>* getValue() override {
		return &(cudaGrouping->EnergyAtomic);
	}
	virtual Cuda::Array<double>* getGradient() override {
		return &(cudaGrouping->grad);
	}
	virtual void value(Cuda::Array<double>& curr_x) override;
	virtual void gradient(Cuda::Array<double>& X) override;
	void updateExtConstraints(std::vector < std::vector<int>>& CInd);
};