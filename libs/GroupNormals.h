#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_Grouping.cuh"
#include <mutex>

class GroupNormals : public ObjectiveFunction
{
private:
	unsigned int startN_x, startN_y, startN_z;
	std::mutex m_value, m_gradient;
public:
	std::shared_ptr<Cuda_Grouping> cudaGrouping;
	GroupNormals(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixX3i& F);
	~GroupNormals();
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual Cuda::Array<double>* gradient(Cuda::Array<double>& X, const bool update) override;
	void updateExtConstraints(std::vector < std::vector<int>>& CInd);
	int numV=0;
	int numF=0;
};