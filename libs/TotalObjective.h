#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_AdamMinimizer.cuh"
#include "cuda_optimization_lib/Cuda_FixAllVertices.cuh"
#include "cuda_optimization_lib/Cuda_SSSymmetricDirichlet.cuh"
#include "cuda_optimization_lib/Cuda_Minimizer.cuh"

class TotalObjective : public ObjectiveFunction
{
private:
	virtual void init_hessian() override;
	int variables_size;
public:
	TotalObjective();
	~TotalObjective();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
	
	// sub objectives
	float Shift_eigen_values = 1000;
	std::vector<std::shared_ptr<ObjectiveFunction>> objectiveList;
};