#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_FixChosenVertices.cuh"
#include <mutex>

class FixChosenVertices : public ObjectiveFunction
{
private:
	virtual void init_hessian() override;
	int startV_x;
	int startV_y;
	int startV_z;
	Eigen::MatrixX3d diff;
	std::mutex m_value, m_gradient;
	std::vector<int> ConstrainedVerticesInd;
	std::vector<int> currConstrainedVerticesInd;
	Eigen::MatrixX3d ConstrainedVerticesPos;
	void internalInitCuda();
public:
	FixChosenVertices();
	~FixChosenVertices();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
	void updateExtConstraints(std::vector<int>& CVInd, Eigen::MatrixX3d& CVPos);
	int numV=0;
	int numF=0;
};