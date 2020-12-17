#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_FixChosenConstraints.cuh"
#include <mutex>

class FixChosenVertices : public ObjectiveFunction
{
private:
	int startV_x;
	int startV_y;
	int startV_z;
	Eigen::MatrixX3d diff;
	std::mutex m_value, m_gradient;
	std::vector<int> ConstrainedVerticesInd;
	std::vector<int> currConstrainedVerticesInd;
	Eigen::MatrixX3d ConstrainedVerticesPos;
public:
	std::shared_ptr<Cuda_FixChosenConstraints> Cuda_FixChosConst;
	FixChosenVertices(const unsigned int numF,const unsigned int numV);
	~FixChosenVertices();
	virtual void init() override;
	virtual void updateX(Cuda::Array<double>& curr_x) override;
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual void gradient(Cuda::Array<double>& X, Eigen::VectorXd& g, const bool update) override;
	void updateExtConstraints(std::vector<int>& CVInd, Eigen::MatrixX3d& CVPos);
	int numV=0;
	int numF=0;
};