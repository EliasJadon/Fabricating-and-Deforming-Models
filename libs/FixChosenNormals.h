#pragma once
#include "ObjectiveFunction.h"
#include <mutex>

class FixChosenNormals : public ObjectiveFunction
{
private:
	int startN_x;
	int startN_y;
	int startN_z;
	Eigen::MatrixX3d diff;
	std::mutex m;
	std::vector<int> ConstrainedNormalsInd;
	std::vector<int> currConstrainedNormalsInd;
	Eigen::MatrixX3d ConstrainedNormalsPos;
public:
	FixChosenNormals();
	~FixChosenNormals();
	virtual void init() override;
	virtual void updateX(Cuda::Array<double>& curr_x) override;
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual void gradient(Cuda::Array<double>& X,Eigen::VectorXd& g, const bool update) override;
	void updateExtConstraints(std::vector<int>& CNormalsInd, Eigen::MatrixX3d& CNormalsPos);
	int numV=0;
	int numF=0;
};