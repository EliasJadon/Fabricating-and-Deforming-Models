#pragma once
#include "ObjectiveFunction.h"
#include <mutex>

class FixChosenSpheres : public ObjectiveFunction
{
private:
	int startC_x;
	int startC_y;
	int startC_z;
	Eigen::MatrixX3d diff;
	std::mutex m;
	std::vector<int> ConstrainedCentersInd;
	std::vector<int> currConstrainedCentersInd;
	Eigen::MatrixX3d ConstrainedCentersPos;
public:
	FixChosenSpheres();
	~FixChosenSpheres();
	virtual void init() override;
	virtual void updateX(Cuda::Array<double>& curr_x) override;
	virtual double value(Cuda::Array<double>& curr_x,const bool update) override;
	virtual void gradient(Cuda::Array<double>& X,Eigen::VectorXd& g, const bool update) override;
	void updateExtConstraints(std::vector<int>& CCentersInd, Eigen::MatrixX3d& CCentersPos);
	int numV=0;
	int numF=0;
};