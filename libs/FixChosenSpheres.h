#pragma once
#include "ObjectiveFunction.h"
#include <mutex>

class FixChosenSpheres : public ObjectiveFunction
{
private:
	virtual void init_hessian() override;
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
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
	void updateExtConstraints(std::vector<int>& CCentersInd, Eigen::MatrixX3d& CCentersPos);
	int numV=0;
	int numF=0;
};