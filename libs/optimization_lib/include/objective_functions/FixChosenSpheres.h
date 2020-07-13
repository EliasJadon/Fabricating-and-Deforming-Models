#pragma once
#include "libs/optimization_lib/include/objective_functions/ObjectiveFunction.h"

class FixChosenSpheres : public ObjectiveFunction
{
private:
	virtual void init_hessian() override;
public:
	FixChosenSpheres();
	~FixChosenSpheres();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
	
	std::vector<int> ConstrainedCentersInd;
	Eigen::MatrixX3d ConstrainedCentersPos;
	Eigen::MatrixX3d CurrConstrainedCentersPos;
	int numV=0;
	int numF=0;
};