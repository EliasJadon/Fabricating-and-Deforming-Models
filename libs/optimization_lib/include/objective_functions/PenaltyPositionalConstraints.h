#pragma once
#include "libs/optimization_lib/include/objective_functions/ObjectiveFunction.h"

class PenaltyPositionalConstraints : public ObjectiveFunction
{
private:
	virtual void init_hessian() override;
public:
	PenaltyPositionalConstraints();
	~PenaltyPositionalConstraints();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
	
	std::vector<int> ConstrainedVerticesInd;
	Eigen::MatrixX3d ConstrainedVerticesPos;
	Eigen::MatrixX3d CurrConstrainedVerticesPos;
	int numV=0;
	int numF=0;
};