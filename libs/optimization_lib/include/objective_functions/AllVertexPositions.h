#pragma once
#include "libs/optimization_lib/include/objective_functions/ObjectiveFunction.h"

class AllVertexPositions : public ObjectiveFunction
{
private:
	virtual void init_hessian() override;
	Eigen::MatrixX3d CurrV;
public:
	AllVertexPositions();
	~AllVertexPositions();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
};