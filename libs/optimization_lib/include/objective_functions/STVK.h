#pragma once
#include "libs/optimization_lib/include/objective_functions/ObjectiveFunction.h"

class STVK : public ObjectiveFunction {
private:	
	double shearModulus, bulkModulus;
	Eigen::VectorXd restShapeArea;
	Eigen::MatrixX3d CurrV;
	std::vector<Eigen::Matrix2d> dXInv, strain;
	std::vector<Eigen::Matrix<double, 3, 2>> F;
	void setRestShapeFromCurrentConfiguration();
	virtual void init_hessian();
public:
	STVK();
	~STVK();
	virtual void init();
	virtual void updateX(const Eigen::VectorXd& X);
	virtual double value(const bool update);
	virtual void gradient(Eigen::VectorXd& g, const bool update);
	virtual void hessian();
};