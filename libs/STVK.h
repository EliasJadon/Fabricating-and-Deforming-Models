#pragma once
#include "ObjectiveFunction.h"

class STVK : public ObjectiveFunction {
private:	
	double shearModulus, bulkModulus;
	Eigen::VectorXd restShapeArea;
	Eigen::MatrixX3d CurrV;
	std::vector<Eigen::Matrix2d> dXInv, strain;
	std::vector<Eigen::Matrix<double, 3, 2>> F;
	void setRestShapeFromCurrentConfiguration();
public:
	STVK();
	~STVK();
	virtual void init();
	virtual void updateX(Cuda::Array<double>& curr_x);
	virtual double value(Cuda::Array<double>& curr_x, const bool update);
	virtual Cuda::Array<double>* gradient(Cuda::Array<double>& X, const bool update) override;
};