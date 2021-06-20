#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/cuda_STVK.cuh"

class STVK : public ObjectiveFunction {
private:	
	void setRestShapeFromCurrentConfiguration();
public:
	Eigen::VectorXd restShapeArea;
	Cuda::Array<double4> dXInv;
	double shearModulus, bulkModulus;

	STVK(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
	~STVK();
	
	virtual double value(Cuda::Array<double>& curr_x, const bool update);
	virtual void gradient(Cuda::Array<double>& X, const bool update) override;
};