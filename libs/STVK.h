#pragma once
#include "ObjectiveFunction.h"

class STVK : public ObjectiveFunction {
private:	
	double shearModulus, bulkModulus;
	Eigen::VectorXd restShapeArea;
	std::vector<double4> dXInv;
	void setRestShapeFromCurrentConfiguration();
	void updateX(Cuda::Array<double>& curr_x);
public:
	Cuda::indices mesh_indices;
	Cuda::Array<double> grad;
	STVK(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
	~STVK();
	virtual double value(Cuda::Array<double>& curr_x, const bool update);
	virtual Cuda::Array<double>* gradient(Cuda::Array<double>& X, const bool update) override;
};