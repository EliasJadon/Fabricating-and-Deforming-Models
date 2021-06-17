#pragma once
#include "ObjectiveFunction.h"

class fixRadius : public ObjectiveFunction {
public:
	int min = 2, max = 10;
	float alpha = 23;

	fixRadius(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
	~fixRadius();
	Cuda::Array<double> grad, EnergyAtomic;
	Cuda::indices mesh_indices;

	virtual Cuda::Array<double>* getValue() override {
		return &EnergyAtomic;
	}
	virtual Cuda::Array<double>* getGradient() override {
		return &grad;
	}
	virtual void value(Cuda::Array<double>& curr_x);
	virtual void gradient(Cuda::Array<double>& X);
};