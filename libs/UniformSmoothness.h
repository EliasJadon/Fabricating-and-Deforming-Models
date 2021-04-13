#pragma once
#include "ObjectiveFunction.h"

class UniformSmoothness : public ObjectiveFunction {
public:
	UniformSmoothness(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
	~UniformSmoothness();
	Cuda::Array<double> grad, EnergyAtomic;
	Cuda::indices mesh_indices;
	Eigen::SparseMatrix<double> L;

	virtual Cuda::Array<double>* getValue() override {
		return &EnergyAtomic;
	}
	virtual Cuda::Array<double>* getGradient() override {
		return &grad;
	}
	virtual void value(Cuda::Array<double>& curr_x);
	virtual void gradient(Cuda::Array<double>& X);
};