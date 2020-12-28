#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/cuda_fixAllVertices.cuh"
#include "cuda_optimization_lib/Cuda_Minimizer.cuh"

class FixAllVertices : public ObjectiveFunction
{
private:
	Eigen::MatrixX3d CurrV;
	void internalInitCuda();
public:
	std::shared_ptr<Cuda_FixAllVertices> cuda_FixAllV;
	FixAllVertices(const Eigen::MatrixXd& V,
		const Eigen::MatrixX3i& F);
	~FixAllVertices();
	virtual Cuda::Array<double>* getValue() override {
		return &(cuda_FixAllV->EnergyAtomic);
	}
	virtual void value(Cuda::Array<double>& curr_x) override;
	virtual Cuda::Array<double>* gradient(Cuda::Array<double>& X, const bool update) override;
};