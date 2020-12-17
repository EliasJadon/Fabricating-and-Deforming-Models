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
	FixAllVertices();
	~FixAllVertices();
	virtual void init() override;
	virtual void updateX(Cuda::Array<double>& curr_x) override;
	virtual double value(Cuda::Array<double>& curr_x,const bool update) override;
	virtual void gradient(Cuda::Array<double>& curr_x,Eigen::VectorXd& g, const bool update) override;
};