#pragma once
#include "ObjectiveFunction.h"
#include "cudaLibrary/cuda_fixAllVertices.cuh"
#include "cudaLibrary/Cuda_Minimizer.cuh"

class FixAllVertices : public ObjectiveFunction
{
private:
	virtual void init_hessian() override;
	Eigen::MatrixX3d CurrV;
	void internalInitCuda();
public:
	FixAllVertices();
	~FixAllVertices();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
};