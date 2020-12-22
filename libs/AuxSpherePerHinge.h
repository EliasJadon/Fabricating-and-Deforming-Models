#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_AuxSpherePerHinge.cuh"

class AuxSpherePerHinge : public ObjectiveFunction
{	
private:
	Eigen::VectorXd restAreaPerFace, restAreaPerHinge;
	int num_hinges = -1;
	std::vector<Eigen::Vector2d> hinges_faceIndex;
	Eigen::VectorXi x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd;
	Eigen::MatrixXi x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd;

	void calculateHinges();	
	void internalInitCuda();
public:
	AuxSpherePerHinge(FunctionType type);
	~AuxSpherePerHinge();
	virtual void init() override;
	virtual void updateX(Cuda::Array<double>& curr_x) override;
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual void gradient(Cuda::Array<double>& X, Eigen::VectorXd& g, const bool update) override;
};

