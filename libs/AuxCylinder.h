#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_AuxCylinder.cuh"

class AuxCylinder : public ObjectiveFunction
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
	std::shared_ptr<Cuda_AuxCylinder> cuda_ACY;
	AuxCylinder(
		const Eigen::MatrixXd& V, 
		const Eigen::MatrixX3i& F,
		const FunctionType type);
	~AuxCylinder();
	virtual Cuda::Array<double>* getValue() override {
		return &(cuda_ACY->EnergyAtomic);
	}
	virtual Cuda::Array<double>* getGradient() override {
		return &(cuda_ACY->grad);
	}
	virtual void value(Cuda::Array<double>& curr_x) override;
	virtual void gradient(Cuda::Array<double>& X) override;
};

