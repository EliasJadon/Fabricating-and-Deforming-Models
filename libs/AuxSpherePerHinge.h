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
	std::shared_ptr<Cuda_AuxSpherePerHinge> cuda_ASH;
	AuxSpherePerHinge(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixX3i& F,
		const FunctionType type);
	~AuxSpherePerHinge();
	virtual Cuda::Array<double>* getValue() override {
		return &(cuda_ASH->EnergyAtomic);
	}
	virtual void value(Cuda::Array<double>& curr_x) override;
	virtual Cuda::Array<double>* gradient(Cuda::Array<double>& X, const bool update) override;
};

