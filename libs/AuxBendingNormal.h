#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_AuxBendingNormal.cuh"

class AuxBendingNormal : public ObjectiveFunction
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
	void pre_minimizer();
	std::shared_ptr<Cuda_AuxBendingNormal> cuda_ABN;
	void UpdateHingesWeights(const std::vector<int> faces, const double add);
	void ClearHingesWeights();
	AuxBendingNormal(
		const Eigen::MatrixXd& V, 
		const Eigen::MatrixX3i& F,
		const FunctionType type);
	~AuxBendingNormal();
	virtual Cuda::Array<double>* getValue() override {
		return &(cuda_ABN->EnergyAtomic);
	}
	virtual Cuda::Array<double>* getGradient() override {
		return &(cuda_ABN->grad);
	}
	virtual void value(Cuda::Array<double>& curr_x) override;
	virtual void gradient(Cuda::Array<double>& X) override;
};

