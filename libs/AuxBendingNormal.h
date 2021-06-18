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
	Eigen::Vector3f colorP, colorM;
	std::shared_ptr<Cuda_AuxBendingNormal> cuda_ABN;

	void Reset_HingesSigmoid(const std::vector<int> faces_indices);
	void Incr_HingesWeights(const std::vector<int> faces_indices, const double add);
	void Set_HingesWeights(const std::vector<int> faces_indices, const double value);
	void Update_HingesSigmoid(const std::vector<int> faces_indices, const double factor);
	void Clear_HingesWeights();
	void Clear_HingesSigmoid();

	AuxBendingNormal(
		const Eigen::MatrixXd& V, 
		const Eigen::MatrixX3i& F,
		const PenaltyFunction type);
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

