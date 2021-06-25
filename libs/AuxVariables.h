#pragma once
#include "ObjectiveFunction.h"

class AuxVariables : public ObjectiveFunction
{	
private:
	double SigmoidParameter;
public:
	Cuda::Array<double> weight_PerHinge, Sigmoid_PerHinge;
	Cuda::PenaltyFunction penaltyFunction;
	Eigen::VectorXd restAreaPerFace, restAreaPerHinge;
	int num_hinges = -1;
	Eigen::VectorXi x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd;
	Eigen::MatrixXi x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd;
	std::vector<Eigen::Vector2d> hinges_faceIndex;
	Eigen::Vector3f colorP, colorM;

	void Inc_SigmoidParameter();
	void Dec_SigmoidParameter(const double target = 0);
	double get_SigmoidParameter();
	void calculateHinges();	
	void Reset_HingesSigmoid(const std::vector<int> faces_indices);
	void Incr_HingesWeights(const std::vector<int> faces_indices, const double add);
	void Set_HingesWeights(const std::vector<int> faces_indices, const double value);
	void Clear_HingesWeights();
	void Clear_HingesSigmoid();
	double Phi(
		const double x,
		const double SigmoidParameter,
		const Cuda::PenaltyFunction penaltyFunction);
	double dPhi_dm(
		const double x,
		const double SigmoidParameter,
		const Cuda::PenaltyFunction penaltyFunction);
	AuxVariables(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F, const Cuda::PenaltyFunction type);
	~AuxVariables();
};

