#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_AuxBendingNormal.cuh"

class AuxBendingNormal : public ObjectiveFunction
{	
private:
	double SigmoidParameter;
public:
	//Dynamic variables
	double w1 = 1, w2 = 100, w3 = 100;
	Cuda::PenaltyFunction penaltyFunction;
	void Inc_SigmoidParameter() {
		SigmoidParameter *= 2;
		for (int hi = 0; hi < mesh_indices.num_hinges; hi++) {
			Sigmoid_PerHinge.host_arr[hi] *= 2;
		}
	}
	void Dec_SigmoidParameter() {
		SigmoidParameter /= 2;
		for (int hi = 0; hi < mesh_indices.num_hinges; hi++) {
			Sigmoid_PerHinge.host_arr[hi] /= 2;
		}
	}
	void Dec_SigmoidParameter(const double target) {
		if (SigmoidParameter > target)
			SigmoidParameter /= 2;
		for (int hi = 0; hi < mesh_indices.num_hinges; hi++) {
			if (Sigmoid_PerHinge.host_arr[hi] > target)
				Sigmoid_PerHinge.host_arr[hi] /= 2;
		}
	}
	double get_SigmoidParameter() {
		return SigmoidParameter;
	}

	//Static variables
	Cuda::Array<double> weight_PerHinge, Sigmoid_PerHinge;
	
public:
	Eigen::VectorXd restAreaPerFace, restAreaPerHinge;
	int num_hinges = -1;
	
	Eigen::VectorXi x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd;
	Eigen::MatrixXi x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd;
	void calculateHinges();	
public:
	std::vector<Eigen::Vector2d> hinges_faceIndex;
	Eigen::Vector3f colorP, colorM;
	void Reset_HingesSigmoid(const std::vector<int> faces_indices);
	void Incr_HingesWeights(const std::vector<int> faces_indices, const double add);
	void Set_HingesWeights(const std::vector<int> faces_indices, const double value);
	void Update_HingesSigmoid(const std::vector<int> faces_indices, const double factor);
	void Clear_HingesWeights();
	void Clear_HingesSigmoid();

	AuxBendingNormal(
		const Eigen::MatrixXd& V, 
		const Eigen::MatrixX3i& F,
		const Cuda::PenaltyFunction type);
	~AuxBendingNormal();
	
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual void gradient(Cuda::Array<double>& X, const bool update) override;
};

