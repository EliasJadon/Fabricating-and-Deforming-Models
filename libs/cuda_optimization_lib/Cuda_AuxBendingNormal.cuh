#pragma once
#include "Cuda_Basics.cuh"

class Cuda_AuxBendingNormal {
private:
	double SigmoidParameter;
public:
	//Dynamic variables
	double w1 = 1, w2 = 100, w3 = 100;
	PenaltyFunction penaltyFunction;
	Cuda::Array<double> grad;
	Cuda::Array<double> EnergyAtomic;
	void Inc_SigmoidParameter() {
		SigmoidParameter *= 2;
		for (int hi = 0; hi < mesh_indices.num_hinges; hi++) {
			Sigmoid_PerHinge.host_arr[hi] *= 2;
		}
		Cuda::MemCpyHostToDevice(Sigmoid_PerHinge);
	}
	void Dec_SigmoidParameter() {
		SigmoidParameter /= 2;
		for (int hi = 0; hi < mesh_indices.num_hinges; hi++) {
			Sigmoid_PerHinge.host_arr[hi] /= 2;
		}
		Cuda::MemCpyHostToDevice(Sigmoid_PerHinge);
	}
	void Dec_SigmoidParameter(const double target) {
		if (SigmoidParameter > target)
			SigmoidParameter /= 2;
		for (int hi = 0; hi < mesh_indices.num_hinges; hi++) {
			if (Sigmoid_PerHinge.host_arr[hi] > target)
				Sigmoid_PerHinge.host_arr[hi] /= 2;
		}
		Cuda::MemCpyHostToDevice(Sigmoid_PerHinge);
	}
	double get_SigmoidParameter() {
		return SigmoidParameter;
	}

	//Static variables
	cudaStream_t stream_value, stream_gradient;
	Cuda::Array<int3> restShapeF;
	Cuda::Array<double> restAreaPerFace, restAreaPerHinge, weight_PerHinge, Sigmoid_PerHinge;
	Cuda::indices mesh_indices;
	Cuda::Array<Cuda::hinge> hinges_faceIndex; 
	Cuda::Array<int> x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd;
	Cuda::Array<Cuda::hinge> x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd;

	Cuda_AuxBendingNormal(const PenaltyFunction type, const int numF, const int numV, const int numH);
	~Cuda_AuxBendingNormal();
	void value(Cuda::Array<double>& curr_x);
	void gradient(Cuda::Array<double>& X);
};