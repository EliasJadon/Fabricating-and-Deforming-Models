#pragma once
#include "Cuda_Basics.cuh"

class Cuda_AuxCylinder {
private:
	double SigmoidParameter;
public:
	//Dynamic variables
	double w1 = 1, w2 = 1, w3 = 100;
	FunctionType functionType; 
	Cuda::Array<double> grad;
	Cuda::Array<double> EnergyAtomic;
	void Inc_SigmoidParameter() {
		SigmoidParameter *= 2;
	}
	void Dec_SigmoidParameter() {
		SigmoidParameter /= 2;
	}
	double get_SigmoidParameter() {
		return SigmoidParameter;
	}

	//Static variables
	cudaStream_t stream_value, stream_gradient;
	Cuda::Array<int3> restShapeF;
	Cuda::Array<double> restAreaPerFace, restAreaPerHinge; 
	Cuda::indices mesh_indices;
	Cuda::Array<Cuda::hinge> hinges_faceIndex; 
	Cuda::Array<int> x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd;
	Cuda::Array<Cuda::hinge> x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd;

	Cuda_AuxCylinder(const FunctionType type, const int numF, const int numV, const int numH);
	~Cuda_AuxCylinder();
	void value(Cuda::Array<double>& curr_x);
	void gradient(Cuda::Array<double>& X);
};