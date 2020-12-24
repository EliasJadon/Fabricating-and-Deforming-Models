#pragma once
#include "Cuda_Basics.cuh"

class Cuda_AuxSpherePerHinge {
public:
	//Dynamic variables
	double w1 = 1, w2 = 100;
	FunctionType functionType; 
	double planarParameter;
	Cuda::Array<double> grad;
	Cuda::Array<double> EnergyAtomic;

	//Static variables
	Cuda::Array<int3> restShapeF;
	Cuda::Array<double> restAreaPerFace, restAreaPerHinge; 
	Cuda::indices mesh_indices;
	Cuda::Array<Cuda::hinge> hinges_faceIndex;
	Cuda::Array<int> x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd; 
	Cuda::Array<Cuda::hinge> x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd; 

	Cuda_AuxSpherePerHinge();
	~Cuda_AuxSpherePerHinge();
	double value(Cuda::Array<double>& curr_x);
	Cuda::Array<double>* gradient(Cuda::Array<double>& X);
};