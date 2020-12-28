#pragma once
#include "Cuda_Basics.cuh"

class Cuda_STVK {
public:
	Cuda::Array<double> grad, Energy, EnergyAtomic, restShapeArea;
	Cuda::Array<int3> restShapeF;
	Cuda::Array<double4> dXInv;
	Cuda::indices mesh_indices;
	double shearModulus, bulkModulus;
	
	Cuda_STVK();
	~Cuda_STVK();
	void value(Cuda::Array<double>& curr_x);
	void gradient(Cuda::Array<double>& X);
};