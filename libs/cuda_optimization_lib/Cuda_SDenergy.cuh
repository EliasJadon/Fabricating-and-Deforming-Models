#pragma once
#include "Cuda_Basics.cuh"

class Cuda_SDenergy {
public:
	Cuda::Array<double> grad, Energy, EnergyAtomic, restShapeArea;
	Cuda::Array<double3> D1d, D2d;
	Cuda::Array<int3> restShapeF;
	Cuda::indices mesh_indices;
	cudaStream_t stream_value, stream_gradient;

	Cuda_SDenergy(const int F, const int V);
	~Cuda_SDenergy();
	void value(Cuda::Array<double>& curr_x);
	void gradient(Cuda::Array<double>& X);
};