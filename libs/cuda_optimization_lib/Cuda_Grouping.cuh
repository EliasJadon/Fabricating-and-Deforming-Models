#pragma once
#include "Cuda_Basics.cuh"

class Cuda_Grouping {
public:
	Cuda_Grouping(const unsigned int numF,
		const unsigned int numV,
		const ConstraintsType Type);
	~Cuda_Grouping();
	Cuda::Array<double> grad, EnergyAtomic;
	Cuda::indices mesh_indices;
	Cuda::Array<int> Const_Ind;
	Cuda::Array<double3> Const_Pos;
	unsigned int startX, startY, startZ;

	double value(Cuda::Array<double>& curr_x);
	void gradient(Cuda::Array<double>& X);
};
