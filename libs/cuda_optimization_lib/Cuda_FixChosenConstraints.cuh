#pragma once
#include "Cuda_Basics.cuh"

class Cuda_FixChosenConstraints {
public:
	Cuda_FixChosenConstraints(const unsigned int numF,
		const unsigned int numV,
		const ConstraintsType Type);
	~Cuda_FixChosenConstraints();
	Cuda::Array<double> grad, EnergyAtomic;
	Cuda::indices mesh_indices;
	Cuda::Array<int> Const_Ind;
	Cuda::Array<double3> Const_Pos;
	unsigned int startX, startY, startZ;

	void value(Cuda::Array<double>& curr_x);
	Cuda::Array<double>* gradient(Cuda::Array<double>& X);
};
