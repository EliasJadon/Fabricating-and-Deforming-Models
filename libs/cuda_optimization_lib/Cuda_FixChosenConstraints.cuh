#pragma once
#include "Cuda_Basics.cuh"

class Cuda_FixChosenConstraints {
public:
	Cuda_FixChosenConstraints(const unsigned int numF,
		const unsigned int numV,
		const unsigned int Type);
	~Cuda_FixChosenConstraints();
	Cuda::Array<double> grad, EnergyAtomic;
	Cuda::indices mesh_indices;
	Cuda::Array<int> Const_Ind;
	Cuda::Array<double3> Const_Pos;
	unsigned int startX, startY, startZ;

	double value();
	void gradient();
};
