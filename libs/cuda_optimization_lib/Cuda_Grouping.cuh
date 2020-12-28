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
	Cuda::Array<int> Group_Ind;
	unsigned int startX, startY, startZ;
	unsigned int num_clusters, max_face_per_cluster;
	
	void value(Cuda::Array<double>& curr_x);
	void gradient(Cuda::Array<double>& X);
};
