//#pragma once
//#include "Cuda_Basics.cuh"
//
//class Cuda_FixChosenConstraints {
//public:
//	Cuda_FixChosenConstraints(const unsigned int numF, const unsigned int numV);
//	~Cuda_FixChosenConstraints();
//	Cuda::Array<double> grad, EnergyAtomic;
//	Cuda::indices mesh_indices;
//	Cuda::Array<int> Const_Ind;
//	Cuda::Array<double3> Const_Pos;
//	unsigned int startX, startY, startZ;
//	cudaStream_t stream_value, stream_gradient;
//
//	void value(Cuda::Array<double>& curr_x);
//	void gradient(Cuda::Array<double>& X);
//};
