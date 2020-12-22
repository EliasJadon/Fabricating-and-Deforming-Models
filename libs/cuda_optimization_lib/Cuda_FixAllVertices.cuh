#pragma once
#include "Cuda_Basics.cuh"

class Cuda_FixAllVertices {
public:
	Cuda::Array<double> grad, EnergyAtomic;
	Cuda::Array<double3> restShapeV;
	unsigned int num_faces, num_vertices;

	Cuda_FixAllVertices();
	~Cuda_FixAllVertices();
	double value(Cuda::Array<double>& curr_x);
	Cuda::Array<double>* gradient(Cuda::Array<double>& X);
};