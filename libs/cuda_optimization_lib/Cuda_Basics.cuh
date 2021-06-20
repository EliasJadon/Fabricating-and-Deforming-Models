#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>

#define USING_CUDA



namespace Cuda 
{
	enum PenaltyFunction { QUADRATIC, EXPONENTIAL, SIGMOID };
	enum OptimizerType { Gradient_Descent, Adam };

	struct indices {
		unsigned int
			startVx, startVy, startVz,
			startNx, startNy, startNz,
			startCx, startCy, startCz,
			startR,
			num_vertices, num_faces, num_hinges;
	};

	struct hinge 
	{
		int f0, f1;
	};
	
	template <typename T> struct Array 
	{
		unsigned int size;
		T* host_arr;
	};

	extern void initIndices(
		indices& I,
		const unsigned int F,
		const unsigned int V,
		const unsigned int H);
	extern hinge newHinge(int f0, int f1);
	
	template<typename T> void FreeMemory(Cuda::Array<T>& a) 
	{
		delete[] a.host_arr;
	}

	template<typename T> void AllocateMemory(Cuda::Array<T>& a, const unsigned int size) 
	{
		if (size < 0) {
			std::cout << "Cuda: the size isn't positive!\n";
			exit(1);
		}
		a.size = size;
		a.host_arr = new T[size];
		if (a.host_arr == NULL) {
			std::cout << "Host: Allocation Failed!!!\n";
			exit(1);
		}
	}

}
