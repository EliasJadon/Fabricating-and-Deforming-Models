#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <math.h>

#define USING_CUDA

enum FunctionType 
{
	QUADRATIC = 0,
	EXPONENTIAL = 1,
	SIGMOID = 2
};

namespace Cuda 
{
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
		T* cuda_arr;
	};

	extern void initIndices(
		indices& I,
		const unsigned int F,
		const unsigned int V,
		const unsigned int H);
	extern hinge newHinge(int f0, int f1);
	extern void view_device_properties();
	extern void initCuda();
	extern void StopCudaDevice();
	extern void CheckErr(const cudaError_t cudaStatus, const int ID = 0);
	extern void copyArrays(Array<double>& a, const Array<double>& b);
	
	template<typename T> void FreeMemory(Cuda::Array<T>& a) 
	{
		delete[] a.host_arr;
		cudaFree(a.cuda_arr);
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
		CheckErr(cudaMalloc((void**)& a.cuda_arr, a.size * sizeof(T)));
	}

	template <typename T> void MemCpyHostToDevice(Array<T>& a) 
	{
		CheckErr(cudaMemcpy(a.cuda_arr, a.host_arr, a.size * sizeof(T), cudaMemcpyHostToDevice));
	}

	template <typename T> void MemCpyDeviceToHost(Array<T>& a) 
	{
		// Copy output vector from GPU buffer to host memory.
		CheckErr(cudaMemcpy(a.host_arr, a.cuda_arr, a.size * sizeof(T), cudaMemcpyDeviceToHost));
	}
}
