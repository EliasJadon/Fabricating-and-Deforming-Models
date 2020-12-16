#pragma once
#include "Cuda_Basics.cuh"

__device__ double3 sub(const double3 a, const double3 b);
__device__ double3 add(double3 a, double3 b);
__device__ double dot(const double3 a, const double3 b);
__device__ double3 mul(const double a, const double3 b);
__device__ double squared_norm(const double3 a);
__device__ double norm(const double3 a);
__device__ double3 normalize(const double3 a);
__device__ double3 cross(const double3 a, const double3 b);
__device__ double atomicAdd(double* address, double val, int flag);
__device__ double Phi(const double x, const double planarParameter, const FunctionType functionType);
__device__ double dPhi_dm(const double x, const double planarParameter, const FunctionType functionType);
template <unsigned int blockSize, typename T>
__device__ void warpReduce(volatile T* sdata, unsigned int tid);
template <unsigned int blockSize, typename T>
__global__ void sumOfArray(T* g_idata, T* g_odata, unsigned int n);
template<typename T> __global__ void setZeroKernel(T* vec);
