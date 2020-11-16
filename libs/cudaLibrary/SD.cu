#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <atomic>
#include <vector>
#include <mutex>
#include "CudaBasics.h"


namespace Cuda {
	__global__ void addKernel(int* c, const int* a, const int* b, const int size)
	{
		int index = threadIdx.x + blockIdx.x * blockDim.x;
		if (index < size)
		{
			c[index] = a[index] + b[index];
		}
	}


	void initCuda() {
		cudaError_t cudaStatus;
		check_devices_properties();
		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			cudaGetErrorString(cudaStatus);
		}
		else {
			printf("cudaSetDevice successfully!\n");
		}
	}

	void StopCudaDevice() {
		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaError_t cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			cudaGetErrorString(cudaStatus);
		}
		else {
			printf("cudaDeviceReset successfully!\n");
		}
	}


	//int main()
	//{
	//    const int arraySize = 5;
	//    const int a[arraySize] = { 1, 2, 3, 4, 5 };
	//    const int b[arraySize] = { 10, 20, 30, 40, 50 };
	//    int c[arraySize] = { 0 };
	//
	//    // Add vectors in parallel.
	//    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
	//    if (cudaStatus != cudaSuccess) {
	//        fprintf(stderr, "addWithCuda failed!");
	//        return 1;
	//    }
	//
	//    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
	//        c[0], c[1], c[2], c[3], c[4]);
	//
	//    // cudaDeviceReset must be called before exiting in order for profiling and
	//    // tracing tools such as Nsight and Visual Profiler to show complete traces.
	//    cudaStatus = cudaDeviceReset();
	//    if (cudaStatus != cudaSuccess) {
	//        fprintf(stderr, "cudaDeviceReset failed!");
	//        return 1;
	//    }
	//
	//    return 0;
	//}

	__global__ void GroupNormals_value_kernel(
		double*** NormalPos,
		int numClusters,
		int rows,
		int cols,
		double* result)
	{
		int index =
			threadIdx.x +
			blockIdx.x * blockDim.x +
			blockIdx.y * blockDim.x * gridDim.x +
			blockIdx.z * blockDim.x * gridDim.x * gridDim.y;

		int cluster_index = blockIdx.x;
		int f1 = blockIdx.y;
		int f2 = blockIdx.z;
		int c = threadIdx.x;
		if (cluster_index < numClusters &&
			f1 < rows &&
			f2 < rows && f2 > f1 && c < cols)
		{
			double diff = (NormalPos[cluster_index][f1][c] - NormalPos[cluster_index][f2][c]);
			result[index] = diff * diff;
		}
		else {
			result[index] = 0;
		}
	}

	double GroupNormals_value(
		double*** NormalPos,
		int numClusters,
		int rows,
		int cols)
	{
		double E = 0;

		dim3 blocksDir(numClusters, rows, rows);
		// Launch a kernel on the GPU with one thread for each element.

		return E;
	}

	// Helper function for using CUDA to add vectors in parallel.
	void addWithCuda(int* c, const int* a, const int* b, unsigned int size)
	{
		int* dev_a = 0;
		int* dev_b = 0;
		int* dev_c = 0;
		cudaError_t cudaStatus;

		// Allocate GPU buffers for three vectors (two input, one output)    .
		cudaStatus = cudaMalloc((void**)& dev_c, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)& dev_a, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		cudaStatus = cudaMalloc((void**)& dev_b, size * sizeof(int));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		// Copy input vectors from host memory to GPU buffers.
		cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

		// Launch a kernel on the GPU with one thread for each element.
		addKernel << <1, size >> > (dev_c, dev_a, dev_b, size);

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
			goto Error;
		}

		// Copy output vector from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!");
			goto Error;
		}

	Error:
		cudaGetErrorString(cudaStatus);
		cudaFree(dev_c);
		cudaFree(dev_a);
		cudaFree(dev_b);
		return;
	}

	void check_devices_properties() {
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		cudaError_t cudaStatus;
		for (int i = 0; i < nDevices; i++) {
			cudaDeviceProp prop;
			cudaStatus = cudaGetDeviceProperties(&prop, i);
			cudaGetErrorString(cudaStatus);
			printf("Device Number: %d\n", i);
			printf("  Device name: %s\n", prop.name);
			printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
			printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
			printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
			printf("  prop.maxThreadsPerBlock = %d\n", prop.maxThreadsPerBlock);
			printf("  prop.maxThreadsDim[0] = %d\n", prop.maxThreadsDim[0]);
			printf("  prop.maxThreadsDim[1] = %d\n", prop.maxThreadsDim[1]);
			printf("  prop.maxThreadsDim[2] = %d\n", prop.maxThreadsDim[2]);
			printf("  prop.maxGridSize[0] = %d\n", prop.maxGridSize[0]);
			printf("  prop.maxGridSize[1] = %d\n", prop.maxGridSize[1]);
			printf("  prop.maxGridSize[2] = %d\n", prop.maxGridSize[2]);
		}
	}
}

