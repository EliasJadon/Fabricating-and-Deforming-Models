#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <atomic>
#include <vector>
#include <mutex>
#include <iostream>
#include "CudaBasics.cuh"


namespace Cuda {
	hinge newHinge(int f0, int f1) {
		hinge a;
		a.f0 = f0;
		a.f1 = f1;
		return a;
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

