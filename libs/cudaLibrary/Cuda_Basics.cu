#include "Cuda_Basics.cuh"


namespace Cuda {
	hinge newHinge(int f0, int f1) {
		hinge a;
		a.f0 = f0;
		a.f1 = f1;
		return a;
	}

	void CheckErr(const cudaError_t cudaStatus, const int ID) {
		if (cudaStatus != cudaSuccess) {
			std::cout << "Error!!!" << std::endl;
			std::cout << "ID = " << ID << std::endl;
			std::cout << "cudaStatus:\t" << cudaGetErrorString(cudaStatus) << std::endl;
			std::cout << "Last Error:\t" << cudaGetErrorString(cudaGetLastError()) << std::endl;
			exit(1);
		}
	}
	
	void initCuda() {
		view_device_properties();
		// Choose which GPU to run on, change this on a multi-GPU system.
		CheckErr(cudaSetDevice(0));
		std::cout << "cudaSetDevice successfully!\n";
		
	}
	void StopCudaDevice() {
		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		CheckErr(cudaDeviceReset());
		std::cout << "cudaDeviceReset successfully!\n";
	}
	void view_device_properties() {
		int nDevices;
		cudaGetDeviceCount(&nDevices);
		for (int i = 0; i < nDevices; i++) {
			cudaDeviceProp prop;
			CheckErr(cudaGetDeviceProperties(&prop, i));
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

	__global__ void copyArraysKernel(double* a, const double* b) {
		int index = blockIdx.x;
		a[index] = b[index];
	}

	void copyArrays(Array<double>& a, const Array<double>& b) {
		copyArraysKernel << <a.size, 1 >> > (a.cuda_arr, b.cuda_arr);
		CheckErr(cudaDeviceSynchronize());
	}
}

