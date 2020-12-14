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
			std::cout << "Device Number: " << i << std::endl;
			std::cout << "\tName: " << prop.name << std::endl;
			std::cout << "\tMemory Clock Rate (KHz): " << prop.memoryClockRate << std::endl;
			std::cout << "\tMemory Bus Width (bits): " << prop.memoryBusWidth << std::endl;
			std::cout << "\tPeak Memory Bandwidth (GB/s): " << 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6 << std::endl;
			std::cout << "\tprop.maxThreadsPerBlock = " << prop.maxThreadsPerBlock << std::endl;
			std::cout << "\tprop.maxThreadsDim[0] = " << prop.maxThreadsDim[0] << std::endl;
			std::cout << "\tprop.maxThreadsDim[1] = " << prop.maxThreadsDim[1] << std::endl;
			std::cout << "\tprop.maxThreadsDim[2] = " << prop.maxThreadsDim[2] << std::endl;
			std::cout << "\tprop.maxGridSize[0] = " << prop.maxGridSize[0] << std::endl;
			std::cout << "\tprop.maxGridSize[1] = " << prop.maxGridSize[1] << std::endl;
			std::cout << "\tprop.maxGridSize[2] = " << prop.maxGridSize[2] << std::endl;
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

