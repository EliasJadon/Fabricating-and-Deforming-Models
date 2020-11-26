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


	namespace Minimizer {
		Array<double> X, p, g, curr_x;

		__global__ void currXKernel(
			double* curr_x,
			const double* X,
			const double* p,
			const double step_size)
		{
			int index = blockIdx.x;
			curr_x[index] = X[index] + step_size * p[index];
		}

		void linesearch_currX(const double step_size) {
			currXKernel << <curr_x.size, 1 >> > (
				curr_x.cuda_arr,
				X.cuda_arr,
				p.cuda_arr,
				step_size);
			CheckErr(cudaDeviceSynchronize());
		}
	}

	namespace AdamMinimizer {
		Array<double> v_adam, s_adam;

		__global__ void stepKernel(
			const double alpha_adam, 
			const double beta1_adam,
			const double beta2_adam,
			const double* g,
			double* v_adam,
			double* s_adam,
			double* p) 
		{
			int index = blockIdx.x;
			v_adam[index] = beta1_adam * v_adam[index] + (1 - beta1_adam) * g[index];
			double tmp = pow(g[index], 2);
			s_adam[index] = beta2_adam * s_adam[index] + (1 - beta2_adam) * tmp; // element-wise square
			tmp = sqrt(s_adam[index]) + 1e-8;
			p[index] = -v_adam[index] / tmp;
		}

		void step(
			const double alpha_adam, 
			const double beta1_adam, 
			const double beta2_adam) 
		{
			stepKernel << <v_adam.size, 1 >> > (
				alpha_adam,
				beta1_adam,
				beta2_adam,
				Minimizer::g.cuda_arr,
				v_adam.cuda_arr,
				s_adam.cuda_arr,
				Minimizer::p.cuda_arr);
			CheckErr(cudaDeviceSynchronize());
		}
	}

}

