#pragma once
#include "CudaBasics.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace Cuda {
	namespace AuxBendingNormal {
		//Dynamic variables
		extern double w1, w2, w3;
		extern FunctionType functionType; //OptimizationUtils::FunctionType /*QUADRATIC = 0,EXPONENTIAL = 1,SIGMOID = 2*/
		extern double planarParameter;
		extern Array<rowVector> CurrV, CurrN; //Eigen::MatrixX3d
		extern Array<double> d_normals;

		extern Array<double> Energy1, Energy2, Energy3;

		//Static variables
		extern Array<double> restAreaPerFace, restAreaPerHinge; //Eigen::VectorXd
		extern int num_hinges, num_faces, num_vertices;
		extern Array<hinge> hinges_faceIndex; //std::vector<Eigen::Vector2d> //num_hinges*2
		extern Array<int> x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd; //Eigen::VectorXi //num_hinges
		extern Array<hinge> x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd; //Eigen::MatrixXi //num_hinges*2
		
		
		extern void Host_freeMemory();
		extern void Device_freeMemory();

		template <typename T> void Device_allocateMem(Array<T>& a) {
			if (a.size <= 0) {
				std::cout << "Cuda: The size of the array isn't initialized yet!\n";
				Host_freeMemory();
				Device_freeMemory();
				exit(1);
			}
			cudaError_t cudaStatus;
			cudaStatus = cudaMalloc((void**)& a.cuda_arr, a.size * sizeof(T));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
				Host_freeMemory();
				Device_freeMemory();
				exit(1);
			}
		}
		template <typename T> void MemCpyHostToDevice(Array<T>& a) {
			if (a.size <= 0) {
				std::cout << "Cuda: The size of the array isn't initialized yet!\n";
				Host_freeMemory();
				Device_freeMemory();
				exit(1);
			}
			cudaError_t cudaStatus;
			cudaStatus = cudaMemcpy(a.cuda_arr, a.host_arr, a.size * sizeof(T), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpyHostToDevice failed!");
				Host_freeMemory();
				Device_freeMemory();
				exit(1);
			}
		}
		template <typename T> void MemCpyDeviceToHost(Array<T>& a) {
			if (a.size <= 0) {
				std::cout << "Cuda: The size of the array isn't initialized yet!\n";
				Host_freeMemory();
				Device_freeMemory();
				exit(1);
			}
			cudaError_t cudaStatus;
			// Copy output vector from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(a.host_arr, a.cuda_arr, a.size * sizeof(T), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpyDeviceToHost failed!");
				Host_freeMemory();
				Device_freeMemory();
				exit(1);
			}
		}

		extern void init();
		extern void updateX();

		template<typename T>
		void allocHostMem(Cuda::Array<T>& a, const unsigned int size) {
			a.size = size;
			a.host_arr = new T[size];
			if (a.host_arr == NULL) {
				std::cout << "Host: Allocation Failed!!!\n";
				exit(1);
			}
		}


	}
}