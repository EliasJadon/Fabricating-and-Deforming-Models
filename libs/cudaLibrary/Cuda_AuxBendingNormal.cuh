#pragma once
#include "CudaBasics.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

namespace Cuda {
	namespace AuxBendingNormal {
		
		//Dynamic variables
		extern double w1, w2, w3;
		extern FunctionType functionType; //OptimizationUtils::FunctionType /*QUADRATIC = 0,EXPONENTIAL = 1,SIGMOID = 2*/
		extern double planarParameter;
		extern Array<rowVector<double>> CurrV, CurrN; //Eigen::MatrixX3d
		extern Array<double> d_normals;
		extern Array<double> grad;
		extern Array<double> EnergyAtomic;

		//Static variables
		extern Array<rowVector<int>> restShapeF;
		extern Array<double> restAreaPerFace, restAreaPerHinge; //Eigen::VectorXd
		extern int num_hinges, num_faces, num_vertices;
		extern Array<hinge> hinges_faceIndex; //std::vector<Eigen::Vector2d> //num_hinges*2
		extern Array<int> x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd; //Eigen::VectorXi //num_hinges
		extern Array<hinge> x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd; //Eigen::MatrixXi //num_hinges*2
		
		
		extern void init();
		extern void updateX();
		extern double value();
		extern void FreeAllVariables();

		template<typename T>
		void FreeMemory(Cuda::Array<T>& a) {
			delete[] a.host_arr;
			cudaFree(a.cuda_arr);
		}
		template<typename T>
		void AllocateMemory(Cuda::Array<T>& a, const unsigned int size) {
			if (size <= 0) {
				std::cout << "Cuda: the size isn't positive!\n";
				FreeAllVariables();
				exit(1);
			}
			a.size = size;
			a.host_arr = new T[size];
			if (a.host_arr == NULL) {
				std::cout << "Host: Allocation Failed!!!\n";
				FreeAllVariables();
				exit(1);
			}
			cudaError_t cudaStatus;
			cudaStatus = cudaMalloc((void**)& a.cuda_arr, a.size * sizeof(T));
			if (cudaStatus != cudaSuccess) {
				std::cout << "Device: Allocation Failed!!!\n";
				FreeAllVariables();
				exit(1);
			}
		}
		template <typename T> void MemCpyHostToDevice(Array<T>& a) {
			cudaError_t cudaStatus;
			cudaStatus = cudaMemcpy(a.cuda_arr, a.host_arr, a.size * sizeof(T), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpyHostToDevice failed!");
				FreeAllVariables();
				exit(1);
			}
		}
		template <typename T> void MemCpyDeviceToHost(Array<T>& a) {
			cudaError_t cudaStatus;
			// Copy output vector from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(a.host_arr, a.cuda_arr, a.size * sizeof(T), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpyDeviceToHost failed!");
				FreeAllVariables();
				exit(1);
			}
		}
		
	}
}