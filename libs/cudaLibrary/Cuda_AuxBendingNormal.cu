#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <iostream>
#include <atomic>
#include <mutex>
#include "Cuda_AuxBendingNormal.h"

namespace Cuda {
	namespace AuxBendingNormal {
		void init(int numvertices, int numfaces, int numhinges)
		{
			num_faces = numfaces;
			num_vertices = numvertices;
			num_hinges = numhinges;

			CurrV.size = num_vertices;
			CurrN.size = num_faces;
			restAreaPerFace.size = num_faces;
			restAreaPerHinge.size = num_hinges;
			d_normals.size = num_hinges;
			hinges_faceIndex.size = num_hinges;
			x0_GlobInd.size = x1_GlobInd.size = x2_GlobInd.size = x3_GlobInd.size 
				= x0_LocInd.size = x1_LocInd.size = x2_LocInd.size = x3_LocInd.size = num_hinges;
			
			//alocate mem on the CPU
			Host_allocateMem(CurrV);
			Host_allocateMem(CurrN);
			Host_allocateMem(restAreaPerFace);
			Host_allocateMem(restAreaPerHinge);
			Host_allocateMem(d_normals);
			Host_allocateMem(hinges_faceIndex);
			Host_allocateMem(x0_GlobInd);
			Host_allocateMem(x1_GlobInd);
			Host_allocateMem(x2_GlobInd);
			Host_allocateMem(x3_GlobInd);
			Host_allocateMem(x0_LocInd);
			Host_allocateMem(x1_LocInd);
			Host_allocateMem(x2_LocInd);
			Host_allocateMem(x3_LocInd);

			//allocate mem on the GPU
			Cuda_allocateMem(CurrV);
			Cuda_allocateMem(CurrN);
			Cuda_allocateMem(restAreaPerFace);
			Cuda_allocateMem(restAreaPerHinge);
			Cuda_allocateMem(d_normals);
			Cuda_allocateMem(hinges_faceIndex);
			Cuda_allocateMem(x0_GlobInd);
			Cuda_allocateMem(x1_GlobInd);
			Cuda_allocateMem(x2_GlobInd);
			Cuda_allocateMem(x3_GlobInd);
			Cuda_allocateMem(x0_LocInd);
			Cuda_allocateMem(x1_LocInd);
			Cuda_allocateMem(x2_LocInd);
			Cuda_allocateMem(x3_LocInd);

			//TODO: init host buffers...

			// Copy input vectors from host memory to GPU buffers.
			Cuda_MemcpyHostToDevice(CurrV);
			Cuda_MemcpyHostToDevice(CurrN);
			Cuda_MemcpyHostToDevice(restAreaPerFace);
			Cuda_MemcpyHostToDevice(restAreaPerHinge);
			Cuda_MemcpyHostToDevice(d_normals);
			Cuda_MemcpyHostToDevice(hinges_faceIndex);
			Cuda_MemcpyHostToDevice(x0_GlobInd);
			Cuda_MemcpyHostToDevice(x1_GlobInd);
			Cuda_MemcpyHostToDevice(x2_GlobInd);
			Cuda_MemcpyHostToDevice(x3_GlobInd);
			Cuda_MemcpyHostToDevice(x0_LocInd);
			Cuda_MemcpyHostToDevice(x1_LocInd);
			Cuda_MemcpyHostToDevice(x2_LocInd);
			Cuda_MemcpyHostToDevice(x3_LocInd);
		}

		template <typename T>
		void Host_allocateMem(Array<T> a) {
			if (a.size <= 0) {
				std::cout << "Host: The size of the array isn't initialized yet!\n";
				Host_freeMemory();
				Cuda_freeMemory();
				exit(1);
			}
			a.host_arr = (T*)malloc(a.size * sizeof(T));
			if (a.host_arr == NULL) {
				std::cout << "Host allocation failed!\n";
				Host_freeMemory();
				Cuda_freeMemory();
				exit(1);
			}
		}

		template <typename T>
		void Cuda_allocateMem(Array<T> a) {
			if (a.size <= 0) {
				std::cout << "Cuda: The size of the array isn't initialized yet!\n";
				Host_freeMemory();
				Cuda_freeMemory();
				exit(1);
			}
			cudaError_t cudaStatus;
			cudaStatus = cudaMalloc((void**)& a.cuda_arr, a.size * sizeof(T));
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMalloc failed!");
				Host_freeMemory();
				Cuda_freeMemory();
				exit(1);
			}
		}

		template <typename T>
		void Cuda_MemcpyHostToDevice(Array<T> a) {
			if (a.size <= 0) {
				std::cout << "Cuda: The size of the array isn't initialized yet!\n";
				Host_freeMemory();
				Cuda_freeMemory();
				exit(1);
			}
			cudaError_t cudaStatus;
			cudaStatus = cudaMemcpy(a.cuda_arr, a.host_arr, a.size * sizeof(T), cudaMemcpyHostToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpyHostToDevice failed!");
				Host_freeMemory();
				Cuda_freeMemory();
				exit(1);
			}
		}

		void Host_freeMemory() {
			free(CurrV.host_arr);
			free(CurrN.host_arr);
			free(restAreaPerFace.host_arr);
			free(restAreaPerHinge.host_arr);
			free(d_normals.host_arr);
			free(hinges_faceIndex.host_arr);
			free(x0_GlobInd.host_arr);
			free(x1_GlobInd.host_arr);
			free(x2_GlobInd.host_arr);
			free(x3_GlobInd.host_arr);
			free(x0_LocInd.host_arr);
			free(x1_LocInd.host_arr);
			free(x2_LocInd.host_arr);
			free(x3_LocInd.host_arr);
		}

		void Cuda_freeMemory() {
			cudaGetErrorString(cudaGetLastError());

			cudaFree(CurrV.cuda_arr);
			cudaFree(CurrN.cuda_arr);
			cudaFree(restAreaPerFace.cuda_arr);
			cudaFree(restAreaPerHinge.cuda_arr);
			cudaFree(d_normals.cuda_arr);
			cudaFree(hinges_faceIndex.cuda_arr);
			cudaFree(x0_GlobInd.cuda_arr);
			cudaFree(x1_GlobInd.cuda_arr);
			cudaFree(x2_GlobInd.cuda_arr);
			cudaFree(x3_GlobInd.cuda_arr);
			cudaFree(x0_LocInd.cuda_arr);
			cudaFree(x1_LocInd.cuda_arr);
			cudaFree(x2_LocInd.cuda_arr);
			cudaFree(x3_LocInd.cuda_arr);
		}

	}
}
