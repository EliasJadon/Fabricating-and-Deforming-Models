#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <vector>
#include <iostream>
#include <atomic>
#include <mutex>
#include "Cuda_AuxBendingNormal.cuh"

namespace Cuda {
	namespace AuxBendingNormal {
		//dynamic variables
		double w1 = 1, w2 = 100, w3 = 100;
		FunctionType functionType;
		double planarParameter;
		Array<rowVector<double>> CurrV, CurrN; //Eigen::MatrixX3d
		Array<double> d_normals;
		//help variables - dynamic
		Array<double> Energy1, Energy2, Energy3;
		
		//Static variables
		Array<rowVector<int>> restShapeF;
		Array<double> restAreaPerFace, restAreaPerHinge; //Eigen::VectorXd
		int num_hinges, num_faces, num_vertices;
		Array<hinge> hinges_faceIndex; //std::vector<Eigen::Vector2d> //num_hinges*2
		Array<int> x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd; //Eigen::VectorXi //num_hinges
		Array<hinge> x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd; //Eigen::MatrixXi //num_hinges*2
		
		__global__ void updateXKernel(
			double* d_normals, 
			const rowVector<double>* Normals, 
			const hinge* Hinges_Findex, 
			const int size)
		{
			int hi = blockIdx.x;
			if (hi < size)
			{
				int f0 = Hinges_Findex[hi].f0;
				int f1 = Hinges_Findex[hi].f1;
				double diffX = Normals[f1].x - Normals[f0].x;
				double diffY = Normals[f1].y - Normals[f0].y;
				double diffZ = Normals[f1].z - Normals[f0].z;
				d_normals[hi] = diffX * diffX + diffY * diffY + diffZ * diffZ;
			}
		}



		__global__ void Energy1Kernel(
			double* result, 
			const double* x,
			const double* area,
			const double planarParameter,
			const FunctionType functionType,
			const int size)
		{
			int hi = blockIdx.x;
			if (hi < size)
			{
				if (functionType == FunctionType::SIGMOID) {
					double x2 = x[hi] * x[hi];
					result[hi] = x2 / (x2 + planarParameter);
				}
				else if (functionType == FunctionType::QUADRATIC)
					result[hi] = x[hi] * x[hi];
				else if (functionType == FunctionType::EXPONENTIAL)
					result[hi] = 0;

				result[hi] *= area[hi];
			}
		}

		__global__ void Energy2Kernel(
			double* result, 
			const rowVector<double>* Normals,
			const int size)
		{
			int fi = blockIdx.x;
			if (fi < size)
			{
				double x2 = Normals[fi].x * Normals[fi].x;
				double y2 = Normals[fi].y * Normals[fi].y;
				double z2 = Normals[fi].z * Normals[fi].z;
				double sqrN = x2 + y2 + z2 - 1;
				result[fi] = sqrN * sqrN;
			}
		}

		template<typename T> __device__ rowVector<T> addVectors(
			rowVector<T> a, 
			rowVector<T> b) 
		{
			rowVector<T> result;
			result.x = a.x + b.x;
			result.y = a.y + b.y;
			result.z = a.z + b.z;
			return result;
		}
		template<typename T> __device__ rowVector<T> subVectors(
			const rowVector<T> a,
			const rowVector<T> b) 
		{
			rowVector<T> result;
			result.x = a.x - b.x;
			result.y = a.y - b.y;
			result.z = a.z - b.z;
			return result;
		}
		template<typename T> __device__ T mulVectors(rowVector<T> a, rowVector<T> b) 
		{
			return  
				a.x * b.x +
				a.y * b.y +
				a.z * b.z;
		}

		__global__ void Energy3Kernel(
			double* result, 
			const rowVector<int>* restShapeF,
			const rowVector<double>* Vertices,
			const rowVector<double>* Normals,
			const int size)
		{
			int fi = blockIdx.x;
			if (fi < size)
			{
				// (N^T*(x1-x0))^2 + (N^T*(x2-x1))^2 + (N^T*(x0-x2))^2
				int x0 = restShapeF[fi].x;
				int x1 = restShapeF[fi].y;
				int x2 = restShapeF[fi].z;

				rowVector<double> e21 = subVectors(Vertices[x2], Vertices[x1]);
				rowVector<double> e10 = subVectors(Vertices[x1], Vertices[x0]);
				rowVector<double> e02 = subVectors(Vertices[x0], Vertices[x2]);
				double d1 = mulVectors(Normals[fi], e21);
				double d2 = mulVectors(Normals[fi], e10);
				double d3 = mulVectors(Normals[fi], e02);
				result[fi] = d1 * d1 + d2 * d2 + d3 * d3;
			}
		}


		/*Energy1Kernel << <num_hinges, 1 >> > (
			Energy1.cuda_arr,
			d_normals.cuda_arr,
			restAreaPerHinge.cuda_arr,
			planarParameter,
			functionType,
			num_hinges);
		Energy2Kernel << <num_faces, 1 >> > (
			Energy2.cuda_arr,
			CurrN.cuda_arr,
			num_faces);
		Energy3Kernel << <num_faces, 1 >> > (
			Energy3.cuda_arr,
			restShapeF.cuda_arr,
			CurrV.cuda_arr,
			CurrN.cuda_arr,
			num_faces);*/

		double value() {
			//Energy1
			Energy1Kernel <<<num_hinges, 1 >>> (
				Energy1.cuda_arr,
				d_normals.cuda_arr,
				restAreaPerHinge.cuda_arr,
				planarParameter,
				functionType,
				num_hinges);
			if (cudaDeviceSynchronize() != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code after launching addKernel!\n");
				exit(1);
			}
			MemCpyDeviceToHost(Energy1);

			//Energy2
			Energy2Kernel <<<num_faces, 1 >>> (
				Energy2.cuda_arr,
				CurrN.cuda_arr,
				num_faces);
			if (cudaDeviceSynchronize() != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code after launching addKernel!\n");
				exit(1);
			}
			MemCpyDeviceToHost(Energy2);
			
			//Energy3
			Energy3Kernel <<<num_faces, 1 >>> (
				Energy3.cuda_arr,
				restShapeF.cuda_arr,
				CurrV.cuda_arr,
				CurrN.cuda_arr,
				num_faces);
			if (cudaDeviceSynchronize() != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code after launching addKernel!\n");
				exit(1);
			}
			MemCpyDeviceToHost(Energy3);
			
			double T1 = 0, T2 = 0, T3 = 0;
			for (int i = 0; i < Energy1.size; i++)
				T1 += Energy1.host_arr[i];
			for (int i = 0; i < Energy2.size; i++)
				T2 += Energy2.host_arr[i];
			for (int i = 0; i < Energy3.size; i++)
				T3 += Energy3.host_arr[i];

			double value =
				Cuda::AuxBendingNormal::w1 * T1 +
				Cuda::AuxBendingNormal::w2 * T2 +
				Cuda::AuxBendingNormal::w3 * T3;
			return value;
		}


		void updateX() {
			MemCpyHostToDevice(CurrV);
			MemCpyHostToDevice(CurrN);
			updateXKernel <<<num_hinges, 1>>> (
				d_normals.cuda_arr, 
				CurrN.cuda_arr, 
				hinges_faceIndex.cuda_arr, 
				num_hinges);
			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			if (cudaDeviceSynchronize() != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code after launching addKernel!\n");
				exit(1);
			}

			////For Debugging...
			//MemCpyDeviceToHost(d_normals);
			//for (int hi = 0; hi < num_hinges; hi++) {
			//	int f0 = hinges_faceIndex.host_arr[hi].f0;
			//	int f1 = hinges_faceIndex.host_arr[hi].f1;
			//	double diffX = CurrN.host_arr[f1].x - CurrN.host_arr[f0].x;
			//	double diffY = CurrN.host_arr[f1].y - CurrN.host_arr[f0].y;
			//	double diffZ = CurrN.host_arr[f1].z - CurrN.host_arr[f0].z;
			//	double expected = diffX * diffX + diffY * diffY + diffZ * diffZ;
			//	double epsilon = 1e-3;
			//	double diff = d_normals.host_arr[hi] - expected;
			//	if (diff > epsilon || diff < -epsilon) {
			//		std::cout << "Error at index" << hi << std::endl;
			//		std::cout << "Expected = " << expected << std::endl;
			//		std::cout << "d_normals.host_arr[hi] = " << d_normals.host_arr[hi] << std::endl;
			//		std::cout << "diff = " << diff << std::endl;
			//		exit(1);
			//	}
			//	else {
			//		std::cout << "okay!\n";
			//	}
			//}
		}

		void FreeAllVariables() {
			cudaGetErrorString(cudaGetLastError());
			FreeMemory(restShapeF);
			FreeMemory(CurrV);
			FreeMemory(CurrN);
			FreeMemory(restAreaPerFace);
			FreeMemory(restAreaPerHinge);
			FreeMemory(d_normals);
			FreeMemory(Energy1);
			FreeMemory(Energy2);
			FreeMemory(Energy3);
			FreeMemory(hinges_faceIndex);
			FreeMemory(x0_GlobInd);
			FreeMemory(x1_GlobInd);
			FreeMemory(x2_GlobInd);
			FreeMemory(x3_GlobInd);
			FreeMemory(x0_LocInd);
			FreeMemory(x1_LocInd);
			FreeMemory(x2_LocInd);
			FreeMemory(x3_LocInd);
		}
	}
}
