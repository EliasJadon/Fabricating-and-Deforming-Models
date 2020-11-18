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
		Array<rowVector> CurrV, CurrN; //Eigen::MatrixX3d
		Array<double> d_normals;
		//help variables - dynamic
		Array<double> Energy1, Energy2, Energy3;
		
		//Static variables
		Array<double> restAreaPerFace, restAreaPerHinge; //Eigen::VectorXd
		int num_hinges, num_faces, num_vertices;
		Array<hinge> hinges_faceIndex; //std::vector<Eigen::Vector2d> //num_hinges*2
		Array<int> x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd; //Eigen::VectorXi //num_hinges
		Array<hinge> x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd; //Eigen::MatrixXi //num_hinges*2
		
		__global__ void updateXKernel(
			double* d_normals, 
			const rowVector* Normals, 
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

		void value() {
			////per hinge
			//Eigen::VectorXd Energy1 = Phi(d_normals);

			////per face
			//double Energy2 = 0; // (||N||^2 - 1)^2
			//for (int fi = 0; fi < restShapeF.rows(); fi++) {
			//	Energy2 += pow(CurrN.row(fi).squaredNorm() - 1, 2);
			//}

			//double Energy3 = 0; // (N^T*(x1-x0))^2 + (N^T*(x2-x1))^2 + (N^T*(x0-x2))^2
			//for (int fi = 0; fi < restShapeF.rows(); fi++) {
			//	int x0 = restShapeF(fi, 0);
			//	int x1 = restShapeF(fi, 1);
			//	int x2 = restShapeF(fi, 2);
			//	Eigen::VectorXd e21 = CurrV.row(x2) - CurrV.row(x1);
			//	Eigen::VectorXd e10 = CurrV.row(x1) - CurrV.row(x0);
			//	Eigen::VectorXd e02 = CurrV.row(x0) - CurrV.row(x2);
			//	Energy3 += pow(CurrN.row(fi) * e21, 2);
			//	Energy3 += pow(CurrN.row(fi) * e10, 2);
			//	Energy3 += pow(CurrN.row(fi) * e02, 2);
			//}

			//double value =
			//	Cuda::AuxBendingNormal::w1 * Energy1.transpose() * restAreaPerHinge +
			//	Cuda::AuxBendingNormal::w2 * Energy2 +
			//	Cuda::AuxBendingNormal::w3 * Energy3;
		}
		void updateX() {
			MemCpyHostToDevice(CurrV);
			MemCpyHostToDevice(CurrN);

			// Launch a kernel on the GPU with one thread for each element.
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

		void Device_freeMemory() {
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
