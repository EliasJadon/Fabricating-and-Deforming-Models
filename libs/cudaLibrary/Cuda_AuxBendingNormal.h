#pragma once
#include "CudaBasics.h"
namespace Cuda {
	namespace AuxBendingNormal {
		template <typename T>
		void Host_allocateMem(Array<T> a);
		template <typename T>
		void Cuda_allocateMem(Array<T> a);
		template <typename T>
		void Cuda_MemcpyHostToDevice(Array<T> a);
		void init(int numvertices, int numfaces, int numhinges);
		void Host_freeMemory();
		void Cuda_freeMemory();

		//Dynamic variables
		double w1, w2, w3;
		int functionType; //OptimizationUtils::FunctionType
		double planarParameter;
		Array<rowVector> CurrV, CurrN; //Eigen::MatrixX3d
		Array<double> d_normals;

		//Static variables
		Array<double> restAreaPerFace, restAreaPerHinge; //Eigen::VectorXd
		int num_hinges, num_faces, num_vertices;
		Array<hinge> hinges_faceIndex; //std::vector<Eigen::Vector2d> //num_hinges*2
		Array<int> x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd; //Eigen::VectorXi //num_hinges
		Array<hinge> x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd; //Eigen::MatrixXi //num_hinges*2
	}
}