#pragma once
#include "Cuda_Basics.cuh"

namespace Cuda {
	namespace AuxSpherePerHinge {
		//Dynamic variables
		extern double w1, w2;
		extern FunctionType functionType; //OptimizationUtils::FunctionType /*QUADRATIC = 0,EXPONENTIAL = 1,SIGMOID = 2*/
		extern double planarParameter;
		extern Array<double> grad;
		extern Array<double> EnergyAtomic;

		//Static variables
		extern Array<int3> restShapeF;
		extern Array<double> restAreaPerFace, restAreaPerHinge; //Eigen::VectorXd
		extern indices mesh_indices;
		extern Array<hinge> hinges_faceIndex; //std::vector<Eigen::Vector2d> //num_hinges*2
		extern Array<int> x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd; //Eigen::VectorXi //num_hinges
		extern Array<hinge> x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd; //Eigen::MatrixXi //num_hinges*2

		extern double value(Cuda::Array<double>& curr_x);
		extern void gradient(Cuda::Array<double>& X);
		extern void FreeAllVariables();
	}
}