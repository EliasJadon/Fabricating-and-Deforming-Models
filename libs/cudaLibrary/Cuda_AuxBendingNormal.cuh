#pragma once
#include "CudaBasics.cuh"


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
		extern void gradient();
		extern void FreeAllVariables();		
	}
}