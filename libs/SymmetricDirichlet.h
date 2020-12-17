#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_Minimizer.cuh"
#include "cuda_optimization_lib/Cuda_Basics.cuh"

class SymmetricDirichlet : public ObjectiveFunction {
private:
	Eigen::VectorXd a;
	Eigen::VectorXd b;
	Eigen::VectorXd c;
	Eigen::VectorXd d;
	Eigen::VectorXd detJ;
	Eigen::MatrixX3d B1, B2;
	Eigen::Matrix3Xd D1d, D2d;
	Eigen::VectorXd restShapeArea;
	Eigen::MatrixX3d CurrV;
	Eigen::Matrix<double, 3, 9> dB1_dX(int fi);
	Eigen::Matrix<double, 3, 9> dB2_dX(int fi);
	Eigen::Matrix<double, 4, 9> dJ_dX(int fi);
	Eigen::Matrix<double, 1, 4> dE_dJ(int fi);

	//sets important properties of the rest shape using the set of points passed in as parameters
	void setRestShapeFromCurrentConfiguration();
public:
	SymmetricDirichlet();
	~SymmetricDirichlet();
	virtual void init();
	virtual void updateX(Cuda::Array<double>& curr_x);
	virtual double value(Cuda::Array<double>& curr_x, const bool update);
	virtual void gradient(Cuda::Array<double>& X, Eigen::VectorXd& g, const bool update);
};