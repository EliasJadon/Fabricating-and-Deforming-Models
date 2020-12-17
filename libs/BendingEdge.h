#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/Cuda_Basics.cuh"

/******************************************************
 Element that defines a preferred angle between 2 triangles
 4 nodes are required and not all of them are equal:

  nodes 0 and 1 are on the actual edge
  nodes 2 and 3 are the other triangle corners

	0 - 2
   / \ /
  3 - 1

******************************************************/

class BendingEdge : public ObjectiveFunction
{	
private:
	//material parameters...
	double k = 0.05;
	//keep track of the rest shape
	Eigen::VectorXd restAngle, restEdgeLength, restArea, restConst;
	int num_hinges = -1;
	Eigen::VectorXi x0_index, x1_index, x2_index, x3_index;
	Eigen::MatrixX3d x0, x1, x2, x3;
	Eigen::VectorXd angle;

	//sets important properties of the rest shape using the set of points passed in as parameters
	void calculateHinges();
	void setRestShapeFromCurrentConfiguration();
	
	//update angle
	void getAngle();

	Eigen::VectorXd AngleFunctionvalue(Eigen::VectorXd angleDifference);
	Eigen::VectorXd dF_d0(Eigen::VectorXd angleDifference);
	Eigen::VectorXd d2F_d0d0(Eigen::VectorXd angleDifference);
	Eigen::Matrix< Eigen::Matrix3d, 4, 4> d20_dxdx(int hi);
	Eigen::Matrix<double, 4, 3> d0_dx(int hi);
	int BendingEdge::x_index(int i, int hi);
public:
	FunctionType functionType;
	float planarParameter;

	BendingEdge(FunctionType type);
	~BendingEdge();
	virtual void init() override;
	virtual void updateX(Cuda::Array<double>& curr_x) override;
	virtual double value(Cuda::Array<double>& curr_x,const bool update) override;
	virtual void gradient(Cuda::Array<double>& X,Eigen::VectorXd& g, const bool update) override;
};

