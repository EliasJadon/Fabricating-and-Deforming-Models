#pragma once
#include "OptimizationUtils.h"
#include "..//..//plugins/console_color.h"
#include "cuda_optimization_lib/Cuda_Basics.cuh"

class ObjectiveFunction
{
public:
	// mesh vertices and faces
	Eigen::MatrixX3i restShapeF;
	Eigen::MatrixX3d restShapeV;
public:
	ObjectiveFunction() {}
	virtual ~ObjectiveFunction(){}
	virtual Cuda::Array<double>* getValue() = 0;
	virtual Cuda::Array<double>* getGradient() = 0;
	virtual void value(Cuda::Array<double>& curr_x) = 0;
	virtual void gradient(Cuda::Array<double>& X) = 0;
	
	void init_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);

	//Finite Differences check point
	void FDGradient(const Cuda::Array<double>& X, Cuda::Array<double>& grad);
    void checkGradient(const Eigen::VectorXd& X);
    
	//weight for each objective function
	float w = 0;
	Eigen::VectorXd Efi;
	double energy_value = 0;
	double gradient_norm = 0;
	std::string name = "Objective function";
};

