#pragma once
#include "OptimizationUtils.h"
#include "..//..//plugins/console_color.h"
#include "cuda_optimization_lib/Cuda_Basics.cuh"

class ObjectiveFunction
{
public:
	inline double_3 getN(const Cuda::Array<double>& X, const int fi) {
		return double_3(
			X.host_arr[fi + mesh_indices.startNx],
			X.host_arr[fi + mesh_indices.startNy],
			X.host_arr[fi + mesh_indices.startNz]
		);
	}

	inline double_3 getC(const Cuda::Array<double>& X, const int fi) {
		return double_3(
			X.host_arr[fi + mesh_indices.startCx],
			X.host_arr[fi + mesh_indices.startCy],
			X.host_arr[fi + mesh_indices.startCz]
		);
	}
	inline double getR(const Cuda::Array<double>& X, const int fi) {
		return X.host_arr[fi + mesh_indices.startR];
	}

	inline double_3 getV(const Cuda::Array<double>& X, const int vi) {
		return double_3(
			X.host_arr[vi + mesh_indices.startVx],
			X.host_arr[vi + mesh_indices.startVy],
			X.host_arr[vi + mesh_indices.startVz]
		);
	}

	ObjectiveFunction() {}
	~ObjectiveFunction(){
		Cuda::FreeMemory(grad);
	}
	virtual double value(Cuda::Array<double>& curr_x, const bool update) = 0;
	virtual void gradient(Cuda::Array<double>& X, const bool update) = 0;
	void init_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);

	//Finite Differences check point
	void FDGradient(const Cuda::Array<double>& X, Cuda::Array<double>& grad);
    void checkGradient(const Eigen::VectorXd& X);
    
	//weight for each objective function
	float w = 0;
	Eigen::VectorXd Efi;
	Cuda::Array<double> grad;
	double energy_value = 0;
	double gradient_norm = 0;
	std::string name = "Objective function";
	Cuda::indices mesh_indices;
	Eigen::MatrixX3i restShapeF;
	Eigen::MatrixX3d restShapeV;
};

