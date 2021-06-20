#pragma once
#include "OptimizationUtils.h"
#include "..//..//plugins/console_color.h"
#include "cuda_optimization_lib/Cuda_Basics.cuh"

class ObjectiveFunction
{
public:

	double dot4(const double4 a, const double4 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
	}

	template<int N> void multiply(
		double3 mat1,
		double mat2[3][N],
		double res[N])
	{
		for (int i = 0; i < N; i++) {
			res[i] = mat1.x * mat2[0][i] + mat1.y * mat2[1][i] + mat1.z * mat2[2][i];
		}
	}
	template<int N> void multiply(
		double4 mat1,
		double mat2[4][N],
		double res[N])
	{
		for (int i = 0; i < N; i++) {
			res[i] =
				mat1.x * mat2[0][i] +
				mat1.y * mat2[1][i] +
				mat1.z * mat2[2][i] +
				mat1.w * mat2[3][i];
		}
	}

	double Phi(
		const double x,
		const double SigmoidParameter,
		const Cuda::PenaltyFunction penaltyFunction)
	{
		if (penaltyFunction == Cuda::PenaltyFunction::SIGMOID) {
			double x2 = pow(x, 2);
			return x2 / (x2 + SigmoidParameter);
		}
		if (penaltyFunction == Cuda::PenaltyFunction::QUADRATIC)
			return pow(x, 2);
		if (penaltyFunction == Cuda::PenaltyFunction::EXPONENTIAL)
			return exp(x * x);
	}
	double3 sub(const double3 a, const double3 b)
	{
		return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
	}
	double3 sub(const double3 a, const Eigen::RowVector3d b)
	{
		return make_double3(a.x - b(0), a.y - b(1), a.z - b(2));
	}
	double3 add(double3 a, double3 b)
	{
		return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
	}
	double dot(const double3 a, const double3 b)
	{
		return a.x * b.x + a.y * b.y + a.z * b.z;
	}
	double3 mul(const double a, const double3 b)
	{
		return make_double3(a * b.x, a * b.y, a * b.z);
	}
	double squared_norm(const double3 a)
	{
		return dot(a, a);
	}
	double norm(const double3 a)
	{
		return sqrt(squared_norm(a));
	}
	double3 normalize(const double3 a)
	{
		return mul(1.0f / norm(a), a);
	}
	double3 cross(const double3 a, const double3 b)
	{
		return make_double3(
			a.y * b.z - a.z * b.y,
			a.z * b.x - a.x * b.z,
			a.x * b.y - a.y * b.x
		);
	}
	double dPhi_dm(
		const double x,
		const double SigmoidParameter,
		const Cuda::PenaltyFunction penaltyFunction)
	{
		if (penaltyFunction == Cuda::PenaltyFunction::SIGMOID)
			return (2 * x * SigmoidParameter) / pow(x * x + SigmoidParameter, 2);
		if (penaltyFunction == Cuda::PenaltyFunction::QUADRATIC)
			return 2 * x;
		if (penaltyFunction == Cuda::PenaltyFunction::EXPONENTIAL)
			return 2 * x * exp(x * x);
	}

	inline double3 getN(const Cuda::Array<double>& X, const int fi) {
		return make_double3(
			X.host_arr[fi + mesh_indices.startNx],
			X.host_arr[fi + mesh_indices.startNy],
			X.host_arr[fi + mesh_indices.startNz]
		);
	}

	inline double3 getC(const Cuda::Array<double>& X, const int fi) {
		return make_double3(
			X.host_arr[fi + mesh_indices.startCx],
			X.host_arr[fi + mesh_indices.startCy],
			X.host_arr[fi + mesh_indices.startCz]
		);
	}
	inline double getR(const Cuda::Array<double>& X, const int fi) {
		return X.host_arr[fi + mesh_indices.startR];
	}

	inline double3 getV(const Cuda::Array<double>& X, const int vi) {
		return make_double3(
			X.host_arr[vi + mesh_indices.startVx],
			X.host_arr[vi + mesh_indices.startVy],
			X.host_arr[vi + mesh_indices.startVz]
		);
	}

public:
	ObjectiveFunction() {
	
	}
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

