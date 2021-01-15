#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/cuda_SDenergy.cuh"

class SDenergy : public ObjectiveFunction {
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
public:
	std::shared_ptr<Cuda_SDenergy> cuda_SD;
	SDenergy(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
	~SDenergy();
	
	virtual Cuda::Array<double>* getValue() override {
		return &(cuda_SD->EnergyAtomic);
	}
	virtual Cuda::Array<double>* getGradient() override {
		return &(cuda_SD->grad);
	}
	virtual void value(Cuda::Array<double>& curr_x);
	virtual void gradient(Cuda::Array<double>& X);
	void updateX(Cuda::Array<double>& curr_x);
};