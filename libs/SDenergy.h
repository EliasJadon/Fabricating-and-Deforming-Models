#pragma once
#include "ObjectiveFunction.h"
#include "cuda_optimization_lib/cuda_SDenergy.cuh"

class SDenergy : public ObjectiveFunction {
private:
	Eigen::Matrix3Xd D1d, D2d;
	Eigen::Matrix<double, 3, 9> dB1_dX(int fi, const Eigen::RowVector3d e10);
	Eigen::Matrix<double, 3, 9> dB2_dX(int fi, const Eigen::RowVector3d e10, const Eigen::RowVector3d e20);
	Eigen::Matrix<double, 4, 9> dJ_dX(
		int fi, 
		const Eigen::RowVector3d V0,
		const Eigen::RowVector3d V1,
		const Eigen::RowVector3d V2);
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
};