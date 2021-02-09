#include "fixRadius.h"

fixRadius::fixRadius(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) 
{
	init_mesh(V, F);
	name = "fix Radius";
	w = 0;
	Cuda::AllocateMemory(grad, (3 * V.rows()) + (10 * F.rows()));
	Cuda::AllocateMemory(EnergyAtomic, 1);
	Cuda::initIndices(mesh_indices, F.rows(), V.rows(), 0);
	Efi.resize(F.rows());
	Efi.setZero();
	std::cout << "\t" << name << " constructor" << std::endl;
}

fixRadius::~fixRadius() {
	FreeMemory(grad);
	FreeMemory(EnergyAtomic);
	std::cout << "\t" << name << " destructor" << std::endl;
}

void fixRadius::value(Cuda::Array<double>& curr_x) {
	Cuda::MemCpyDeviceToHost(curr_x);
	EnergyAtomic.host_arr[0] = 0;
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const int startR = mesh_indices.startR;
		double R = curr_x.host_arr[fi + startR];
		double rounded_R = round(factor * R) / (double)factor;
		EnergyAtomic.host_arr[0] += pow(R - rounded_R, 2);
	}
	Cuda::MemCpyHostToDevice(EnergyAtomic);
}

void fixRadius::gradient(Cuda::Array<double>& X)
{
	Cuda::MemCpyDeviceToHost(X);
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const int startR = mesh_indices.startR;
		double R = X.host_arr[fi + startR];
		double rounded_R = round(factor * R) / (double)factor;
		grad.host_arr[fi + startR] += 2 * (R - rounded_R);;
	}
	Cuda::MemCpyHostToDevice(grad);
}
