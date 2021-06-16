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
		double val;
		if (R < (min / alpha))
			val = pow(alpha * R - min, 2);
		if (R > (max / alpha))
			val = pow(alpha * R - max, 2);
		else {
			//val = pow(sin(alpha * M_PI * R), 2);
			double rounded_R = round(alpha * R) / (double)alpha;
			val = pow(R - rounded_R, 2);
		}
			
		
		EnergyAtomic.host_arr[0] += val;
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

		double val = 0;
		if (R < (min / alpha))
			val = 2 * alpha * (alpha * R - min);
		if (R > (max / alpha))
			val = 2 * alpha * (alpha * R - max);
		else {
			//val = alpha * M_PI * sin(2 * alpha * M_PI * R);

			double rounded_R = round(alpha * R) / (double)alpha;
			val = 2 * (R - rounded_R);
		}
		
		grad.host_arr[fi + startR] += val;
	}
	Cuda::MemCpyHostToDevice(grad);
}
