#include "ObjectiveFunction.h"

void ObjectiveFunction::init_mesh(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) {
	Eigen::MatrixX3d V3d(V.rows(), 3);
	if (V.cols() == 2) {
		V3d.leftCols(2) = V;
		V3d.col(2).setZero();
	}
	else if (V.cols() == 3) {
		V3d = V;
	}
	this->restShapeV = V3d;
	this->restShapeF = F;
}

void ObjectiveFunction::FDGradient(const Cuda::Array<double>& X, Cuda::Array<double>& grad)
{
	Cuda::Array<double> Xd;
	Cuda::AllocateMemory(grad, X.size);
	Cuda::AllocateMemory(Xd, X.size);
	for (int i = 0; i < X.size; i++) {
		Xd.host_arr[i] = X.host_arr[i];
	}
	Cuda::MemCpyHostToDevice(Xd);
	const double dX = 10e-6;
	double f_P, f_M;

    //this is a very slow method that evaluates the gradient of the objective function through FD...
    for (int i = 0; i < X.size; i++) {
        Xd.host_arr[i] = X.host_arr[i] + dX;
		Cuda::MemCpyHostToDevice(Xd);
        f_P = value(Xd, false);

        Xd.host_arr[i] = X.host_arr[i] - dX;
		Cuda::MemCpyHostToDevice(Xd); 
		f_M = value(Xd, false);
		
        //now reset the ith param value
        Xd.host_arr[i] = X.host_arr[i];
        grad.host_arr[i] = (f_P - f_M) / (2 * dX);
    }
}

void ObjectiveFunction::checkGradient(const Eigen::VectorXd& X)
{
	Cuda::Array<double> XX;
	Cuda::AllocateMemory(XX, X.size());
	for (int i = 0; i < XX.size; i++) {
		XX.host_arr[i] = X(i);
	}
	Cuda::MemCpyHostToDevice(XX);

	
	Cuda::Array<double>* Analytic_gradient = gradient(XX, false);
	if (Analytic_gradient == NULL)
	{
		return;
	}
	Cuda::MemCpyDeviceToHost(*Analytic_gradient);
	Cuda::Array<double> FD_gradient;
	FDGradient(XX, FD_gradient);
	assert(FD_gradient.size == Analytic_gradient->size && "The size of analytic gradient & FD gradient must be equal!");
	double tol = 1e-4;
	double eps = 1e-10;
	

	double Analytic_gradient_norm = 0;
	double FD_gradient_norm = 0;
	for (int i = 0; i < XX.size; i++) {
		Analytic_gradient_norm += pow(Analytic_gradient->host_arr[i], 2);
		FD_gradient_norm += pow(FD_gradient.host_arr[i], 2);
	}
	//std::cout << "g= " << Analytic_gradient  << std::endl;
	//std::cout << "FD= " << FD_gradient << std::endl;
	
	std::cout << name << ": g.norm() = " << Analytic_gradient_norm << "(analytic) , " << FD_gradient_norm << "(FD)" << std::endl;
	for (int i = 0; i < Analytic_gradient->size; i++) {
        double absErr = abs(FD_gradient.host_arr[i] - Analytic_gradient->host_arr[i]);
        double relError = 2 * absErr / (eps + Analytic_gradient->host_arr[i] + FD_gradient.host_arr[i]);
        if (relError > tol && absErr > 1e-6) {
			std::cout << name << "\t" << i << ":\tAnalytic val: " <<
				Analytic_gradient->host_arr[i] << ", FD val: " << FD_gradient.host_arr[i] <<
				". Error: " << absErr << "(" << relError * 100 << "%%)\n";
        }
    }
}
