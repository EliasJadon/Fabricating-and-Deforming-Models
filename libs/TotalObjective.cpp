#include "TotalObjective.h"

void TotalObjective::FDGradient(const Cuda::Array<double>& X, Cuda::Array<double>& grad)
{
	Cuda::Array<double> Xd;
	Cuda::AllocateMemory(grad, X.size);
	Cuda::AllocateMemory(Xd, X.size);
	for (int i = 0; i < X.size; i++) {
		Xd.host_arr[i] = X.host_arr[i];
	}
	Cuda::MemCpyHostToDevice(Xd);
	const double dX = 1e-4;
	double f_P, f_M;

	//this is a very slow method that evaluates the gradient of the objective function through FD...
	Cuda::Array<double>* E;
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

void TotalObjective::checkGradient(const Eigen::VectorXd& X)
{
	Cuda::Array<double> XX;
	Cuda::AllocateMemory(XX, X.size());
	for (int i = 0; i < XX.size; i++) {
		XX.host_arr[i] = X(i);
	}
	Cuda::MemCpyHostToDevice(XX);

	gradient(XX,false);
	Cuda::CheckErr(cudaDeviceSynchronize());
	Cuda::Array<double>* Analytic_gradient = &(cuda_Minimizer->g);
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
		if (relError > tol && absErr > 1e-5) {
			std::cout << name << "\t" << i << ":\tAnalytic val: " <<
				Analytic_gradient->host_arr[i] << ", FD val: " << FD_gradient.host_arr[i] <<
				". Error: " << absErr << "(" << relError * 100 << "%%)\n";
		}
	}
}

TotalObjective::TotalObjective()
{
	name = "Total objective";
	std::cout << "\t" << name << " constructor" << std::endl;
}

TotalObjective::~TotalObjective()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

double TotalObjective::value(Cuda::Array<double>& curr_x, const bool update)
{
	for (auto& obj : objectiveList)
		if (obj->w != 0)
			obj->value(curr_x);
	
	Cuda::CheckErr(cudaDeviceSynchronize());

	double f = 0;
	for (auto& obj : objectiveList)
		if (obj->w != 0) {
			Cuda::Array<double>* E = obj->getValue();
			Cuda::MemCpyDeviceToHost(*E);
			if (update)
				obj->energy_value = E->host_arr[0];
			f += obj->w * E->host_arr[0];
		}
	
	if (update) {
		energy_value = f;
		//update face coloring for STVK & Symmetric-Dirichlet
		std::shared_ptr<STVK> stvk = std::dynamic_pointer_cast<STVK>(objectiveList[2]);
		std::shared_ptr<SDenergy> SD = std::dynamic_pointer_cast<SDenergy>(objectiveList[3]);
		Cuda::MemCpyDeviceToHost(stvk->cuda_STVK->Energy);
		Cuda::MemCpyDeviceToHost(SD->cuda_SD->Energy);
		for (int i = 0; i < stvk->cuda_STVK->Energy.size; i++)
			stvk->Efi(i) = stvk->cuda_STVK->Energy.host_arr[i];
		for (int i = 0; i < SD->cuda_SD->Energy.size; i++)
			SD->Efi(i) = SD->cuda_SD->Energy.host_arr[i];
	}
	return f;
}

void TotalObjective::gradient(Cuda::Array<double>& X, const bool update)
{
	for (auto& obj : objectiveList)
		if (obj->w)
			obj->gradient(X);	
	Cuda::CheckErr(cudaDeviceSynchronize());
	cuda_Minimizer->TotalGradient(
		objectiveList[0]->getGradient()->cuda_arr, objectiveList[0]->w,
		objectiveList[1]->getGradient()->cuda_arr, objectiveList[1]->w,
		objectiveList[2]->getGradient()->cuda_arr, objectiveList[2]->w,
		objectiveList[3]->getGradient()->cuda_arr, objectiveList[3]->w,
		objectiveList[4]->getGradient()->cuda_arr, objectiveList[4]->w,
		objectiveList[5]->getGradient()->cuda_arr, objectiveList[5]->w,
		objectiveList[6]->getGradient()->cuda_arr, objectiveList[6]->w,
		objectiveList[7]->getGradient()->cuda_arr, objectiveList[7]->w
	);
		
	//TODO: update the gradient norm for all the energies
	/*if(update)
		gradient_norm = g.norm();*/
}