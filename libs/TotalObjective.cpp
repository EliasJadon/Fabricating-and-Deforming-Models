#include "TotalObjective.h"

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

		//update face coloring for STVK
		std::shared_ptr<STVK> stvk = std::dynamic_pointer_cast<STVK>(objectiveList[2]);
		Cuda::MemCpyDeviceToHost(stvk->cuda_STVK->Energy);
		for (int i = 0; i < stvk->cuda_STVK->Energy.size; i++) {
			stvk->Efi(i) = stvk->cuda_STVK->Energy.host_arr[i];
		}
	}
	return f;
}

void TotalObjective::gradient(
	std::shared_ptr<Cuda_Minimizer> cuda_Minimizer,
	Cuda::Array<double>& X, 
	const bool update)
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
		objectiveList[7]->getGradient()->cuda_arr, objectiveList[7]->w,
		objectiveList[8]->getGradient()->cuda_arr, objectiveList[8]->w
	);
		
	//TODO: update the gradient norm for all the energies
	/*if(update)
		gradient_norm = g.norm();*/
}