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
	double f=0;
	for (auto &obj : objectiveList)
		if (obj->w != 0)
			f += obj->w * obj->value(curr_x, update);
	if (update)
		energy_value = f;
	return f;
}

void TotalObjective::gradient(
	std::shared_ptr<Cuda_Minimizer> cuda_Minimizer,
	Cuda::Array<double>& X, 
	const bool update)
{
	
	for (auto& obj : objectiveList) {
		if (obj->w) {
			Cuda::Array<double>* obj_grad = obj->gradient(X, update);
		}
	}
	std::shared_ptr<AuxSpherePerHinge> ASH = std::dynamic_pointer_cast<AuxSpherePerHinge>(objectiveList[0]);
	std::shared_ptr<AuxBendingNormal> ABN = std::dynamic_pointer_cast<AuxBendingNormal>(objectiveList[1]);
	std::shared_ptr<STVK> stvk = std::dynamic_pointer_cast<STVK>(objectiveList[2]);
	std::shared_ptr<FixAllVertices> FAV = std::dynamic_pointer_cast<FixAllVertices>(objectiveList[3]);
	std::shared_ptr<FixChosenConstraints> FCV = std::dynamic_pointer_cast<FixChosenConstraints>(objectiveList[4]);
	std::shared_ptr<FixChosenConstraints> FCN = std::dynamic_pointer_cast<FixChosenConstraints>(objectiveList[5]);
	std::shared_ptr<FixChosenConstraints> FCC = std::dynamic_pointer_cast<FixChosenConstraints>(objectiveList[6]);
	std::shared_ptr<Grouping> GroS = std::dynamic_pointer_cast<Grouping>(objectiveList[7]);
	std::shared_ptr<Grouping> GroN = std::dynamic_pointer_cast<Grouping>(objectiveList[8]);

	cuda_Minimizer->TotalGradient(
		ABN->cuda_ABN->grad.cuda_arr, ABN->w,//AuxBendingNormal
		FAV->cuda_FixAllV->grad.cuda_arr, FAV->w,//FixAllVertices
		ASH->cuda_ASH->grad.cuda_arr, ASH->w,//AuxSpherePerHinge
		FCV->Cuda_FixChosConst->grad.cuda_arr, FCV->w,//FixChosenVertices
		FCN->Cuda_FixChosConst->grad.cuda_arr, FCN->w,//FixChosenVertices
		FCC->Cuda_FixChosConst->grad.cuda_arr, FCC->w,//FixChosenVertices
		GroN->cudaGrouping->grad.cuda_arr, GroN->w,//GroupNormals
		GroS->cudaGrouping->grad.cuda_arr, GroS->w,//GroupSpheres
		stvk->cuda_STVK->grad.cuda_arr, stvk->w
	);
			
	/*if(update)
		gradient_norm = g.norm();*/
}