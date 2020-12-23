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
	if (objectiveList[0]->w)
		Cuda::AuxSpherePerHinge::gradient(X);

	if (objectiveList[1]->w)
		Cuda::AuxBendingNormal::gradient(X);

	if (objectiveList[2]->w) // FixAllVertices
		objectiveList[2]->gradient(X, true);
	
	if (objectiveList[3]->w) //FixChosenVertices
		objectiveList[3]->gradient(X, true);

	if (objectiveList[4]->w) //FixChosenNormals
		objectiveList[4]->gradient(X, true);

	if (objectiveList[5]->w) //FixChosenCenters
		objectiveList[5]->gradient(X, true);

	if (objectiveList[7]->w) //GroupNormals
		objectiveList[7]->gradient(X, true);
		
	
	std::shared_ptr<FixChosenConstraints> FCV = std::dynamic_pointer_cast<FixChosenConstraints>(objectiveList[3]);
	std::shared_ptr<FixChosenConstraints> FCN = std::dynamic_pointer_cast<FixChosenConstraints>(objectiveList[4]);
	std::shared_ptr<FixChosenConstraints> FCC = std::dynamic_pointer_cast<FixChosenConstraints>(objectiveList[5]);
	std::shared_ptr<FixAllVertices> FAV = std::dynamic_pointer_cast<FixAllVertices>(objectiveList[2]);
	std::shared_ptr<GroupNormals> GroN = std::dynamic_pointer_cast<GroupNormals>(objectiveList[7]);

	cuda_Minimizer->TotalGradient(
		Cuda::AuxBendingNormal::grad.cuda_arr, objectiveList[1]->w,//AuxBendingNormal
		FAV->cuda_FixAllV->grad.cuda_arr, FAV->w,//FixAllVertices
		Cuda::AuxSpherePerHinge::grad.cuda_arr, objectiveList[0]->w,//AuxSpherePerHinge
		FCV->Cuda_FixChosConst->grad.cuda_arr, FCV->w,		//FixChosenVertices
		FCN->Cuda_FixChosConst->grad.cuda_arr, FCN->w,		//FixChosenVertices
		FCC->Cuda_FixChosConst->grad.cuda_arr, FCC->w,		//FixChosenVertices
		GroN->grad.cuda_arr, GroN->w		//GroupNormals
	);
			
	/*g.setZero(variables_size);
	for (auto &objective : objectiveList) {
		if (objective->w != 0)
		{
			Eigen::VectorXd gi;
			objective->gradient(gi, update);
			g += objective->w*gi;
		}
	}

	if(update)
		gradient_norm = g.norm();*/
}