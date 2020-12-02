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

void TotalObjective::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	variables_size = 3 * restShapeV.rows() + 7 * restShapeF.rows();
	//assume that each objective's member have been set outside
	for (auto &objective : objectiveList)
		objective->init();
//	for (auto& objective : objectiveList)
	//	objective->updateX(Eigen::VectorXd::Random(variables_size));
	init_hessian();
}

void TotalObjective::updateX(const Eigen::VectorXd& X)
{
	for (auto& objective : objectiveList)
		if (objective->w != 0)
			objective->updateX(X);
}

double TotalObjective::value(const bool update)
{
	double f=0;
	for (auto &obj : objectiveList)
		if (obj->w != 0)
			f += obj->w * obj->value(update);

	if (update)
		energy_value = f;
	return f;
}

void TotalObjective::gradient(Eigen::VectorXd& g, const bool update)
{
	Cuda::AuxBendingNormal::gradient();
	Cuda::FixAllVertices::gradient();
	Cuda::Minimizer::TotalGradient(
		objectiveList[1]->w,	//AuxBendingNormal
		objectiveList[5]->w		//FixAllVertices
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

void TotalObjective::hessian()
{
	II.clear(); JJ.clear(); SS.clear();
	
	for (auto const &objective : objectiveList)
	{
		if (objective->w != 0) {
			objective->hessian();
			std::vector<double> SSi; SSi.resize(objective->SS.size());
			for (int i = 0; i < objective->SS.size(); i++)
				SSi[i] = objective->w * objective->SS[i];

			SS.insert(SS.end(), SSi.begin(), SSi.end());
			II.insert(II.end(), objective->II.begin(), objective->II.end());
			JJ.insert(JJ.end(), objective->JJ.begin(), objective->JJ.end());
		}
	}

	// shift the diagonal of the hessian
	for (int i = 0; i < variables_size; i++) {
		II.push_back(i);
		JJ.push_back(i);
		SS.push_back(1e-6 + Shift_eigen_values);
	}
	assert(SS.size() == II.size() && SS.size() == JJ.size());
}

void TotalObjective::init_hessian()
{
	
}

