#include "objective_functions/FixChosenSpheres.h"

FixChosenSpheres::FixChosenSpheres()
{
    name = "Fix Chosen Spheres";
	w = 0.05;
	std::cout << "\t" << name << " constructor" << std::endl;
}

FixChosenSpheres::~FixChosenSpheres()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void FixChosenSpheres::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if(numV==0 || numF == 0)
		throw name + " must define members numV & numF before init()!";
	init_hessian();
}

void FixChosenSpheres::updateX(const Eigen::VectorXd& X)
{
	CurrConstrainedCentersPos.resizeLike(ConstrainedCentersPos);
	for (int i = 0; i < ConstrainedCentersInd.size(); i++)
	{
		int startC = 3 * numV + 3 * numF;
		CurrConstrainedCentersPos.row(i) <<
			X(ConstrainedCentersInd[i] + (0 * numF) + startC),	//X-coordinate
			X(ConstrainedCentersInd[i] + (1 * numF) + startC),	//Y-coordinate
			X(ConstrainedCentersInd[i] + (2 * numF) + startC);	//Z-coordinate
	}
}

double FixChosenSpheres::value(const bool update)
{
	if (CurrConstrainedCentersPos.rows() != ConstrainedCentersPos.rows()) {
		return 0;
	}
	double E = (ConstrainedCentersPos - CurrConstrainedCentersPos).squaredNorm();
	if (update) {
		energy_value = E;
	}
	return E;
}

void FixChosenSpheres::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(numV * 3 + numF * 7);
	g.setZero();

	if (CurrConstrainedCentersPos.rows() == ConstrainedCentersPos.rows()) {
		Eigen::MatrixX3d diff = (CurrConstrainedCentersPos - ConstrainedCentersPos);
		for (int i = 0; i < ConstrainedCentersInd.size(); i++)
		{
			int startC = 3 * numV + 3 * numF;
			g(ConstrainedCentersInd[i] + (0 * numF) + startC) = 2 * diff(i, 0); //X-coordinate
			g(ConstrainedCentersInd[i] + (1 * numF) + startC) = 2 * diff(i, 1); //Y-coordinate
			g(ConstrainedCentersInd[i] + (2 * numF) + startC) = 2 * diff(i, 2); //Z-coordinate
		}
	}

	if(update)
		gradient_norm = g.norm();
}

void FixChosenSpheres::hessian()
{
	fill(SS.begin(), SS.end(), 0);
	for (int i = 0; i < ConstrainedCentersInd.size(); i++)
	{
		int startC = 3 * numV + 3 * numF;
		SS[ConstrainedCentersInd[i] + (0 * numF) + startC] = 2;
		SS[ConstrainedCentersInd[i] + (1 * numF) + startC] = 2;
		SS[ConstrainedCentersInd[i] + (2 * numF) + startC] = 2;
	}
}

void FixChosenSpheres::init_hessian()
{
	II.resize(3 * numV + 7 * numF);
	JJ.resize(3 * numV + 7 * numF);
	for (int i = 0; i < (3 * numV + 7 * numF); i++)
	{
		II[i] = i;
		JJ[i] = i;
	}
	SS = std::vector<double>(II.size(), 0.);
}