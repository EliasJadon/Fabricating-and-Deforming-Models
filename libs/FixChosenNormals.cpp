#include "FixChosenNormals.h"

FixChosenNormals::FixChosenNormals()
{
    name = "Fix Chosen Normals";
	w = 500;
	std::cout << "\t" << name << " constructor" << std::endl;
}

FixChosenNormals::~FixChosenNormals()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void FixChosenNormals::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if(numV==0 || numF == 0)
		throw name + " must define members numV & numF before init()!";
	startN_x = (3 * numV) + (0 * numF);
	startN_y = (3 * numV) + (1 * numF);
	startN_z = (3 * numV) + (2 * numF);
	init_hessian();
}

void FixChosenNormals::updateExtConstraints(
	std::vector<int>& CNormalsInd,
	Eigen::MatrixX3d& CNormalsPos)
{
	m.lock();
	ConstrainedNormalsInd = CNormalsInd;
	ConstrainedNormalsPos = CNormalsPos;
	m.unlock();
}


void FixChosenNormals::updateX(const Eigen::VectorXd& X)
{
	m.lock();
	Eigen::MatrixX3d CurrConstrainedNormalsPos;
	CurrConstrainedNormalsPos.resizeLike(ConstrainedNormalsPos);
	for (int i = 0; i < ConstrainedNormalsInd.size(); i++)
	{
		CurrConstrainedNormalsPos.row(i) <<
			X(ConstrainedNormalsInd[i] + startN_x),	//X-coordinate
			X(ConstrainedNormalsInd[i] + startN_y),	//Y-coordinate
			X(ConstrainedNormalsInd[i] + startN_z);	//Z-coordinate
	}
	diff = (CurrConstrainedNormalsPos - ConstrainedNormalsPos);
	currConstrainedNormalsInd = ConstrainedNormalsInd;
	m.unlock();
}

double FixChosenNormals::value(const bool update)
{
	double E = diff.squaredNorm();
	if (update) {
		energy_value = E;
	}
	return E;
}

void FixChosenNormals::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(numV * 3 + numF * 7);
	g.setZero();
	for (int i = 0; i < currConstrainedNormalsInd.size(); i++)
	{
		g(currConstrainedNormalsInd[i] + startN_x) = 2 * diff(i, 0); //X-coordinate
		g(currConstrainedNormalsInd[i] + startN_y) = 2 * diff(i, 1); //Y-coordinate
		g(currConstrainedNormalsInd[i] + startN_z) = 2 * diff(i, 2); //Z-coordinate
	}
	if(update)
		gradient_norm = g.norm();
}

void FixChosenNormals::hessian()
{
	fill(SS.begin(), SS.end(), 0);
	for (int i = 0; i < currConstrainedNormalsInd.size(); i++)
	{
		SS[currConstrainedNormalsInd[i] + startN_x] = 2;
		SS[currConstrainedNormalsInd[i] + startN_y] = 2;
		SS[currConstrainedNormalsInd[i] + startN_z] = 2;
	}
}

void FixChosenNormals::init_hessian()
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