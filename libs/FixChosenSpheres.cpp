#include "FixChosenSpheres.h"

FixChosenSpheres::FixChosenSpheres()
{
    name = "Fix Chosen Spheres";
	//w = 500;
	w = 0;
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
	startC_x = (3 * numV) + (3 * numF) + (0 * numF);
	startC_y = (3 * numV) + (3 * numF) + (1 * numF);
	startC_z = (3 * numV) + (3 * numF) + (2 * numF);
}

void FixChosenSpheres::updateExtConstraints(
	std::vector<int>& CCentersInd,
	Eigen::MatrixX3d& CCentersPos)
{
	m.lock();
	ConstrainedCentersInd = CCentersInd;
	ConstrainedCentersPos = CCentersPos;
	m.unlock();
}

void FixChosenSpheres::updateX(Cuda::Array<double>& curr_x)
{
	//m.lock();
	//Eigen::MatrixX3d CurrConstrainedCentersPos;
	//CurrConstrainedCentersPos.resizeLike(ConstrainedCentersPos);
	//for (int i = 0; i < ConstrainedCentersInd.size(); i++)
	//{
	//	CurrConstrainedCentersPos.row(i) <<
	//		X(ConstrainedCentersInd[i] + startC_x),	//X-coordinate
	//		X(ConstrainedCentersInd[i] + startC_y),	//Y-coordinate
	//		X(ConstrainedCentersInd[i] + startC_z);	//Z-coordinate
	//}
	//diff = (CurrConstrainedCentersPos - ConstrainedCentersPos);
	//currConstrainedCentersInd = ConstrainedCentersInd;
	//m.unlock();
}

double FixChosenSpheres::value(Cuda::Array<double>& curr_x, const bool update)
{
	double E = diff.squaredNorm();
	if (update) {
		energy_value = E;
	}
	return E;
}

void FixChosenSpheres::gradient(Cuda::Array<double>& X, Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(numV * 3 + numF * 7);
	g.setZero();
	for (int i = 0; i < currConstrainedCentersInd.size(); i++)
	{
		g(currConstrainedCentersInd[i] + startC_x) = 2 * diff(i, 0); //X-coordinate
		g(currConstrainedCentersInd[i] + startC_y) = 2 * diff(i, 1); //Y-coordinate
		g(currConstrainedCentersInd[i] + startC_z) = 2 * diff(i, 2); //Z-coordinate
	}
	if(update)
		gradient_norm = g.norm();
}