#include "objective_functions/FixChosenVertices.h"

FixChosenVertices::FixChosenVertices()
{
    name = "Fix Chosen Vertices";
	w = 500;
	std::cout << "\t" << name << " constructor" << std::endl;
}

FixChosenVertices::~FixChosenVertices() 
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void FixChosenVertices::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if(numV==0 || numF == 0)
		throw name + " must define members numV & numF before init()!";
	startV_x = (0 * numV);
	startV_y = (1 * numV);
	startV_z = (2 * numV);
	init_hessian();
}

void FixChosenVertices::updateExtConstraints(
	std::vector<int>& CVInd,
	Eigen::MatrixX3d& CVPos)
{
	m.lock();
	ConstrainedVerticesInd = CVInd;
	ConstrainedVerticesPos = CVPos;
	m.unlock();
}

void FixChosenVertices::updateX(const Eigen::VectorXd& X)
{
	m.lock();
	Eigen::MatrixX3d CurrConstrainedVerticesPos;
	CurrConstrainedVerticesPos.resizeLike(ConstrainedVerticesPos);
	for (int i = 0; i < ConstrainedVerticesInd.size(); i++)
	{
		CurrConstrainedVerticesPos.row(i) <<
			X(ConstrainedVerticesInd[i] + startV_x),	//X-coordinate
			X(ConstrainedVerticesInd[i] + startV_y),	//Y-coordinate
			X(ConstrainedVerticesInd[i] + startV_z);	//Z-coordinate
	}
	diff = (CurrConstrainedVerticesPos - ConstrainedVerticesPos);
	currConstrainedVerticesInd = ConstrainedVerticesInd;
	m.unlock();
}

double FixChosenVertices::value(const bool update)
{
	double E = diff.squaredNorm();
	if (update) {
		energy_value = E;
	}
	return E;
}

void FixChosenVertices::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(numV * 3 + numF * 7);
	g.setZero();
	for (int i = 0; i < currConstrainedVerticesInd.size(); i++)
	{
		g(currConstrainedVerticesInd[i] + startV_x) = 2 * diff(i, 0); //X-coordinate
		g(currConstrainedVerticesInd[i] + startV_y) = 2 * diff(i, 1); //Y-coordinate
		g(currConstrainedVerticesInd[i] + startV_z) = 2 * diff(i, 2); //Z-coordinate
	}
	if(update)
		gradient_norm = g.norm();
}

void FixChosenVertices::hessian()
{
	fill(SS.begin(), SS.end(), 0);
	for (int i = 0; i < currConstrainedVerticesInd.size(); i++)
	{
		SS[currConstrainedVerticesInd[i] + startV_x] = 2;
		SS[currConstrainedVerticesInd[i] + startV_y] = 2;
		SS[currConstrainedVerticesInd[i] + startV_z] = 2;
	}
}

void FixChosenVertices::init_hessian()
{
	II.resize((3 * numV) + 1);
	JJ.resize((3 * numV) + 1);
	for (int i = 0; i < 3*numV; i++)
	{
		II[i] = i;
		JJ[i] = i;
	}
	II[3 * numV] = 3 * numV + 7 * numF - 1;
	JJ[3 * numV] = 3 * numV + 7 * numF - 1;
	SS = std::vector<double>(II.size(), 0.);
}