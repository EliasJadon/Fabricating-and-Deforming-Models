#include "objective_functions/AllVertexPositions.h"

AllVertexPositions::AllVertexPositions()
{
    name = "All Vertex Positions";
	w = 0;
	std::cout << "\t" << name << " constructor" << std::endl;
}

AllVertexPositions::~AllVertexPositions()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void AllVertexPositions::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if (restShapeV.size() == 0 || restShapeF.size() == 0)
		throw name + " must define members V,F before init()!";
	init_hessian();
}

void AllVertexPositions::updateX(const Eigen::VectorXd& X)
{
	assert(X.rows() == (restShapeV.size() + restShapeF.size()));
	CurrV = Eigen::Map<const Eigen::MatrixX3d>(X.middleRows(0,restShapeV.size()).data(), restShapeV.rows(), 3);
}

double AllVertexPositions::value(const bool update)
{
	double E = (CurrV - restShapeV).squaredNorm();
	if (update)
		energy_value = E;
	return E;
}

void AllVertexPositions::gradient(Eigen::VectorXd& g, const bool update)
{
	int n = restShapeV.rows();
	g.conservativeResize(restShapeV.size()+ restShapeF.size());
	g.setZero();

	Eigen::MatrixX3d diff = CurrV - restShapeV;
	for (int i = 0; i < n; i++) {
		g(i + (0 * n)) = 2 * diff(i, 0); //X-coordinate
		g(i + (1 * n)) = 2 * diff(i, 1); //Y-coordinate
		g(i + (2 * n)) = 2 * diff(i, 2); //Z-coordinate
	}
	if(update)
		gradient_norm = g.norm();
}

void AllVertexPositions::hessian()
{
	// The hessian is constant!
	// Empty on purpose
}

void AllVertexPositions::init_hessian()
{
	II.clear(); JJ.clear(); SS.clear();
	int n = restShapeV.rows();
	for (int i = 0; i < 3*n; i++)
	{
		II.push_back(i);
		JJ.push_back(i);
		SS.push_back(2);
	}
	II.push_back(restShapeV.size() + restShapeF.size() - 1);
	JJ.push_back(restShapeV.size() + restShapeF.size() - 1);
	SS.push_back(0);
}