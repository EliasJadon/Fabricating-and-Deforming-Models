#include "objective_functions/PenaltyPositionalConstraints.h"

PenaltyPositionalConstraints::PenaltyPositionalConstraints()
{
    name = "Positional Constraints";
	w = 10000;
	std::cout << name << " constructor" << std::endl;
}

PenaltyPositionalConstraints::~PenaltyPositionalConstraints() 
{
	std::cout << name << " destructor" << std::endl;
}

void PenaltyPositionalConstraints::init()
{
	std::cout << name << " initialization" << std::endl;
	if(numV==0 || numF == 0)
		throw name + " must define members numV & numF before init()!";
	init_hessian();
}

void PenaltyPositionalConstraints::updateX(const Eigen::VectorXd& X)
{
	CurrConstrainedVerticesPos.resizeLike(ConstrainedVerticesPos);
	for (int i = 0; i < ConstrainedVerticesInd.size(); i++)
	{
		CurrConstrainedVerticesPos.row(i) <<
			X(ConstrainedVerticesInd[i] + (0 * numV)),	//X-coordinate
			X(ConstrainedVerticesInd[i] + (1 * numV)),	//Y-coordinate
			X(ConstrainedVerticesInd[i] + (2 * numV));	//Z-coordinate
	}
}

double PenaltyPositionalConstraints::value(const bool update)
{
	if (CurrConstrainedVerticesPos.rows() != ConstrainedVerticesPos.rows()) {
		return 0;
	}
	double E = (ConstrainedVerticesPos - CurrConstrainedVerticesPos).squaredNorm();
	if (update) {
		energy_value = E;
	}
	
	return E;
}

void PenaltyPositionalConstraints::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(numV * 3);
	g.setZero();

	if (CurrConstrainedVerticesPos.rows() == ConstrainedVerticesPos.rows()) {
		Eigen::MatrixX3d diff = (CurrConstrainedVerticesPos - ConstrainedVerticesPos);
		for (int i = 0; i < ConstrainedVerticesInd.size(); i++)
		{
			g(ConstrainedVerticesInd[i] + (0 * numV)) = 2 * diff(i, 0); //X-coordinate
			g(ConstrainedVerticesInd[i] + (1 * numV)) = 2 * diff(i, 1); //Y-coordinate
			g(ConstrainedVerticesInd[i] + (2 * numV)) = 2 * diff(i, 2); //Z-coordinate
		}
	}

	if(update)
		gradient_norm = g.norm();
}

void PenaltyPositionalConstraints::hessian()
{
	fill(SS.begin(), SS.end(), 0);
	for (int i = 0; i < ConstrainedVerticesInd.size(); i++)
	{
		SS[ConstrainedVerticesInd[i] + (0 * numV)] = 2;
		SS[ConstrainedVerticesInd[i] + (1 * numV)] = 2;
		SS[ConstrainedVerticesInd[i] + (2 * numV)] = 2;
	}
}

void PenaltyPositionalConstraints::init_hessian()
{
	II.resize(3 * numV);
	JJ.resize(3 * numV);
	for (int i = 0; i < 3*numV; i++)
	{
		II[i] = i;
		JJ[i] = i;
	}
	SS = std::vector<double>(II.size(), 0.);
}