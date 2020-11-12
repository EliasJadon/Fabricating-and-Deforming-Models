#include "GroupNormals.h"

GroupNormals::GroupNormals()
{
    name = "Group Normals";
	w = 0.05;
	std::cout << "\t" << name << " constructor" << std::endl;
}

GroupNormals::~GroupNormals()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void GroupNormals::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if(numV==0 || numF == 0)
		throw name + " must define members numV & numF before init()!";
	startN = 3 * numV;
	startN_x = (0 * numF) + (3 * numV);
	startN_y = (1 * numF) + (3 * numV);
	startN_z = (2 * numF) + (3 * numV);
	init_hessian();
}

void GroupNormals::updateX(const Eigen::VectorXd& X)
{
	m.lock();
	NormalPos.resize(GroupsInd.size());
	for (int ci = 0; ci < GroupsInd.size(); ci++) {
		//for each cluster
		NormalPos[ci].resize(GroupsInd[ci].size(),3);
		for (int fi = 0; fi < GroupsInd[ci].size(); fi++) {
			int startN = 3 * numV;
			NormalPos[ci].row(fi) <<
				X(GroupsInd[ci][fi] + startN_x),	//X-coordinate
				X(GroupsInd[ci][fi] + startN_y),	//Y-coordinate
				X(GroupsInd[ci][fi] + startN_z);	//Z-coordinate
		}
	}
	currGroupsInd = GroupsInd;
	m.unlock();
}

void GroupNormals::updateExtConstraints(std::vector < std::vector<int>>& CInd)
{
	m.lock();
	GroupsInd = CInd;
	m.unlock();
}

double GroupNormals::value(const bool update)
{
	double E = 0;
	for (int ci = 0; ci < currGroupsInd.size(); ci++)
		for (int f1 = 0; f1 < NormalPos[ci].rows(); f1++)
			for (int f2 = f1 + 1; f2 < NormalPos[ci].rows(); f2++)
				E += (NormalPos[ci].row(f1) - NormalPos[ci].row(f2)).squaredNorm();
	if (update)
		energy_value = E;
	return E;
}

void GroupNormals::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(numV * 3 + numF * 7);
	g.setZero();

	for (int ci = 0; ci < currGroupsInd.size(); ci++) {
		for (int f1 = 0; f1 < NormalPos[ci].rows(); f1++) {
			for (int f2 = f1 + 1; f2 < NormalPos[ci].rows(); f2++) {
				Eigen::RowVector3d diffN = NormalPos[ci].row(f1) - NormalPos[ci].row(f2);
				for (int xyz = 0; xyz < 3; xyz++) {
					//f1 derivative
					g(currGroupsInd[ci][f1] + (xyz * numF) + startN) += 2 * diffN(xyz);
					//f2 derivative
					g(currGroupsInd[ci][f2] + (xyz * numF) + startN) += -2 * diffN(xyz);
				}
			}
		}
	}
	if(update)
		gradient_norm = g.norm();
}

void GroupNormals::hessian()
{
	auto PushTriple = [&](const int row, const int col, const double val) {
		if (col >= row) {
			II.push_back(row);
			JJ.push_back(col);
			SS.push_back(val);
		}
	};
	
	II.clear();
	JJ.clear();
	SS.clear();

	for (int ci = 0; ci < currGroupsInd.size(); ci++) {
		for (int f1 = 0; f1 < NormalPos[ci].rows(); f1++) {
			for (int f2 = f1 + 1; f2 < NormalPos[ci].rows(); f2++) {
				for (int xyz = 0; xyz < 3; xyz++) {
					// d2E/df1df1
					PushTriple(
						currGroupsInd[ci][f1] + (xyz * numF) + startN,
						currGroupsInd[ci][f1] + (xyz * numF) + startN,
						2
					);
					// d2E/df1df2
					PushTriple(
						currGroupsInd[ci][f1] + (xyz * numF) + startN,
						currGroupsInd[ci][f2] + (xyz * numF) + startN,
						-2
					);
					// d2E/df2df1
					PushTriple(
						currGroupsInd[ci][f2] + (xyz * numF) + startN,
						currGroupsInd[ci][f1] + (xyz * numF) + startN,
						-2
					);
					// d2E/df2df2
					PushTriple(
						currGroupsInd[ci][f2] + (xyz * numF) + startN,
						currGroupsInd[ci][f2] + (xyz * numF) + startN,
						2
					);
				}
			}
		}
	}
	PushTriple(
		3 * numV + 7 * numF - 1,
		3 * numV + 7 * numF - 1,
		0
	);
}

void GroupNormals::init_hessian()
{
	
}