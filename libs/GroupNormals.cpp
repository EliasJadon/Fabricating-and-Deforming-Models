#include "GroupNormals.h"

GroupNormals::GroupNormals(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixX3i& F)
{
	init_mesh(V, F);
	numV = V.rows();
	numF = F.rows();
    name = "Group Normals";
	//w = 0.05;
	w = 0;

	std::cout << "\t" << name << " initialization" << std::endl;
	if (numV == 0 || numF == 0)
		throw name + " must define members numV & numF before init()!";
	startN = 3 * numV;
	startN_x = (0 * numF) + (3 * numV);
	startN_y = (1 * numF) + (3 * numV);
	startN_z = (2 * numF) + (3 * numV);

	std::cout << "\t" << name << " constructor" << std::endl;
}

GroupNormals::~GroupNormals()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

//void GroupNormals::updateX(Cuda::Array<double>& curr_x)
//{
	//m.lock();
	//NormalPos.resize(GroupsInd.size());
	//for (int ci = 0; ci < GroupsInd.size(); ci++) {
	//	//for each cluster
	//	NormalPos[ci].resize(GroupsInd[ci].size(),3);
	//	for (int fi = 0; fi < GroupsInd[ci].size(); fi++) {
	//		int startN = 3 * numV;
	//		NormalPos[ci].row(fi) <<
	//			X(GroupsInd[ci][fi] + startN_x),	//X-coordinate
	//			X(GroupsInd[ci][fi] + startN_y),	//Y-coordinate
	//			X(GroupsInd[ci][fi] + startN_z);	//Z-coordinate
	//	}
	//}
	//currGroupsInd = GroupsInd;
	//m.unlock();
//}

void GroupNormals::updateExtConstraints(std::vector < std::vector<int>>& CInd)
{
	m.lock();
	GroupsInd = CInd;
	m.unlock();
}

double GroupNormals::value(Cuda::Array<double>& curr_x, const bool update)
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

Cuda::Array<double>* GroupNormals::gradient(Cuda::Array<double>& X, const bool update)
{
	return NULL;

	//g.conservativeResize(numV * 3 + numF * 7);
	//g.setZero();

	//for (int ci = 0; ci < currGroupsInd.size(); ci++) {
	//	for (int f1 = 0; f1 < NormalPos[ci].rows(); f1++) {
	//		for (int f2 = f1 + 1; f2 < NormalPos[ci].rows(); f2++) {
	//			Eigen::RowVector3d diffN = NormalPos[ci].row(f1) - NormalPos[ci].row(f2);
	//			for (int xyz = 0; xyz < 3; xyz++) {
	//				//f1 derivative
	//				g(currGroupsInd[ci][f1] + (xyz * numF) + startN) += 2 * diffN(xyz);
	//				//f2 derivative
	//				g(currGroupsInd[ci][f2] + (xyz * numF) + startN) += -2 * diffN(xyz);
	//			}
	//		}
	//	}
	//}
	//if(update)
	//	gradient_norm = g.norm();
}
