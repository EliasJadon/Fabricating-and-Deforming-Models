#include "GroupSpheres.h"

GroupSpheres::GroupSpheres()
{
    name = "Group Spheres";
	//w = 0.05;
	w = 0;
	std::cout << "\t" << name << " constructor" << std::endl;
}

GroupSpheres::~GroupSpheres()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void GroupSpheres::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if(numV==0 || numF == 0)
		throw name + " must define members numV & numF before init()!";

	startC = (3 * numV) + (3 * numF);
	startC_x = (3 * numV) + (3 * numF) + (0 * numF);
	startC_y = (3 * numV) + (3 * numF) + (1 * numF);
	startC_z = (3 * numV) + (3 * numF) + (2 * numF);
	startR = (3 * numV) + (6 * numF);
	init_hessian();
}

void GroupSpheres::updateX(const Eigen::VectorXd& X)
{
	m.lock();
	SphereCenterPos.resize(GroupsInd.size());
	SphereRadiusLen.resize(GroupsInd.size());
	for (int ci = 0; ci < GroupsInd.size(); ci++) {
		//for each cluster
		SphereCenterPos[ci].resize(GroupsInd[ci].size(),3);
		SphereRadiusLen[ci].resize(GroupsInd[ci].size());
		for (int fi = 0; fi < GroupsInd[ci].size(); fi++) {
			
			SphereCenterPos[ci].row(fi) <<
				X(GroupsInd[ci][fi] + startC_x),	//X-coordinate
				X(GroupsInd[ci][fi] + startC_y),	//Y-coordinate
				X(GroupsInd[ci][fi] + startC_z);	//Z-coordinate
			SphereRadiusLen[ci](fi) = X(GroupsInd[ci][fi] + startR);
		}
	}
	currGroupsInd = GroupsInd;
	m.unlock();
}

void GroupSpheres::updateExtConstraints(std::vector < std::vector<int>>& CInd) {
	m.lock();
	GroupsInd = CInd;
	m.unlock();
}


double GroupSpheres::value(const bool update)
{
	double E = 0;
	for (int ci = 0; ci < currGroupsInd.size(); ci++)
		for (int f1 = 0; f1 < SphereCenterPos[ci].rows(); f1++)
			for (int f2 = f1 + 1; f2 < SphereCenterPos[ci].rows(); f2++) {
				E += (SphereCenterPos[ci].row(f1) - SphereCenterPos[ci].row(f2)).squaredNorm();
				E += pow(SphereRadiusLen[ci](f1)- SphereRadiusLen[ci](f2), 2);
			}
	if (update)
		energy_value = E;
	return E;
}

void GroupSpheres::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(numV * 3 + numF * 7);
	g.setZero();

	for (int ci = 0; ci < currGroupsInd.size(); ci++) {
		for (int f1 = 0; f1 < SphereCenterPos[ci].rows(); f1++) {
			for (int f2 = f1 + 1; f2 < SphereCenterPos[ci].rows(); f2++) {
				Eigen::RowVector3d diffC = SphereCenterPos[ci].row(f1) - SphereCenterPos[ci].row(f2);
				double diffR = SphereRadiusLen[ci](f1) - SphereRadiusLen[ci](f2);
				for (int xyz = 0; xyz < 3; xyz++) {
					//f1 derivative
					g(currGroupsInd[ci][f1] + (xyz * numF) + startC) += 2 * diffC(xyz);
					//f2 derivative
					g(currGroupsInd[ci][f2] + (xyz * numF) + startC) += -2 * diffC(xyz);
				}
				//f1 derivative
				g(currGroupsInd[ci][f1] + startR) += 2 * diffR;
				//f2 derivative
				g(currGroupsInd[ci][f2] + startR) += -2 * diffR;
			}
		}
	}
	if(update)
		gradient_norm = g.norm();
}

void GroupSpheres::hessian()
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
		for (int f1 = 0; f1 < SphereCenterPos[ci].rows(); f1++) {
			for (int f2 = f1 + 1; f2 < SphereCenterPos[ci].rows(); f2++) {
				for (int xyz = 0; xyz < 3; xyz++) {
					// d2E/df1df1
					PushTriple(
						currGroupsInd[ci][f1] + (xyz * numF) + startC,
						currGroupsInd[ci][f1] + (xyz * numF) + startC,
						2
					);
					// d2E/df1df2
					PushTriple(
						currGroupsInd[ci][f1] + (xyz * numF) + startC,
						currGroupsInd[ci][f2] + (xyz * numF) + startC,
						-2
					);
					// d2E/df2df1
					PushTriple(
						currGroupsInd[ci][f2] + (xyz * numF) + startC,
						currGroupsInd[ci][f1] + (xyz * numF) + startC,
						-2
					);
					// d2E/df2df2
					PushTriple(
						currGroupsInd[ci][f2] + (xyz * numF) + startC,
						currGroupsInd[ci][f2] + (xyz * numF) + startC,
						2
					);
				}
				// d2E/df1df1
				PushTriple(
					currGroupsInd[ci][f1] + startR,
					currGroupsInd[ci][f1] + startR,
					2
				);
				// d2E/df1df2
				PushTriple(
					currGroupsInd[ci][f1] + startR,
					currGroupsInd[ci][f2] + startR,
					-2
				);
				// d2E/df2df1
				PushTriple(
					currGroupsInd[ci][f2] + startR,
					currGroupsInd[ci][f1] + startR,
					-2
				);
				// d2E/df2df2
				PushTriple(
					currGroupsInd[ci][f2] + startR,
					currGroupsInd[ci][f2] + startR,
					2
				);
			}
		}
	}
	PushTriple(
		3 * numV + 7 * numF - 1,
		3 * numV + 7 * numF - 1,
		0
	);
}

void GroupSpheres::init_hessian()
{
	
}