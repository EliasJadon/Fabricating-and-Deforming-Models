#include "objective_functions/ClusterSpheres.h"

ClusterSpheres::ClusterSpheres()
{
    name = "Cluster Spheres";
	w = 10000;
	std::cout << "\t" << name << " constructor" << std::endl;
}

ClusterSpheres::~ClusterSpheres()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void ClusterSpheres::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if(numV==0 || numF == 0)
		throw name + " must define members numV & numF before init()!";
	init_hessian();
}

void ClusterSpheres::updateX(const Eigen::VectorXd& X)
{
	SphereCenterPos.resize(getNumberOfClusters());
	SphereRadiusLen.resize(getNumberOfClusters());
	for (int ci = 0; ci < getNumberOfClusters(); ci++) {
		//for each cluster
		SphereCenterPos[ci].resize(ClustersInd[ci].size(),3);
		SphereRadiusLen[ci].resize(ClustersInd[ci].size());
		for (int fi = 0; fi < ClustersInd[ci].size(); fi++) {
			int startC = 3 * numV + 3 * numF;
			int startR = 3 * numV + 6 * numF;
			SphereCenterPos[ci].row(fi) <<
				X(ClustersInd[ci][fi] + (0 * numF) + startC),	//X-coordinate
				X(ClustersInd[ci][fi] + (1 * numF) + startC),	//Y-coordinate
				X(ClustersInd[ci][fi] + (2 * numF) + startC);	//Z-coordinate
			SphereRadiusLen[ci](fi) = X(ClustersInd[ci][fi] + startR);
		}
	}
}

int ClusterSpheres::getNumberOfClusters() {
	return ClustersInd.size();
}

int ClusterSpheres::CheckInputValidation() {
	if ((SphereCenterPos.size() != ClustersInd.size()) 
		|| (SphereRadiusLen.size() != ClustersInd.size()))
		return 0;
	for (int ci = 0; ci < getNumberOfClusters(); ci++)
		if ((SphereCenterPos[ci].rows() != ClustersInd[ci].size()) 
			||(SphereRadiusLen[ci].size() != ClustersInd[ci].size()))
			return 0;
	return 1;
}

double ClusterSpheres::value(const bool update)
{
	double E = 0;
	
	for (int ci = 0; ci < getNumberOfClusters(); ci++) {
		//for each cluster
		if (!CheckInputValidation())
			return 0;
		for (int f1 = 0; f1 < SphereCenterPos[ci].rows(); f1++)
			for (int f2 = f1 + 1; f2 < SphereCenterPos[ci].rows(); f2++) {
				E += (SphereCenterPos[ci].row(f1) - SphereCenterPos[ci].row(f2)).squaredNorm();
				E += pow(SphereRadiusLen[ci](f1)- SphereRadiusLen[ci](f2), 2);
			}
	}
	if (update)
		energy_value = E;
	return E;
}

void ClusterSpheres::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(numV * 3 + numF * 7);
	g.setZero();

	for (int ci = 0; ci < getNumberOfClusters(); ci++) {
		//for each cluster
		if (!CheckInputValidation()) {
			g.setZero(); 
			return;
		}
		int startC = 3 * numV + 3 * numF;
		int startR = 3 * numV + 6 * numF;
		for (int f1 = 0; f1 < SphereCenterPos[ci].rows(); f1++) {
			for (int f2 = f1 + 1; f2 < SphereCenterPos[ci].rows(); f2++) {
				Eigen::RowVector3d diffC = SphereCenterPos[ci].row(f1) - SphereCenterPos[ci].row(f2);
				double diffR = SphereRadiusLen[ci](f1) - SphereRadiusLen[ci](f2);
				for (int xyz = 0; xyz < 3; xyz++) {
					//f1 derivative
					g(ClustersInd[ci][f1] + (xyz * numF) + startC) += 2 * diffC(xyz);
					//f2 derivative
					g(ClustersInd[ci][f2] + (xyz * numF) + startC) += -2 * diffC(xyz); 
				}
				//f1 derivative
				g(ClustersInd[ci][f1] + startR) += 2 * diffR;
				//f2 derivative
				g(ClustersInd[ci][f2] + startR) += -2 * diffR;
			}
		}
	}
	if(update)
		gradient_norm = g.norm();
}

void ClusterSpheres::hessian()
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

	for (int ci = 0; ci < getNumberOfClusters(); ci++) {
		//for each cluster
		if (!CheckInputValidation()) {
			II.clear();
			JJ.clear();
			SS.clear();
			PushTriple(
				3 * numV + 7 * numF - 1,
				3 * numV + 7 * numF - 1,
				0
			);
			return;
		}
		int startC = 3 * numV + 3 * numF;
		int startR = 3 * numV + 6 * numF;
		for (int f1 = 0; f1 < SphereCenterPos[ci].rows(); f1++) {
			for (int f2 = f1 + 1; f2 < SphereCenterPos[ci].rows(); f2++) {
				for (int xyz = 0; xyz < 3; xyz++) {
					// d2E/df1df1
					PushTriple(
						ClustersInd[ci][f1] + (xyz * numF) + startC,
						ClustersInd[ci][f1] + (xyz * numF) + startC,
						2
					);
					// d2E/df1df2
					PushTriple(
						ClustersInd[ci][f1] + (xyz * numF) + startC,
						ClustersInd[ci][f2] + (xyz * numF) + startC,
						-2
					);
					// d2E/df2df1
					PushTriple(
						ClustersInd[ci][f2] + (xyz * numF) + startC,
						ClustersInd[ci][f1] + (xyz * numF) + startC,
						-2
					);
					// d2E/df2df2
					PushTriple(
						ClustersInd[ci][f2] + (xyz * numF) + startC,
						ClustersInd[ci][f2] + (xyz * numF) + startC,
						2
					);
				}
				// d2E/df1df1
				PushTriple(
					ClustersInd[ci][f1] + startR,
					ClustersInd[ci][f1] + startR,
					2
				);
				// d2E/df1df2
				PushTriple(
					ClustersInd[ci][f1] + startR,
					ClustersInd[ci][f2] + startR,
					-2
				);
				// d2E/df2df1
				PushTriple(
					ClustersInd[ci][f2] + startR,
					ClustersInd[ci][f1] + startR,
					-2
				);
				// d2E/df2df2
				PushTriple(
					ClustersInd[ci][f2] + startR,
					ClustersInd[ci][f2] + startR,
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

void ClusterSpheres::init_hessian()
{
	
}