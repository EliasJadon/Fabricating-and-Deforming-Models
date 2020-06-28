#include "objective_functions/ClusterNormals.h"

ClusterNormals::ClusterNormals()
{
    name = "Cluster Normals";
	w = 10000;
	std::cout << "\t" << name << " constructor" << std::endl;
}

ClusterNormals::~ClusterNormals()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void ClusterNormals::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if(numV==0 || numF == 0)
		throw name + " must define members numV & numF before init()!";
	init_hessian();
}

void ClusterNormals::updateX(const Eigen::VectorXd& X)
{
	NormalPos.resize(getNumberOfClusters());
	for (int ci = 0; ci < getNumberOfClusters(); ci++) {
		//for each cluster
		NormalPos[ci].resize(ClustersInd[ci].size(),3);
		for (int fi = 0; fi < ClustersInd[ci].size(); fi++) {
			int startN = 3 * numV;
			NormalPos[ci].row(fi) <<
				X(ClustersInd[ci][fi] + (0 * numF) + startN),	//X-coordinate
				X(ClustersInd[ci][fi] + (1 * numF) + startN),	//Y-coordinate
				X(ClustersInd[ci][fi] + (2 * numF) + startN);	//Z-coordinate
		}
	}
}

int ClusterNormals::getNumberOfClusters() {
	return ClustersInd.size();
}

int ClusterNormals::CheckInputValidation() {
	if (NormalPos.size() != ClustersInd.size())
		return 0;
	for (int ci = 0; ci < getNumberOfClusters(); ci++)
		if (NormalPos[ci].rows() != ClustersInd[ci].size())
			return 0;
	return 1;
}

double ClusterNormals::value(const bool update)
{
	double E = 0;
	for (int ci = 0; ci < getNumberOfClusters(); ci++) {
		//for each cluster
		if (!CheckInputValidation())
			return 0;
		for (int f1 = 0; f1 < NormalPos[ci].rows(); f1++)
			for (int f2 = f1 + 1; f2 < NormalPos[ci].rows(); f2++)
				E += (NormalPos[ci].row(f1) - NormalPos[ci].row(f2)).squaredNorm();
	}
	if (update)
		energy_value = E;
	return E;
}

void ClusterNormals::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(numV * 3 + numF * 7);
	g.setZero();

	for (int ci = 0; ci < getNumberOfClusters(); ci++) {
		//for each cluster
		if (!CheckInputValidation()) {
			g.setZero(); 
			return;
		}
		int startN = 3 * numV;
		for (int f1 = 0; f1 < NormalPos[ci].rows(); f1++) {
			for (int f2 = f1 + 1; f2 < NormalPos[ci].rows(); f2++) {
				Eigen::RowVector3d diffN = NormalPos[ci].row(f1) - NormalPos[ci].row(f2);
				for (int xyz = 0; xyz < 3; xyz++) {
					//f1 derivative
					g(ClustersInd[ci][f1] + (xyz * numF) + startN) += 2 * diffN(xyz);
					//f2 derivative
					g(ClustersInd[ci][f2] + (xyz * numF) + startN) += -2 * diffN(xyz);
				}
			}
		}
	}
	if(update)
		gradient_norm = g.norm();
}

void ClusterNormals::hessian()
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
		int startN = 3 * numV;
		for (int f1 = 0; f1 < NormalPos[ci].rows(); f1++) {
			for (int f2 = f1 + 1; f2 < NormalPos[ci].rows(); f2++) {
				for (int xyz = 0; xyz < 3; xyz++) {
					// d2E/df1df1
					PushTriple(
						ClustersInd[ci][f1] + (xyz * numF) + startN,
						ClustersInd[ci][f1] + (xyz * numF) + startN,
						2
					);
					// d2E/df1df2
					PushTriple(
						ClustersInd[ci][f1] + (xyz * numF) + startN,
						ClustersInd[ci][f2] + (xyz * numF) + startN,
						-2
					);
					// d2E/df2df1
					PushTriple(
						ClustersInd[ci][f2] + (xyz * numF) + startN,
						ClustersInd[ci][f1] + (xyz * numF) + startN,
						-2
					);
					// d2E/df2df2
					PushTriple(
						ClustersInd[ci][f2] + (xyz * numF) + startN,
						ClustersInd[ci][f2] + (xyz * numF) + startN,
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

void ClusterNormals::init_hessian()
{
	
}