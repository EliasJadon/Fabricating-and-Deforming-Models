#include "objective_functions/ClusterCenters.h"

ClusterCenters::ClusterCenters()
{
    name = "Cluster Centers";
	w = 10000;
	std::cout << "\t" << name << " constructor" << std::endl;
}

ClusterCenters::~ClusterCenters()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void ClusterCenters::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if(numV==0 || numF == 0)
		throw name + " must define members numV & numF before init()!";
	init_hessian();
}

void ClusterCenters::updateX(const Eigen::VectorXd& X)
{
	for (int ci = 0; ci < getNumberOfClusters(); ci++) {
		//for each cluster
		CurrClustersPos[ci].resize(ClustersInd[ci].size(),3);
		for (int fi = 0; fi < ClustersInd[ci].size(); fi++) {
			int startC = 3 * numV + 3 * numF;
			CurrClustersPos[ci].row(fi) <<
				X(ClustersInd[ci][fi] + (0 * numF) + startC),	//X-coordinate
				X(ClustersInd[ci][fi] + (1 * numF) + startC),	//Y-coordinate
				X(ClustersInd[ci][fi] + (2 * numF) + startC);	//Z-coordinate
		}
	}
}

int ClusterCenters::getNumberOfClusters() {
	return ClustersInd.size();
}

double ClusterCenters::value(const bool update)
{
	double E = 0;
	for (int ci = 0; ci < getNumberOfClusters(); ci++) {
		//for each cluster
		if (CurrClustersPos[ci].rows() != ClustersInd[ci].size()) 
			return 0;
		for (int f1 = 0; f1 < CurrClustersPos[ci].rows(); f1++) 
			for (int f2 = f1+1; f2 < CurrClustersPos[ci].rows(); f2++) 
				E += (CurrClustersPos[ci].row(f1) - CurrClustersPos[ci].row(f2)).squaredNorm();
	}
	if (update)
		energy_value = E;
	return E;
}

void ClusterCenters::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(numV * 3 + numF * 7);
	g.setZero();

	for (int ci = 0; ci < getNumberOfClusters(); ci++) {
		//for each cluster
		if (CurrClustersPos[ci].rows() != ClustersInd[ci].size()) {
			g.setZero(); 
			return;
		}
		int startC = 3 * numV + 3 * numF;
		for (int f1 = 0; f1 < CurrClustersPos[ci].rows(); f1++) {
			for (int f2 = f1 + 1; f2 < CurrClustersPos[ci].rows(); f2++) {
				Eigen::RowVector3d diff = CurrClustersPos[ci].row(f1) - CurrClustersPos[ci].row(f2);
				for (int xyz = 0; xyz < 3; xyz++) {
					//f1 derivative
					g(ClustersInd[ci][f1] + (xyz * numF) + startC) += 2 * diff(xyz);
					//f2 derivative
					g(ClustersInd[ci][f2] + (xyz * numF) + startC) += -2 * diff(xyz); 
				}
			}
		}
	}
	if(update)
		gradient_norm = g.norm();
}

void ClusterCenters::hessian()
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
		if (CurrClustersPos[ci].rows() != ClustersInd[ci].size()) {
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
		for (int f1 = 0; f1 < CurrClustersPos[ci].rows(); f1++) {
			for (int f2 = f1 + 1; f2 < CurrClustersPos[ci].rows(); f2++) {
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
			}
		}
	}
	PushTriple(
		3 * numV + 7 * numF - 1,
		3 * numV + 7 * numF - 1,
		0
	);
}

void ClusterCenters::init_hessian()
{
	
}