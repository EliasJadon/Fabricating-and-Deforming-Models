#include "GroupNormals.h"

GroupNormals::GroupNormals(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixX3i& F,
	const ConstraintsType type)
{
	init_mesh(V, F);
	name = "Group Normals";
	w = 0.05;
	cudaGrouping = std::make_shared<Cuda_Grouping>(F.rows(), V.rows(), type);
	std::cout << "\t" << name << " constructor" << std::endl;
}

GroupNormals::~GroupNormals()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void GroupNormals::updateExtConstraints(std::vector < std::vector<int>>& CInd)
{
	m_value.lock();
	m_gradient.lock();
	
	cudaGrouping->num_clusters = CInd.size();
	cudaGrouping->max_face_per_cluster = 0;
	for (std::vector<int>& C : CInd)
		cudaGrouping->max_face_per_cluster = std::max<int>(cudaGrouping->max_face_per_cluster, C.size());
	
	Cuda::FreeMemory(cudaGrouping->Group_Ind);
	Cuda::FreeMemory(cudaGrouping->EnergyAtomic);
	Cuda::AllocateMemory(cudaGrouping->Group_Ind, cudaGrouping->num_clusters * cudaGrouping->max_face_per_cluster);
	Cuda::AllocateMemory(cudaGrouping->EnergyAtomic, cudaGrouping->num_clusters * pow(cudaGrouping->max_face_per_cluster, 2));

	for (int c = 0; c < cudaGrouping->num_clusters; c++) {
		for (int f = 0; f < cudaGrouping->max_face_per_cluster; f++) {
			const unsigned int globslIndex = f + c * cudaGrouping->max_face_per_cluster;
			if (f < CInd[c].size()) {
				cudaGrouping->Group_Ind.host_arr[globslIndex] = CInd[c][f];
			}
			else {
				cudaGrouping->Group_Ind.host_arr[globslIndex] = -1;
			}
		}
	}
	Cuda::MemCpyHostToDevice(cudaGrouping->Group_Ind);

	m_gradient.unlock();
	m_value.unlock();
}

double GroupNormals::value(Cuda::Array<double>& curr_x, const bool update)
{
	Cuda::MemCpyDeviceToHost(curr_x);

	m_value.lock();
	double E = 0;
	for (int ci = 0; ci < cudaGrouping->num_clusters; ci++)
		for (int f1 = 0; f1 < cudaGrouping->max_face_per_cluster; f1++)
			for (int f2 = f1 + 1; f2 < cudaGrouping->max_face_per_cluster; f2++) {
				const unsigned int indexF1 = cudaGrouping->Group_Ind.host_arr[ci * cudaGrouping->max_face_per_cluster + f1];
				const unsigned int indexF2 = cudaGrouping->Group_Ind.host_arr[ci * cudaGrouping->max_face_per_cluster + f2];
				if (indexF1 != -1 && indexF2 != -1) {
					Eigen::Vector3d NormalPos1(
						curr_x.host_arr[indexF1 + cudaGrouping->startX],	//X-coordinate
						curr_x.host_arr[indexF1 + cudaGrouping->startY],	//Y-coordinate
						curr_x.host_arr[indexF1 + cudaGrouping->startZ]		//Z-coordinate
					);
					Eigen::Vector3d NormalPos2(
						curr_x.host_arr[indexF2 + cudaGrouping->startX],	//X-coordinate
						curr_x.host_arr[indexF2 + cudaGrouping->startY],	//Y-coordinate
						curr_x.host_arr[indexF2 + cudaGrouping->startZ]		//Z-coordinate
					);
					E += (NormalPos1 - NormalPos2).squaredNorm();
				}
			}
	m_value.unlock();
	if (update)
		energy_value = E;
	return E;
}

Cuda::Array<double>* GroupNormals::gradient(Cuda::Array<double>& X, const bool update)
{
	m_gradient.lock();
	Cuda::Array<double>* g = cudaGrouping->gradient(X);
	m_gradient.unlock();
	return g;
}
