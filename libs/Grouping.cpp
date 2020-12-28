#include "Grouping.h"

Grouping::Grouping(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixX3i& F,
	const ConstraintsType type)
{
	init_mesh(V, F);
	if(type == ConstraintsType::NORMALS)
		name = "Group Normals";
	if(type == ConstraintsType::SPHERES)
		name = "Group Spheres";

	w = 0.05;
	cudaGrouping = std::make_shared<Cuda_Grouping>(F.rows(), V.rows(), type);
	std::cout << "\t" << name << " constructor" << std::endl;
}

Grouping::~Grouping()
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void Grouping::updateExtConstraints(std::vector < std::vector<int>>& CInd)
{
	m_value.lock();
	m_gradient.lock();
	
	cudaGrouping->num_clusters = CInd.size();
	cudaGrouping->max_face_per_cluster = 0;
	for (std::vector<int>& C : CInd)
		cudaGrouping->max_face_per_cluster = std::max<int>(cudaGrouping->max_face_per_cluster, C.size());
	
	Cuda::FreeMemory(cudaGrouping->Group_Ind);
	Cuda::AllocateMemory(cudaGrouping->Group_Ind, cudaGrouping->num_clusters * cudaGrouping->max_face_per_cluster);
	
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

void Grouping::value(Cuda::Array<double>& curr_x)
{
	m_value.lock();
	cudaGrouping->value(curr_x);
	m_value.unlock();
}

Cuda::Array<double>* Grouping::gradient(Cuda::Array<double>& X, const bool update)
{
	m_gradient.lock();
	Cuda::Array<double>* g = cudaGrouping->gradient(X);
	m_gradient.unlock();
	return g;
}
