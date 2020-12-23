#include "GroupNormals.h"

GroupNormals::GroupNormals(
	const Eigen::MatrixXd& V,
	const Eigen::MatrixX3i& F)
{
	init_mesh(V, F);
	numV = V.rows();
	numF = F.rows();
    name = "Group Normals";
	w = 0.05;
	startN_x = (0 * numF) + (3 * numV);
	startN_y = (1 * numF) + (3 * numV);
	startN_z = (2 * numF) + (3 * numV);

	Cuda::AllocateMemory(grad, (3 * numV) + (7 * numF));
	//init host buffers...
	for (int i = 0; i < grad.size; i++) {
		grad.host_arr[i] = 0;
	}
	// Copy input vectors from host memory to GPU buffers.
	Cuda::MemCpyHostToDevice(grad);

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

	GroupsInd = CInd;

	m_gradient.unlock();
	m_value.unlock();
}

double GroupNormals::value(Cuda::Array<double>& curr_x, const bool update)
{
	Cuda::MemCpyDeviceToHost(curr_x);

	m_value.lock();
	double E = 0;
	for (int ci = 0; ci < GroupsInd.size(); ci++)
		for (int f1 = 0; f1 < GroupsInd[ci].size(); f1++)
			for (int f2 = f1 + 1; f2 < GroupsInd[ci].size(); f2++) {
				Eigen::Vector3d NormalPos1(
					curr_x.host_arr[GroupsInd[ci][f1] + startN_x],	//X-coordinate
					curr_x.host_arr[GroupsInd[ci][f1] + startN_y],	//Y-coordinate
					curr_x.host_arr[GroupsInd[ci][f1] + startN_z]	//Z-coordinate
				);
				Eigen::Vector3d NormalPos2(
					curr_x.host_arr[GroupsInd[ci][f2] + startN_x],	//X-coordinate
					curr_x.host_arr[GroupsInd[ci][f2] + startN_y],	//Y-coordinate
					curr_x.host_arr[GroupsInd[ci][f2] + startN_z]	//Z-coordinate
				);
				E += (NormalPos1 - NormalPos2).squaredNorm();
			}
	m_value.unlock();
	if (update)
		energy_value = E;
	return E;
}

Cuda::Array<double>* GroupNormals::gradient(Cuda::Array<double>& X, const bool update)
{
	m_gradient.lock();
	
	Cuda::MemCpyDeviceToHost(X);

	for (int i = 0; i < grad.size; i++) {
		grad.host_arr[i] = 0;
	}

	for (int ci = 0; ci < GroupsInd.size(); ci++) {
		for (int f1 = 0; f1 < GroupsInd[ci].size(); f1++) {
			for (int f2 = f1 + 1; f2 < GroupsInd[ci].size(); f2++) {
				Eigen::Vector3d NormalPos1(
					X.host_arr[GroupsInd[ci][f1] + startN_x],	//X-coordinate
					X.host_arr[GroupsInd[ci][f1] + startN_y],	//Y-coordinate
					X.host_arr[GroupsInd[ci][f1] + startN_z]	//Z-coordinate
				);
				Eigen::Vector3d NormalPos2(
					X.host_arr[GroupsInd[ci][f2] + startN_x],	//X-coordinate
					X.host_arr[GroupsInd[ci][f2] + startN_y],	//Y-coordinate
					X.host_arr[GroupsInd[ci][f2] + startN_z]	//Z-coordinate
				);
				Eigen::RowVector3d diffN = NormalPos1 - NormalPos2;
				
				grad.host_arr[GroupsInd[ci][f1] + startN_x] += 2 * diffN(0);
				grad.host_arr[GroupsInd[ci][f2] + startN_x] += -2 * diffN(0);
				
				grad.host_arr[GroupsInd[ci][f1] + startN_y] += 2 * diffN(1);
				grad.host_arr[GroupsInd[ci][f2] + startN_y] += -2 * diffN(1);
				
				grad.host_arr[GroupsInd[ci][f1] + startN_z] += 2 * diffN(2);
				grad.host_arr[GroupsInd[ci][f2] + startN_z] += -2 * diffN(2);
				
			}
		}
	}

	Cuda::MemCpyHostToDevice(grad);
	
	m_gradient.unlock();
	return &grad;
}
