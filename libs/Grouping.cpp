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
	if(type == ConstraintsType::CYLINDERS)
		name = "Group Cylinders";

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
	this->CInd = CInd;
	m_gradient.unlock();
	m_value.unlock();
}

void Grouping::value(Cuda::Array<double>& curr_x)
{
	m_value.lock(); 
	if (cudaGrouping->type != ConstraintsType::CYLINDERS)
		cudaGrouping->value(curr_x);
	else {
		Cuda::MemCpyDeviceToHost(curr_x);
		cudaGrouping->EnergyAtomic.host_arr[0] = 0;
		for (int ci = 0; ci < CInd.size(); ci++) {
			for (int f0 = 0; f0 < CInd[ci].size(); f0++) {
				for (int f1 = f0 + 1; f1 < CInd[ci].size(); f1++) {
					Eigen::Vector3d A0(
						curr_x.host_arr[f0 + cudaGrouping->mesh_indices.startAx],
						curr_x.host_arr[f0 + cudaGrouping->mesh_indices.startAy],
						curr_x.host_arr[f0 + cudaGrouping->mesh_indices.startAz]
					);
					Eigen::Vector3d A1(
						curr_x.host_arr[f1 + cudaGrouping->mesh_indices.startAx],
						curr_x.host_arr[f1 + cudaGrouping->mesh_indices.startAy],
						curr_x.host_arr[f1 + cudaGrouping->mesh_indices.startAz]
					);
					Eigen::Vector3d C0(
						curr_x.host_arr[f0 + cudaGrouping->mesh_indices.startCx],
						curr_x.host_arr[f0 + cudaGrouping->mesh_indices.startCy],
						curr_x.host_arr[f0 + cudaGrouping->mesh_indices.startCz]
					);
					Eigen::Vector3d C1(
						curr_x.host_arr[f1 + cudaGrouping->mesh_indices.startCx],
						curr_x.host_arr[f1 + cudaGrouping->mesh_indices.startCy],
						curr_x.host_arr[f1 + cudaGrouping->mesh_indices.startCz]
					);
					Eigen::Vector3d c10 = C1 - C0;
					double R0 = curr_x.host_arr[f0 + cudaGrouping->mesh_indices.startR];
					double R1 = curr_x.host_arr[f1 + cudaGrouping->mesh_indices.startR];
					
					double d_cylinder_dir = (A1 - A0).squaredNorm();
					double d_center0 = (c10 - c10.norm() * A0).squaredNorm();
					double d_center1 = (c10 - c10.norm() * A1).squaredNorm();
					double d_radius = pow(R1 - R0, 2);
					cudaGrouping->EnergyAtomic.host_arr[0] += (d_cylinder_dir + d_center0 + d_center1 + d_radius);
				}
			}
		}
		Cuda::MemCpyHostToDevice(cudaGrouping->EnergyAtomic);
	}
	m_value.unlock();
}

void Grouping::gradient(Cuda::Array<double>& X)
{
	m_gradient.lock();
	if (cudaGrouping->type != ConstraintsType::CYLINDERS)
		cudaGrouping->gradient(X);
	else {
		Cuda::MemCpyDeviceToHost(X);
		for (int i = 0; i < cudaGrouping->grad.size; i++)
			cudaGrouping->grad.host_arr[i] = 0;

		for (int ci = 0; ci < CInd.size(); ci++) {
			for (int f0 = 0; f0 < CInd[ci].size(); f0++) {
				for (int f1 = f0 + 1; f1 < CInd[ci].size(); f1++) {
					Eigen::Vector3d A0(
						X.host_arr[f0 + cudaGrouping->mesh_indices.startAx],
						X.host_arr[f0 + cudaGrouping->mesh_indices.startAy],
						X.host_arr[f0 + cudaGrouping->mesh_indices.startAz]
					);
					Eigen::Vector3d A1(
						X.host_arr[f1 + cudaGrouping->mesh_indices.startAx],
						X.host_arr[f1 + cudaGrouping->mesh_indices.startAy],
						X.host_arr[f1 + cudaGrouping->mesh_indices.startAz]
					);
					Eigen::Vector3d C0(
						X.host_arr[f0 + cudaGrouping->mesh_indices.startCx],
						X.host_arr[f0 + cudaGrouping->mesh_indices.startCy],
						X.host_arr[f0 + cudaGrouping->mesh_indices.startCz]
					);
					Eigen::Vector3d C1(
						X.host_arr[f1 + cudaGrouping->mesh_indices.startCx],
						X.host_arr[f1 + cudaGrouping->mesh_indices.startCy],
						X.host_arr[f1 + cudaGrouping->mesh_indices.startCz]
					);
					Eigen::Vector3d c10 = C1 - C0;
					double norm_c10 = c10.norm();
					double R0 = X.host_arr[f0 + cudaGrouping->mesh_indices.startR];
					double R1 = X.host_arr[f1 + cudaGrouping->mesh_indices.startR];

					double d_cylinder_dir = (A1 - A0).squaredNorm();
					Eigen::Vector3d center0 = c10 - c10.norm() * A0;
					double d_center0 = center0.squaredNorm();
					Eigen::Vector3d center1 = c10 - c10.norm() * A1;
					double d_center1 = center1.squaredNorm();
					double d_radius = pow(R1 - R0, 2);
					
					auto& I = cudaGrouping->mesh_indices;
					//A0(0);
					cudaGrouping->grad.host_arr[f0 + I.startAx] += (2 * (A0(0) - A1(0))) - (2 * center0(0) * norm_c10);
					//A1(0)
					cudaGrouping->grad.host_arr[f1 + I.startAx] += (2 * (A1(0) - A0(0))) - (2 * center1(0) * norm_c10);
					//A0(1)
					cudaGrouping->grad.host_arr[f0 + I.startAy] += (2 * (A0(1) - A1(1))) - (2 * center0(1) * norm_c10);
					//A1(1)
					cudaGrouping->grad.host_arr[f1 + I.startAy] += (2 * (A1(1) - A0(1))) - (2 * center1(1) * norm_c10);
					//A0(2)
					cudaGrouping->grad.host_arr[f0 + I.startAz] += (2 * (A0(2) - A1(2))) - (2 * center0(2) * norm_c10);
					//A1(2)
					cudaGrouping->grad.host_arr[f1 + I.startAz] += (2 * (A1(2) - A0(2))) - (2 * center1(2) * norm_c10);
					//R0
					cudaGrouping->grad.host_arr[f0 + I.startR] += 2 * (R0 - R1);
					//R1
					cudaGrouping->grad.host_arr[f1 + I.startR] += 2 * (R1 - R0);

					//C0(0)
					cudaGrouping->grad.host_arr[f0 + I.startCx] +=
						2 * (
							-center0(0) - center1(0) +
							center0(0) * ((A0(0) * (C1(0) - C0(0))) / norm_c10) +
							center0(1) * ((A0(1) * (C1(0) - C0(0))) / norm_c10) +
							center0(2) * ((A0(2) * (C1(0) - C0(0))) / norm_c10) +
							center1(0) * ((A1(0) * (C1(0) - C0(0))) / norm_c10) +
							center1(1) * ((A1(1) * (C1(0) - C0(0))) / norm_c10) +
							center1(2) * ((A1(2) * (C1(0) - C0(0))) / norm_c10)
							);
					//C1(0)
					cudaGrouping->grad.host_arr[f1 + I.startCx] +=
						2 * (
							center0(0) + center1(0) +
							center0(0) * ((A0(0) * (C0(0) - C1(0))) / norm_c10) +
							center0(1) * ((A0(1) * (C0(0) - C1(0))) / norm_c10) +
							center0(2) * ((A0(2) * (C0(0) - C1(0))) / norm_c10) +
							center1(0) * ((A1(0) * (C0(0) - C1(0))) / norm_c10) +
							center1(1) * ((A1(1) * (C0(0) - C1(0))) / norm_c10) +
							center1(2) * ((A1(2) * (C0(0) - C1(0))) / norm_c10)
							);
					//C0(1)
					cudaGrouping->grad.host_arr[f0 + I.startCy] +=
						2 * (
							-center0(1) - center1(1) +
							center0(0) * ((A0(0) * (C1(1) - C0(1))) / norm_c10) +
							center0(1) * ((A0(1) * (C1(1) - C0(1))) / norm_c10) +
							center0(2) * ((A0(2) * (C1(1) - C0(1))) / norm_c10) +
							center1(0) * ((A1(0) * (C1(1) - C0(1))) / norm_c10) +
							center1(1) * ((A1(1) * (C1(1) - C0(1))) / norm_c10) +
							center1(2) * ((A1(2) * (C1(1) - C0(1))) / norm_c10)
							);
					//C1(1)
					cudaGrouping->grad.host_arr[f1 + I.startCy] +=
						2 * (
							center0(1) + center1(1) +
							center0(0) * ((A0(0) * (C0(1) - C1(1))) / norm_c10) +
							center0(1) * ((A0(1) * (C0(1) - C1(1))) / norm_c10) +
							center0(2) * ((A0(2) * (C0(1) - C1(1))) / norm_c10) +
							center1(0) * ((A1(0) * (C0(1) - C1(1))) / norm_c10) +
							center1(1) * ((A1(1) * (C0(1) - C1(1))) / norm_c10) +
							center1(2) * ((A1(2) * (C0(1) - C1(1))) / norm_c10)
							);
					//C0(2)
					cudaGrouping->grad.host_arr[f0 + I.startCz] +=
						2 * (
							-center0(2) - center1(2) +
							center0(0) * ((A0(0) * (C1(2) - C0(2))) / norm_c10) +
							center0(1) * ((A0(1) * (C1(2) - C0(2))) / norm_c10) +
							center0(2) * ((A0(2) * (C1(2) - C0(2))) / norm_c10) +
							center1(0) * ((A1(0) * (C1(2) - C0(2))) / norm_c10) +
							center1(1) * ((A1(1) * (C1(2) - C0(2))) / norm_c10) +
							center1(2) * ((A1(2) * (C1(2) - C0(2))) / norm_c10)
							);
					//C1(2)
					cudaGrouping->grad.host_arr[f1 + I.startCz] +=
						2 * (
							center0(2) + center1(2) +
							center0(0) * ((A0(0) * (C0(2) - C1(2))) / norm_c10) +
							center0(1) * ((A0(1) * (C0(2) - C1(2))) / norm_c10) +
							center0(2) * ((A0(2) * (C0(2) - C1(2))) / norm_c10) +
							center1(0) * ((A1(0) * (C0(2) - C1(2))) / norm_c10) +
							center1(1) * ((A1(1) * (C0(2) - C1(2))) / norm_c10) +
							center1(2) * ((A1(2) * (C0(2) - C1(2))) / norm_c10)
							);
				}
			}
		}

		Cuda::MemCpyHostToDevice(cudaGrouping->grad);
	}
	m_gradient.unlock();
}
