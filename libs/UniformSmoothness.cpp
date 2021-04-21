#include "UniformSmoothness.h"
#include <igl/adjacency_matrix.h>
#include <igl/sum.h>
#include <igl/diag.h>

UniformSmoothness::UniformSmoothness(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F)
{
	init_mesh(V, F);
	name = "Uniform Smoothness";
	w = 0.023;
	Cuda::AllocateMemory(grad, (3 * V.rows()) + (10 * F.rows()));
	Cuda::AllocateMemory(EnergyAtomic, 1);
	Cuda::initIndices(mesh_indices, F.rows(), V.rows(), 0);
	Efi.resize(F.rows());
	Efi.setZero();

	// Prepare The Uniform Laplacian 
	// Mesh in (V,F)
	Eigen::SparseMatrix<double> A;
	igl::adjacency_matrix((Eigen::MatrixXi)F, A);
    // sum each row 
	Eigen::SparseVector<double> Asum;
	igl::sum(A,1,Asum);
    // Convert row sums into diagonal of sparse matrix
	Eigen::SparseMatrix<double> Adiag;
    igl::diag(Asum,Adiag);
    // Build uniform laplacian
    L = A-Adiag;

	//For Debugging...
	//std::cout << "The uniform Laplacian is:\n" << L << std::endl;

	std::cout << "\t" << name << " constructor" << std::endl;
}

UniformSmoothness::~UniformSmoothness() {
	FreeMemory(grad);
	FreeMemory(EnergyAtomic);
	std::cout << "\t" << name << " destructor" << std::endl;
}

void UniformSmoothness::value(Cuda::Array<double>& curr_x) {
	Cuda::MemCpyDeviceToHost(curr_x);
	// Energy = ||L * x||^2
	// Energy = ||diag(L,L,L) * (x;y;z)||^2
	Eigen::VectorXd X(restShapeV.rows()), Y(restShapeV.rows()), Z(restShapeV.rows());
	for (int vi = 0; vi < restShapeV.rows(); vi++) {
		X(vi) = curr_x.host_arr[vi + mesh_indices.startVx];
		Y(vi) = curr_x.host_arr[vi + mesh_indices.startVy];
		Z(vi) = curr_x.host_arr[vi + mesh_indices.startVz];
	}
	EnergyAtomic.host_arr[0] = 
		(L * X).squaredNorm() + 
		(L * Y).squaredNorm() + 
		(L * Z).squaredNorm();
	Cuda::MemCpyHostToDevice(EnergyAtomic);
}

void UniformSmoothness::gradient(Cuda::Array<double>& input_X)
{
	Cuda::MemCpyDeviceToHost(input_X);
	for (int i = 0; i < grad.size; i++)
		grad.host_arr[i] = 0;
	// gradient = 2*||L * x|| * L
	Eigen::VectorXd X(restShapeV.rows()), Y(restShapeV.rows()), Z(restShapeV.rows());
	for (int vi = 0; vi < restShapeV.rows(); vi++) {
		X(vi) = input_X.host_arr[vi + mesh_indices.startVx];
		Y(vi) = input_X.host_arr[vi + mesh_indices.startVy];
		Z(vi) = input_X.host_arr[vi + mesh_indices.startVz];
	}
	Eigen::VectorXd grad_X = 2 * L * (L * X);
	Eigen::VectorXd grad_Y = 2 * L * (L * Y);
	Eigen::VectorXd grad_Z = 2 * L * (L * Z);
	

	for (int vi = 0; vi < restShapeV.rows(); vi++) {
		grad.host_arr[vi + mesh_indices.startVx] += grad_X(vi);
		grad.host_arr[vi + mesh_indices.startVy] += grad_Y(vi);
		grad.host_arr[vi + mesh_indices.startVz] += grad_Z(vi);
	}
	Cuda::MemCpyHostToDevice(grad);
}
