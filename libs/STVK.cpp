#include "STVK.h"

STVK::STVK(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) 
{
	init_mesh(V, F);
	name = "STVK";
	w = 0.6;
	Efi.resize(F.rows());
	Efi.setZero();
	cuda_STVK = std::make_shared<Cuda_STVK>();
	cuda_STVK->shearModulus = 0.3;
	cuda_STVK->bulkModulus = 1.5;

	Cuda::AllocateMemory(cuda_STVK->dXInv, F.rows());
	Cuda::AllocateMemory(cuda_STVK->Energy, F.rows());
	Cuda::AllocateMemory(cuda_STVK->restShapeF, F.rows());
	Cuda::AllocateMemory(cuda_STVK->restShapeArea, F.rows());
	Cuda::initIndices(cuda_STVK->mesh_indices, F.rows(), V.rows(), 0);
	Cuda::AllocateMemory(cuda_STVK->grad, 3 * V.rows() + 7 * F.rows());
	Cuda::AllocateMemory(cuda_STVK->EnergyAtomic, 1);

	setRestShapeFromCurrentConfiguration();
	std::cout << "\t" << name << " constructor" << std::endl;
}

STVK::~STVK() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void STVK::setRestShapeFromCurrentConfiguration() {
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		//Vertices in 3D
		Eigen::VectorXd V0_3D = restShapeV.row(restShapeF(fi, 0));
		Eigen::VectorXd V1_3D = restShapeV.row(restShapeF(fi, 1));
		Eigen::VectorXd V2_3D = restShapeV.row(restShapeF(fi, 2));
		Eigen::VectorXd e10 = V1_3D - V0_3D;
		Eigen::VectorXd e20 = V2_3D - V0_3D;

		//Flatten Vertices to 2D
		double h = e10.norm();
		double temp = e20.transpose() * e10;
		double i = temp / h;
		double j = sqrt(e20.squaredNorm() - pow(i, 2));
		Eigen::Vector2d V0_2D(0, 0);
		Eigen::Vector2d V1_2D(h, 0);
		Eigen::Vector2d V2_2D(i, j);


		//matrix that holds three edge vectors
		Eigen::Matrix2d dX;
		dX <<
			V1_2D[0], V2_2D[0],
			V1_2D[1], V2_2D[1];
		Eigen::Matrix2d inv = dX.inverse();//TODO .inverse() is baaad
		cuda_STVK->dXInv.host_arr[fi] = make_double4(inv(0, 0), inv(0, 1), inv(1, 0), inv(1, 1));
	}
	//compute the area for each triangle
	Eigen::VectorXd HrestShapeArea;
	igl::doublearea(restShapeV, restShapeF, HrestShapeArea);
	HrestShapeArea /= 2;
	for (int fi = 0; fi < cuda_STVK->restShapeArea.size; fi++) {
		cuda_STVK->restShapeArea.host_arr[fi] = HrestShapeArea(fi);
		cuda_STVK->restShapeF.host_arr[fi] = make_int3(
			restShapeF(fi, 0),
			restShapeF(fi, 1),
			restShapeF(fi, 2)
		);
	}
	//init grad
	for (int i = 0; i < cuda_STVK->grad.size; i++) {
		cuda_STVK->grad.host_arr[i] = 0;
	}

	Cuda::MemCpyHostToDevice(cuda_STVK->restShapeF);
	Cuda::MemCpyHostToDevice(cuda_STVK->grad);
	Cuda::MemCpyHostToDevice(cuda_STVK->dXInv);
	Cuda::MemCpyHostToDevice(cuda_STVK->restShapeArea);
}

void STVK::value(Cuda::Array<double>& curr_x) {
	cuda_STVK->value(curr_x);
}

void STVK::gradient(Cuda::Array<double>& X)
{
	cuda_STVK->gradient(X);
}
