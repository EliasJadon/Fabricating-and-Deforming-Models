#include "STVK.h"

double3 sub(const double3 a, const double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

template<int R1, int C1_R2, int C2>
void multiply(
	double mat1[R1][C1_R2],
	double mat2[C1_R2][C2],
	double res[R1][C2])
{
	int i, j, k;
	for (i = 0; i < R1; i++) {
		for (j = 0; j < C2; j++) {
			res[i][j] = 0;
			for (k = 0; k < C1_R2; k++)
				res[i][j] += mat1[i][k] * mat2[k][j];
		}
	}
}
template<int R1, int C1_R2, int C2>
void multiplyTranspose(
	double mat1[C1_R2][R1],
	double mat2[C1_R2][C2],
	double res[R1][C2])
{
	int i, j, k;
	for (i = 0; i < R1; i++) {
		for (j = 0; j < C2; j++) {
			res[i][j] = 0;
			for (k = 0; k < C1_R2; k++)
				res[i][j] += mat1[k][i] * mat2[k][j];
		}
	}
}


STVK::STVK(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) 
{
	init_mesh(V, F);
	name = "STVK";
	w = 0.6;
	cuda_STVK = std::make_shared<Cuda_STVK>();
	cuda_STVK->shearModulus = 0.3;
	cuda_STVK->bulkModulus = 1.5;

	Cuda::AllocateMemory(cuda_STVK->dXInv, F.rows());
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

double STVK::value(Cuda::Array<double>& curr_x, const bool update) {
	Cuda::MemCpyDeviceToHost(curr_x);
	
	
	
	Eigen::VectorXd Energy(restShapeF.rows());
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const unsigned int v0i = restShapeF(fi, 0);
		const unsigned int v1i = restShapeF(fi, 1);
		const unsigned int v2i = restShapeF(fi, 2);
		double3 V0 = make_double3(
			curr_x.host_arr[v0i + cuda_STVK->mesh_indices.startVx],
			curr_x.host_arr[v0i + cuda_STVK->mesh_indices.startVy],
			curr_x.host_arr[v0i + cuda_STVK->mesh_indices.startVz]
		);
		double3 V1 = make_double3(
			curr_x.host_arr[v1i + cuda_STVK->mesh_indices.startVx],
			curr_x.host_arr[v1i + cuda_STVK->mesh_indices.startVy],
			curr_x.host_arr[v1i + cuda_STVK->mesh_indices.startVz]
		);
		double3 V2 = make_double3(
			curr_x.host_arr[v2i + cuda_STVK->mesh_indices.startVx],
			curr_x.host_arr[v2i + cuda_STVK->mesh_indices.startVy],
			curr_x.host_arr[v2i + cuda_STVK->mesh_indices.startVz]
		);

		double3 e10 = sub(V1, V0);
		double3 e20 = sub(V2, V0);
		double dx[3][2];
		dx[0][0] = e10.x; dx[0][1] = e20.x;
		dx[1][0] = e10.y; dx[1][1] = e20.y;
		dx[2][0] = e10.z; dx[2][1] = e20.z;

		double F[3][2];
		double dxInv[2][2];
		dxInv[0][0] = cuda_STVK->dXInv.host_arr[fi].x;
		dxInv[0][1] = cuda_STVK->dXInv.host_arr[fi].y;
		dxInv[1][0] = cuda_STVK->dXInv.host_arr[fi].z;
		dxInv[1][1] = cuda_STVK->dXInv.host_arr[fi].w;
		multiply<3, 2, 2>(dx, dxInv, F);

		//compute the Green Strain = 1/2 * (F'F-I)
		double strain[2][2];
		multiplyTranspose<2, 3, 2>(F, F, strain);
		strain[0][0] -= 1; strain[1][1] -= 1;
		strain[0][0] *= 0.5;
		strain[0][1] *= 0.5;
		strain[1][0] *= 0.5;
		strain[1][1] *= 0.5;

		Energy(fi) =
			cuda_STVK->shearModulus * (pow(strain[0][0], 2) + pow(strain[1][0], 2) + pow(strain[0][1], 2) + pow(strain[1][1], 2)) +
			(cuda_STVK->bulkModulus / 2) * pow((strain[0][0] + strain[1][1]), 2);
	}
	double total_energy = 0;
	for (int i = 0; i < cuda_STVK->restShapeArea.size; i++) {
		total_energy += cuda_STVK->restShapeArea.host_arr[i] * Energy(i);
	}
	
	if (update) {
		Efi = Energy;
		energy_value = total_energy;
	}
	return total_energy;
}

Cuda::Array<double>* STVK::gradient(Cuda::Array<double>& X, const bool update)
{
	return cuda_STVK->gradient(X);
}
