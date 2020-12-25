#include "STVK.h"

template<int R1, int C1_R2, int C2> void multiply(
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
template<int R1, int C1_R2, int C2> void multiplyTranspose(
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
double3 sub(const double3 a, const double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}



STVK::STVK(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) 
{
	init_mesh(V, F);
	name = "STVK";
	w = 0;
	shearModulus = 0.3;
	bulkModulus = 1.5;
	Cuda::AllocateMemory(dXInv, F.rows());
	Cuda::initIndices(mesh_indices, F.rows(), V.rows(), 0);
	Cuda::AllocateMemory(grad, 3 * V.rows() + 7 * F.rows());
	setRestShapeFromCurrentConfiguration();
	std::cout << "\t" << name << " constructor" << std::endl;
}

STVK::~STVK() {
	Cuda::FreeMemory(dXInv);
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
		dXInv.host_arr[fi] = make_double4(inv(0, 0), inv(0, 1), inv(1, 0), inv(1, 1)); 
	}
	//compute the area for each triangle
	igl::doublearea(restShapeV, restShapeF, restShapeArea);
	restShapeArea /= 2;
	Cuda::MemCpyHostToDevice(dXInv);
}

double STVK::value(Cuda::Array<double>& curr_x, const bool update) {
	Cuda::MemCpyDeviceToHost(curr_x);
	
	
	
	Eigen::VectorXd Energy(restShapeF.rows());
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const unsigned int v0i = restShapeF(fi, 0);
		const unsigned int v1i = restShapeF(fi, 1);
		const unsigned int v2i = restShapeF(fi, 2);
		double3 V0 = make_double3(
			curr_x.host_arr[v0i + mesh_indices.startVx],
			curr_x.host_arr[v0i + mesh_indices.startVy],
			curr_x.host_arr[v0i + mesh_indices.startVz]
		);
		double3 V1 = make_double3(
			curr_x.host_arr[v1i + mesh_indices.startVx],
			curr_x.host_arr[v1i + mesh_indices.startVy],
			curr_x.host_arr[v1i + mesh_indices.startVz]
		);
		double3 V2 = make_double3(
			curr_x.host_arr[v2i + mesh_indices.startVx],
			curr_x.host_arr[v2i + mesh_indices.startVy],
			curr_x.host_arr[v2i + mesh_indices.startVz]
		);

		double3 e10 = sub(V1, V0);
		double3 e20 = sub(V2, V0);
		double dx[3][2];
		dx[0][0] = e10.x; dx[0][1] = e20.x;
		dx[1][0] = e10.y; dx[1][1] = e20.y;
		dx[2][0] = e10.z; dx[2][1] = e20.z;

		double F[3][2];
		double dxInv[2][2];
		dxInv[0][0] = dXInv.host_arr[fi].x;
		dxInv[0][1] = dXInv.host_arr[fi].y;
		dxInv[1][0] = dXInv.host_arr[fi].z;
		dxInv[1][1] = dXInv.host_arr[fi].w;
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
			shearModulus * (pow(strain[0][0], 2) + pow(strain[1][0], 2) + pow(strain[0][1], 2) + pow(strain[1][1], 2)) +
			(bulkModulus / 2) * pow((strain[0][0] + strain[1][1]), 2);
	}
	double total_energy = restShapeArea.transpose() * Energy;
	
	if (update) {
		Efi = Energy;
		energy_value = total_energy;
	}
	return total_energy;
}

Cuda::Array<double>* STVK::gradient(Cuda::Array<double>& X, const bool update)
{
	Cuda::MemCpyDeviceToHost(X);


	for (int i = 0; i < grad.size; i++) {
		grad.host_arr[i] = 0;
	}

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		const unsigned int v0i = restShapeF(fi, 0);
		const unsigned int v1i = restShapeF(fi, 1);
		const unsigned int v2i = restShapeF(fi, 2);
		double3 V0 = make_double3(
			X.host_arr[v0i + mesh_indices.startVx],
			X.host_arr[v0i + mesh_indices.startVy],
			X.host_arr[v0i + mesh_indices.startVz]
		);
		double3 V1 = make_double3(
			X.host_arr[v1i + mesh_indices.startVx],
			X.host_arr[v1i + mesh_indices.startVy],
			X.host_arr[v1i + mesh_indices.startVz]
		);
		double3 V2 = make_double3(
			X.host_arr[v2i + mesh_indices.startVx],
			X.host_arr[v2i + mesh_indices.startVy],
			X.host_arr[v2i + mesh_indices.startVz]
		);

		double3 e10 = sub(V1, V0);
		double3 e20 = sub(V2, V0);
		double dx[3][2];
		dx[0][0] = e10.x; dx[0][1] = e20.x;
		dx[1][0] = e10.y; dx[1][1] = e20.y;
		dx[2][0] = e10.z; dx[2][1] = e20.z;

		double F[3][2];
		double dxInv[2][2];
		dxInv[0][0] = dXInv.host_arr[fi].x;
		dxInv[0][1] = dXInv.host_arr[fi].y;
		dxInv[1][0] = dXInv.host_arr[fi].z;
		dxInv[1][1] = dXInv.host_arr[fi].w;
		multiply<3, 2, 2>(dx, dxInv, F);

		//compute the Green Strain = 1/2 * (F'F-I)
		double strain[2][2];
		multiplyTranspose<2, 3, 2>(F, F, strain);
		strain[0][0] -= 1; strain[1][1] -= 1;
		strain[0][0] *= 0.5;
		strain[0][1] *= 0.5;
		strain[1][0] *= 0.5;
		strain[1][1] *= 0.5;

		double dF_dX[6][9] = { 0 };
		dF_dX[0][0] = -dXInv.host_arr[fi].x - dXInv.host_arr[fi].z;
		dF_dX[0][1] = dXInv.host_arr[fi].x;
		dF_dX[0][2] = dXInv.host_arr[fi].z; 

		dF_dX[1][0] = -dXInv.host_arr[fi].y - dXInv.host_arr[fi].w;
		dF_dX[1][1] = dXInv.host_arr[fi].y;
		dF_dX[1][2] = dXInv.host_arr[fi].w;

		dF_dX[2][3] = -dXInv.host_arr[fi].x - dXInv.host_arr[fi].z;
		dF_dX[2][4] = dXInv.host_arr[fi].x;
		dF_dX[2][5] = dXInv.host_arr[fi].z; 
							
		dF_dX[3][3] = -dXInv.host_arr[fi].y - dXInv.host_arr[fi].w;
		dF_dX[3][4] = dXInv.host_arr[fi].y;
		dF_dX[3][5] = dXInv.host_arr[fi].w;
				
		dF_dX[4][6] = -dXInv.host_arr[fi].x - dXInv.host_arr[fi].z;
		dF_dX[4][7] = dXInv.host_arr[fi].x;
		dF_dX[4][8] = dXInv.host_arr[fi].z;
				
		dF_dX[5][6] = -dXInv.host_arr[fi].y - dXInv.host_arr[fi].w;
		dF_dX[5][7] = dXInv.host_arr[fi].y;
		dF_dX[5][8] = dXInv.host_arr[fi].w;

		double dstrain_dF[4][6] = { 0 };
		dstrain_dF[0][0] = F[0][0];
		dstrain_dF[0][2] = F[1][0];
		dstrain_dF[0][4] = F[2][0];
			
		dstrain_dF[1][0] = 0.5 * F[0][1];
		dstrain_dF[1][1] = 0.5 * F[0][0];
		dstrain_dF[1][2] = 0.5 * F[1][1];
		dstrain_dF[1][3] = 0.5 * F[1][0];
		dstrain_dF[1][4] = 0.5 * F[2][1];
		dstrain_dF[1][5] = 0.5 * F[2][0];
			
		dstrain_dF[2][0] = 0.5 * F[0][1];
		dstrain_dF[2][1] = 0.5 * F[0][0];
		dstrain_dF[2][2] = 0.5 * F[1][1];
		dstrain_dF[2][3] = 0.5 * F[1][0];
		dstrain_dF[2][4] = 0.5 * F[2][1];
		dstrain_dF[2][5] = 0.5 * F[2][0];
			
		dstrain_dF[3][1] = F[0][1];
		dstrain_dF[3][3] = F[1][1];
		dstrain_dF[3][5] = F[2][1];
		
		double dE_dJ[1][4];
		dE_dJ[0][0] = restShapeArea[fi] * (2 * shearModulus * strain[0][0] + bulkModulus * (strain[0][0] + strain[1][1]));
		dE_dJ[0][1] = restShapeArea[fi] * (2 * shearModulus * strain[0][1]);
		dE_dJ[0][2] = restShapeArea[fi] * (2 * shearModulus * strain[1][0]);
		dE_dJ[0][3] = restShapeArea[fi] * (2 * shearModulus * strain[1][1] + bulkModulus * (strain[0][0] + strain[1][1]));
		
		double dE_dX[1][9];
		double temp[1][6];
		multiply<1, 4, 6>(dE_dJ, dstrain_dF, temp);
		multiply<1, 6, 9>(temp, dF_dX, dE_dX);

		grad.host_arr[v0i + mesh_indices.startVx] += dE_dX[0][0];
		grad.host_arr[v1i + mesh_indices.startVx] += dE_dX[0][1];
		grad.host_arr[v2i + mesh_indices.startVx] += dE_dX[0][2];
		grad.host_arr[v0i + mesh_indices.startVy] += dE_dX[0][3];
		grad.host_arr[v1i + mesh_indices.startVy] += dE_dX[0][4];
		grad.host_arr[v2i + mesh_indices.startVy] += dE_dX[0][5];
		grad.host_arr[v0i + mesh_indices.startVz] += dE_dX[0][6];
		grad.host_arr[v1i + mesh_indices.startVz] += dE_dX[0][7];
		grad.host_arr[v2i + mesh_indices.startVz] += dE_dX[0][8];
	}
	Cuda::MemCpyHostToDevice(grad);
	return &grad;
}
