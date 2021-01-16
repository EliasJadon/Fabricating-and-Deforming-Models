#include "SDenergy.h"

double3 sub(const double3 a, const double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
double dot(const double3 a, const double3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
double3 mul(const double a, const double3 b)
{
	return make_double3(a * b.x, a * b.y, a * b.z);
}
double squared_norm(const double3 a)
{
	return dot(a, a);
}
double norm(const double3 a)
{
	return sqrt(squared_norm(a));
}
double3 normalize(const double3 a)
{
	return mul(1.0f / norm(a), a);
}
double3 cross(const double3 a, const double3 b)
{
	return make_double3(
		a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x
	);
}
template<int N> void multiply(
	double3 mat1,
	double mat2[3][N],
	double res[N])
{
	for (int i = 0; i < N; i++) {
		res[i] = mat1.x * mat2[0][i] + mat1.y * mat2[1][i] + mat1.z * mat2[2][i];
	}
}
template<int N> void multiply(
	double4 mat1,
	double mat2[4][N],
	double res[N])
{
	for (int i = 0; i < N; i++) {
		res[i] = 
			mat1.x * mat2[0][i] + 
			mat1.y * mat2[1][i] + 
			mat1.z * mat2[2][i] + 
			mat1.w * mat2[3][i];
	}
}


SDenergy::SDenergy(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) {
	init_mesh(V, F);
	name = "Symmetric Dirichlet";
	w = 0.6;

	Efi.resize(F.rows());
	Efi.setZero();

	Eigen::MatrixX3d D1cols, D2cols;
	OptimizationUtils::computeSurfaceGradientPerFace(restShapeV, restShapeF, D1cols, D2cols);
	
	//compute the area for each triangle
	Eigen::VectorXd restShapeArea;
	igl::doublearea(restShapeV, restShapeF, restShapeArea);
	restShapeArea /= 2;

	cuda_SD = std::make_shared<Cuda_SDenergy>(F.rows(), V.rows());
	for (int i = 0; i < cuda_SD->Energy.size; i++)
		cuda_SD->Energy.host_arr[i] = 0;
	for (int i = 0; i < cuda_SD->restShapeF.size; i++)
		cuda_SD->restShapeF.host_arr[i] = make_int3(restShapeF(i, 0), restShapeF(i, 1), restShapeF(i, 2));
	for (int i = 0; i < cuda_SD->restShapeArea.size; i++)
		cuda_SD->restShapeArea.host_arr[i] = restShapeArea(i);
	for (int i = 0; i < cuda_SD->grad.size; i++)
		cuda_SD->grad.host_arr[i] = 0;
	for (int i = 0; i < cuda_SD->EnergyAtomic.size; i++)
		cuda_SD->EnergyAtomic.host_arr[i] = 0;
	for (int i = 0; i < cuda_SD->D1d.size; i++)
		cuda_SD->D1d.host_arr[i] = make_double3(D1cols(i, 0), D1cols(i, 1), D1cols(i, 2));
	for (int i = 0; i < cuda_SD->D2d.size; i++)
		cuda_SD->D2d.host_arr[i] = make_double3(D2cols(i, 0), D2cols(i, 1), D2cols(i, 2));
	
	Cuda::MemCpyHostToDevice(cuda_SD->Energy);
	Cuda::MemCpyHostToDevice(cuda_SD->restShapeF);
	Cuda::MemCpyHostToDevice(cuda_SD->restShapeArea);
	Cuda::MemCpyHostToDevice(cuda_SD->grad);
	Cuda::MemCpyHostToDevice(cuda_SD->D1d);
	Cuda::MemCpyHostToDevice(cuda_SD->D2d);
	Cuda::MemCpyHostToDevice(cuda_SD->EnergyAtomic);
	std::cout << "\t" << name << " constructor" << std::endl;
}

SDenergy::~SDenergy() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void SDenergy::value(Cuda::Array<double>& curr_x) {
	cuda_SD->value(curr_x);
}

void SDenergy::gradient(Cuda::Array<double>& X)
{
	Cuda::MemCpyDeviceToHost(X);
	for (int i = 0; i < cuda_SD->grad.size; i++)
		cuda_SD->grad.host_arr[i] = 0;

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		int v0_index = restShapeF(fi, 0);
		int v1_index = restShapeF(fi, 1);
		int v2_index = restShapeF(fi, 2);
		double3 V0 = make_double3(
			X.host_arr[v0_index + cuda_SD->mesh_indices.startVx],
			X.host_arr[v0_index + cuda_SD->mesh_indices.startVy],
			X.host_arr[v0_index + cuda_SD->mesh_indices.startVz]
		);
		double3 V1 = make_double3(
			X.host_arr[v1_index + cuda_SD->mesh_indices.startVx],
			X.host_arr[v1_index + cuda_SD->mesh_indices.startVy],
			X.host_arr[v1_index + cuda_SD->mesh_indices.startVz]
		);
		double3 V2 = make_double3(
			X.host_arr[v2_index + cuda_SD->mesh_indices.startVx],
			X.host_arr[v2_index + cuda_SD->mesh_indices.startVy],
			X.host_arr[v2_index + cuda_SD->mesh_indices.startVz]
		);

		double3 e10 = sub(V1, V0);
		double3 e20 = sub(V2, V0);
		double3 B1 = normalize(e10);
		double3 B2 = normalize(cross(cross(B1, e20), B1));

		double3 Xi = make_double3(
			dot(V0, B1),
			dot(V1, B1),
			dot(V2, B1)
		);
		double3 Yi = make_double3(
			dot(V0, B2),
			dot(V1, B2),
			dot(V2, B2)
		);

		//prepare jacobian		
		const double a = dot(cuda_SD->D1d.host_arr[fi], Xi);
		const double b = dot(cuda_SD->D1d.host_arr[fi], Yi);
		const double c = dot(cuda_SD->D2d.host_arr[fi], Xi);
		const double d = dot(cuda_SD->D2d.host_arr[fi], Yi);
		const double detJ = a * d - b * c;
		const double det2 = pow(detJ, 2);
		const double a2 = pow(a, 2);
		const double b2 = pow(b, 2);
		const double c2 = pow(c, 2);
		const double d2 = pow(d, 2);
		const double det3 = pow(detJ, 3);
		const double Fnorm = a2 + b2 + c2 + d2;

		double4 de_dJ = make_double4(
			cuda_SD->restShapeArea.host_arr[fi] * (a + a / det2 - d * Fnorm / det3),
			cuda_SD->restShapeArea.host_arr[fi] * (b + b / det2 + c * Fnorm / det3),
			cuda_SD->restShapeArea.host_arr[fi] * (c + c / det2 + b * Fnorm / det3),
			cuda_SD->restShapeArea.host_arr[fi] * (d + d / det2 - a * Fnorm / det3)
		);
		double dE_dX[9],dj_dx[4][9];
		dJ_dX(dj_dx,fi, V0, V1, V2);
		multiply<9>(de_dJ, dj_dx, dE_dX);

		cuda_SD->grad.host_arr[restShapeF(fi, 0) + cuda_SD->mesh_indices.startVx] += dE_dX[0];
		cuda_SD->grad.host_arr[restShapeF(fi, 1) + cuda_SD->mesh_indices.startVx] += dE_dX[1];
		cuda_SD->grad.host_arr[restShapeF(fi, 2) + cuda_SD->mesh_indices.startVx] += dE_dX[2];
		cuda_SD->grad.host_arr[restShapeF(fi, 0) + cuda_SD->mesh_indices.startVy] += dE_dX[3];
		cuda_SD->grad.host_arr[restShapeF(fi, 1) + cuda_SD->mesh_indices.startVy] += dE_dX[4];
		cuda_SD->grad.host_arr[restShapeF(fi, 2) + cuda_SD->mesh_indices.startVy] += dE_dX[5];
		cuda_SD->grad.host_arr[restShapeF(fi, 0) + cuda_SD->mesh_indices.startVz] += dE_dX[6];
		cuda_SD->grad.host_arr[restShapeF(fi, 1) + cuda_SD->mesh_indices.startVz] += dE_dX[7];
		cuda_SD->grad.host_arr[restShapeF(fi, 2) + cuda_SD->mesh_indices.startVz] += dE_dX[8];
	}
	Cuda::MemCpyHostToDevice(cuda_SD->grad);
}

void SDenergy::dJ_dX(
	double g[4][9],
	int fi, 
	const double3 V0,
	const double3 V1,
	const double3 V2)
{
	double dV0_dX[3][9] = { 0 }, dV1_dX[3][9] = { 0 }, dV2_dX[3][9] = { 0 };
	dV0_dX[0][0] = 1; dV0_dX[1][3] = 1; dV0_dX[2][6] = 1;
	dV1_dX[0][1] = 1; dV1_dX[1][4] = 1; dV1_dX[2][7] = 1;
	dV2_dX[0][2] = 1; dV2_dX[1][5] = 1; dV2_dX[2][8] = 1;
	double3 e10 = sub(V1, V0);
	double3 e20 = sub(V2, V0);
	double3 B1 = normalize(e10);
	double3 B2 = normalize(cross(cross(B1, e20), B1));

	double db1_dX[3][9], db2_dX[3][9], XX[3][9], YY[3][9];
	dB1_dX(db1_dX, fi, e10);
	dB2_dX(db2_dX, fi, e10, e20);
	
	double res1[9], res2[9], res3[9], res4[9], res5[9], res6[9];
	multiply<9>(V0, db1_dX, res1);
	multiply<9>(V1, db1_dX, res2);
	multiply<9>(V2, db1_dX, res3);
	multiply<9>(B1, dV0_dX, res4);
	multiply<9>(B1, dV1_dX, res5);
	multiply<9>(B1, dV2_dX, res6);
	for (int i = 0; i < 9; i++) {
		XX[0][i] = res1[i] + res4[i];
		XX[1][i] = res2[i] + res5[i];
		XX[2][i] = res3[i] + res6[i];
	}

	multiply<9>(V0, db2_dX, res1);
	multiply<9>(V1, db2_dX, res2);
	multiply<9>(V2, db2_dX, res3);
	multiply<9>(B2, dV0_dX, res4);
	multiply<9>(B2, dV1_dX, res5);
	multiply<9>(B2, dV2_dX, res6);
	for (int i = 0; i < 9; i++) {
		YY[0][i] = res1[i] + res4[i];
		YY[1][i] = res2[i] + res5[i];
		YY[2][i] = res3[i] + res6[i];
	}
	multiply<9>(cuda_SD->D1d.host_arr[fi], XX, res1);
	multiply<9>(cuda_SD->D1d.host_arr[fi], YY, res2);
	multiply<9>(cuda_SD->D2d.host_arr[fi], XX, res3);
	multiply<9>(cuda_SD->D2d.host_arr[fi], YY, res4);
	for (int i = 0; i < 9; i++) {
		g[0][i] = res1[i];
		g[1][i] = res2[i];
		g[2][i] = res3[i];
		g[3][i] = res4[i];
	}
}

void SDenergy::dB1_dX(double g[3][9], int fi, const double3 e10)
{
	double Norm = norm(e10);
	double Norm3 = pow(Norm, 3);
	double dB1x_dx0 = -(pow(e10.y, 2) + pow(e10.z, 2)) / Norm3;
	double dB1y_dy0 = -(pow(e10.x, 2) + pow(e10.z, 2)) / Norm3;
	double dB1z_dz0 = -(pow(e10.x, 2) + pow(e10.y, 2)) / Norm3;
	double dB1x_dy0 = (e10.y * e10.x) / Norm3;
	double dB1x_dz0 = (e10.z * e10.x) / Norm3;
	double dB1y_dz0 = (e10.z * e10.y) / Norm3;
	double B1gradient[3][9] = {
		dB1x_dx0, -dB1x_dx0, 0, dB1x_dy0, -dB1x_dy0, 0, dB1x_dz0, -dB1x_dz0, 0,
		dB1x_dy0, -dB1x_dy0, 0, dB1y_dy0, -dB1y_dy0, 0, dB1y_dz0, -dB1y_dz0, 0,
		dB1x_dz0, -dB1x_dz0, 0, dB1y_dz0, -dB1y_dz0, 0, dB1z_dz0, -dB1z_dz0, 0
	};
	for (int i = 0; i < 3; i++) {
		for (int j = 0; j < 9; j++) {
			g[i][j] = B1gradient[i][j];
		}
	}
}

void SDenergy::dB2_dX(double g[3][9], int fi, const double3 e10, const double3 e20)
{
	double3 b2 = cross(cross(e10, e20), e10);
	double NormB2 = norm(b2);
	double NormB2_2 = pow(NormB2, 2);

	double dxyz[3][6] = {
		-e10.y * e20.y - e10.z * e20.z,
		-e10.x * e20.y + 2 * e10.y * e20.x,
		2 * e10.z * e20.x - e10.x * e20.z,
		pow(e10.y, 2) + pow(e10.z, 2),
		-e10.y * e10.x,
		-e10.x * e10.z,
		2 * e10.x * e20.y - e10.y * e20.x,
		-e10.z * e20.z - e20.x * e10.x,
		-e10.y * e20.z + 2 * e10.z * e20.y,
		-e10.x * e10.y,
		pow(e10.z, 2) + pow(e10.x, 2),
		-e10.z * e10.y,
		-e10.z * e20.x + 2 * e10.x * e20.z,
		2 * e10.y * e20.z - e10.z * e20.y,
		-e10.x * e20.x - e10.y * e20.y,
		-e10.x * e10.z,
		-e10.z * e10.y,
		pow(e10.x, 2) + pow(e10.y, 2)
	};
		

	double dnorm[6] = {
		(b2.x * dxyz[0][0] + b2.y * dxyz[1][0] + b2.z * dxyz[2][0]) / NormB2,
		(b2.x * dxyz[0][1] + b2.y * dxyz[1][1] + b2.z * dxyz[2][1]) / NormB2,
		(b2.x * dxyz[0][2] + b2.y * dxyz[1][2] + b2.z * dxyz[2][2]) / NormB2,
		(b2.x * dxyz[0][3] + b2.y * dxyz[1][3] + b2.z * dxyz[2][3]) / NormB2,
		(b2.x * dxyz[0][4] + b2.y * dxyz[1][4] + b2.z * dxyz[2][4]) / NormB2,
		(b2.x * dxyz[0][5] + b2.y * dxyz[1][5] + b2.z * dxyz[2][5]) / NormB2
	};
		
	g[0][1] = (dxyz[0][0] * NormB2 - b2.x * dnorm[0]) / NormB2_2;
	g[0][2] = (dxyz[0][3] * NormB2 - b2.x * dnorm[3]) / NormB2_2;
	g[0][4] = (dxyz[0][1] * NormB2 - b2.x * dnorm[1]) / NormB2_2;
	g[0][5] = (dxyz[0][4] * NormB2 - b2.x * dnorm[4]) / NormB2_2;
	g[0][7] = (dxyz[0][2] * NormB2 - b2.x * dnorm[2]) / NormB2_2;
	g[0][8] = (dxyz[0][5] * NormB2 - b2.x * dnorm[5]) / NormB2_2;
	g[0][0] = -g[0][1]- g[0][2];
	g[0][3] = -g[0][4]- g[0][5];
	g[0][6] = -g[0][7]- g[0][8];

	g[1][1] = (dxyz[1][0] * NormB2 - b2.y * dnorm[0]) / NormB2_2;
	g[1][2] = (dxyz[1][3] * NormB2 - b2.y * dnorm[3]) / NormB2_2;
	g[1][4] = (dxyz[1][1] * NormB2 - b2.y * dnorm[1]) / NormB2_2;
	g[1][5] = (dxyz[1][4] * NormB2 - b2.y * dnorm[4]) / NormB2_2;
	g[1][7] = (dxyz[1][2] * NormB2 - b2.y * dnorm[2]) / NormB2_2;
	g[1][8] = (dxyz[1][5] * NormB2 - b2.y * dnorm[5]) / NormB2_2;
	g[1][0] = -g[1][1] - g[1][2];
	g[1][3] = -g[1][4] - g[1][5];
	g[1][6] = -g[1][7] - g[1][8];

	g[2][1] = (dxyz[2][0] * NormB2 - b2.z * dnorm[0]) / NormB2_2;
	g[2][2] = (dxyz[2][3] * NormB2 - b2.z * dnorm[3]) / NormB2_2;
	g[2][4] = (dxyz[2][1] * NormB2 - b2.z * dnorm[1]) / NormB2_2;
	g[2][5] = (dxyz[2][4] * NormB2 - b2.z * dnorm[4]) / NormB2_2;
	g[2][7] = (dxyz[2][2] * NormB2 - b2.z * dnorm[2]) / NormB2_2;
	g[2][8] = (dxyz[2][5] * NormB2 - b2.z * dnorm[5]) / NormB2_2;
	g[2][0] = -g[2][1] - g[2][2];
	g[2][3] = -g[2][4] - g[2][5];
	g[2][6] = -g[2][7] - g[2][8];
}