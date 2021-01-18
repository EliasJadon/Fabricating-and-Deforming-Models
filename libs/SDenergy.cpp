#include "SDenergy.h"

double3 sub(const double3 a, const double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
double dot(const double3 a, const double3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
double dot4(const double4 a, const double4 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
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
		const int v0_index = restShapeF(fi, 0);
		const int v1_index = restShapeF(fi, 1);
		const int v2_index = restShapeF(fi, 2);
		const int startX = cuda_SD->mesh_indices.startVx;
		const int startY = cuda_SD->mesh_indices.startVy;
		const int startZ = cuda_SD->mesh_indices.startVz;
		double3 V0 = make_double3(
			X.host_arr[v0_index + startX],
			X.host_arr[v0_index + startY],
			X.host_arr[v0_index + startZ]
		);
		double3 V1 = make_double3(
			X.host_arr[v1_index + startX],
			X.host_arr[v1_index + startY],
			X.host_arr[v1_index + startZ]
		);
		double3 V2 = make_double3(
			X.host_arr[v2_index + startX],
			X.host_arr[v2_index + startY],
			X.host_arr[v2_index + startZ]
		);
		double3 e10 = sub(V1, V0);
		double3 e20 = sub(V2, V0);
		double3 B1 = normalize(e10);
		double3 B2 = normalize(cross(cross(B1, e20), B1));
		double3 Xi = make_double3(dot(V0, B1), dot(V1, B1), dot(V2, B1));
		double3 Yi = make_double3(dot(V0, B2), dot(V1, B2), dot(V2, B2));
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
		double4 dj_dx[9];
		dJ_dX(dj_dx, fi, V0, V1, V2);

		cuda_SD->grad.host_arr[v0_index + cuda_SD->mesh_indices.startVx] += dot4(de_dJ, dj_dx[0]);
		cuda_SD->grad.host_arr[v1_index + cuda_SD->mesh_indices.startVx] += dot4(de_dJ, dj_dx[1]);
		cuda_SD->grad.host_arr[v2_index + cuda_SD->mesh_indices.startVx] += dot4(de_dJ, dj_dx[2]);
		cuda_SD->grad.host_arr[v0_index + cuda_SD->mesh_indices.startVy] += dot4(de_dJ, dj_dx[3]);
		cuda_SD->grad.host_arr[v1_index + cuda_SD->mesh_indices.startVy] += dot4(de_dJ, dj_dx[4]);
		cuda_SD->grad.host_arr[v2_index + cuda_SD->mesh_indices.startVy] += dot4(de_dJ, dj_dx[5]);
		cuda_SD->grad.host_arr[v0_index + cuda_SD->mesh_indices.startVz] += dot4(de_dJ, dj_dx[6]);
		cuda_SD->grad.host_arr[v1_index + cuda_SD->mesh_indices.startVz] += dot4(de_dJ, dj_dx[7]);
		cuda_SD->grad.host_arr[v2_index + cuda_SD->mesh_indices.startVz] += dot4(de_dJ, dj_dx[8]);
	}
	Cuda::MemCpyHostToDevice(cuda_SD->grad);
}

void SDenergy::dJ_dX(
	double4 (&g)[9],
	int fi, 
	const double3 V0,
	const double3 V1,
	const double3 V2)
{
	double3 e10 = sub(V1, V0);
	double3 e20 = sub(V2, V0);
	double3 B1 = normalize(e10);
	double3 B2 = normalize(cross(cross(B1, e20), B1));

	double3 db1_dX[9], db2_dX[9], XX[9], YY[9];
	dB1_dX(db1_dX, fi, e10);
	dB2_dX(db2_dX, fi, e10, e20);
	
	for (int i = 0; i < 9; i++) {
		XX[i].x = dot(V0, db1_dX[i]);
		XX[i].y = dot(V1, db1_dX[i]);
		XX[i].z = dot(V2, db1_dX[i]);
	}
	XX[0].x += B1.x; XX[3].x += B1.y; XX[6].x += B1.z;
	XX[1].y += B1.x; XX[4].y += B1.y; XX[7].y += B1.z;
	XX[2].z += B1.x; XX[5].z += B1.y; XX[8].z += B1.z;

	for (int i = 0; i < 9; i++) {
		YY[i].x = dot(V0, db2_dX[i]);
		YY[i].y = dot(V1, db2_dX[i]);
		YY[i].z = dot(V2, db2_dX[i]);
	}
	YY[0].x += B2.x; YY[3].x += B2.y; YY[6].x += B2.z;
	YY[1].y += B2.x; YY[4].y += B2.y; YY[7].y += B2.z;
	YY[2].z += B2.x; YY[5].z += B2.y; YY[8].z += B2.z;

	for (int i = 0; i < 9; i++) {
		g[i].x = dot(cuda_SD->D1d.host_arr[fi], XX[i]);
		g[i].y = dot(cuda_SD->D1d.host_arr[fi], YY[i]);
		g[i].z = dot(cuda_SD->D2d.host_arr[fi], XX[i]);
		g[i].w = dot(cuda_SD->D2d.host_arr[fi], YY[i]);
	}
}

void SDenergy::dB1_dX(double3 (&g)[9], int fi, const double3 e10)
{
	double Norm = norm(e10);
	double Norm3 = pow(Norm, 3);
	double dB1x_dx0 = -(pow(e10.y, 2) + pow(e10.z, 2)) / Norm3;
	double dB1y_dy0 = -(pow(e10.x, 2) + pow(e10.z, 2)) / Norm3;
	double dB1z_dz0 = -(pow(e10.x, 2) + pow(e10.y, 2)) / Norm3;
	double dB1x_dy0 = (e10.y * e10.x) / Norm3;
	double dB1x_dz0 = (e10.z * e10.x) / Norm3;
	double dB1y_dz0 = (e10.z * e10.y) / Norm3;
	g[0] = make_double3(dB1x_dx0, dB1x_dy0, dB1x_dz0);
	g[1] = mul(-1, g[0]);
	g[2] = make_double3(0, 0, 0);
	g[3] = make_double3(dB1x_dy0, dB1y_dy0, dB1y_dz0);
	g[4] = mul(-1, g[3]);
	g[5] = make_double3(0, 0, 0);
	g[6] = make_double3(dB1x_dz0, dB1y_dz0, dB1z_dz0);
	g[7] = mul(-1, g[6]);
	g[8] = make_double3(0, 0, 0);
}

void SDenergy::dB2_dX(double3 (&g)[9], int fi, const double3 e10, const double3 e20)
{
	double3 b2 = cross(cross(e10, e20), e10);
	double NormB2 = norm(b2);
	double NormB2_2 = pow(NormB2, 2);
	double3 dxyz[6] = {
		make_double3(-e10.y * e20.y - e10.z * e20.z,2 * e10.x * e20.y - e10.y * e20.x,-e10.z * e20.x + 2 * e10.x * e20.z),
		make_double3(-e10.x * e20.y + 2 * e10.y * e20.x,-e10.z * e20.z - e20.x * e10.x,2 * e10.y * e20.z - e10.z * e20.y),
		make_double3(2 * e10.z * e20.x - e10.x * e20.z,-e10.y * e20.z + 2 * e10.z * e20.y,-e10.x * e20.x - e10.y * e20.y),
		make_double3(pow(e10.y, 2) + pow(e10.z, 2),-e10.x * e10.y,-e10.x * e10.z),
		make_double3(-e10.y * e10.x,pow(e10.z, 2) + pow(e10.x, 2),-e10.z * e10.y),
		make_double3(-e10.x * e10.z,-e10.z * e10.y,pow(e10.x, 2) + pow(e10.y, 2))
	};
	double dnorm[6] = {
		dot(b2, dxyz[0]) / NormB2,
		dot(b2, dxyz[1]) / NormB2,
		dot(b2, dxyz[2]) / NormB2,
		dot(b2, dxyz[3]) / NormB2,
		dot(b2, dxyz[4]) / NormB2,
		dot(b2, dxyz[5]) / NormB2
	};
	g[1].x = (dxyz[0].x * NormB2 - b2.x * dnorm[0]) / NormB2_2;
	g[2].x = (dxyz[3].x * NormB2 - b2.x * dnorm[3]) / NormB2_2;
	g[4].x = (dxyz[1].x * NormB2 - b2.x * dnorm[1]) / NormB2_2;
	g[5].x = (dxyz[4].x * NormB2 - b2.x * dnorm[4]) / NormB2_2;
	g[7].x = (dxyz[2].x * NormB2 - b2.x * dnorm[2]) / NormB2_2;
	g[8].x = (dxyz[5].x * NormB2 - b2.x * dnorm[5]) / NormB2_2;
	g[0].x = -g[1].x - g[2].x;
	g[3].x = -g[4].x - g[5].x;
	g[6].x = -g[7].x - g[8].x;

	g[1].y = (dxyz[0].y * NormB2 - b2.y * dnorm[0]) / NormB2_2;
	g[2].y = (dxyz[3].y * NormB2 - b2.y * dnorm[3]) / NormB2_2;
	g[4].y = (dxyz[1].y * NormB2 - b2.y * dnorm[1]) / NormB2_2;
	g[5].y = (dxyz[4].y * NormB2 - b2.y * dnorm[4]) / NormB2_2;
	g[7].y = (dxyz[2].y * NormB2 - b2.y * dnorm[2]) / NormB2_2;
	g[8].y = (dxyz[5].y * NormB2 - b2.y * dnorm[5]) / NormB2_2;
	g[0].y = -g[1].y - g[2].y;
	g[3].y = -g[4].y - g[5].y;
	g[6].y = -g[7].y - g[8].y;

	g[1].z = (dxyz[0].z * NormB2 - b2.z * dnorm[0]) / NormB2_2;
	g[2].z = (dxyz[3].z * NormB2 - b2.z * dnorm[3]) / NormB2_2;
	g[4].z = (dxyz[1].z * NormB2 - b2.z * dnorm[1]) / NormB2_2;
	g[5].z = (dxyz[4].z * NormB2 - b2.z * dnorm[4]) / NormB2_2;
	g[7].z = (dxyz[2].z * NormB2 - b2.z * dnorm[2]) / NormB2_2;
	g[8].z = (dxyz[5].z * NormB2 - b2.z * dnorm[5]) / NormB2_2;
	g[0].z = -g[1].z - g[2].z;
	g[3].z = -g[4].z - g[5].z;
	g[6].z = -g[7].z - g[8].z;
}