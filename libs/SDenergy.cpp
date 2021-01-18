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
	double Norm_e10_3 = pow(norm(e10), 3);


	double3 db1_dX, db2_dX[9], XX, YY;
	dB2_dX(db2_dX, fi, e10, e20);
	
	db1_dX = make_double3((-(pow(e10.y, 2) + pow(e10.z, 2)) / Norm_e10_3), ((e10.y * e10.x) / Norm_e10_3), ((e10.z * e10.x) / Norm_e10_3));
	XX = make_double3(dot(V0, db1_dX) + B1.x, dot(V1, db1_dX), dot(V2, db1_dX));
	YY = make_double3(dot(V0, db2_dX[0]) + B2.x, dot(V1, db2_dX[0]), dot(V2, db2_dX[0]));
	g[0] = make_double4(
		dot(cuda_SD->D1d.host_arr[fi], XX),
		dot(cuda_SD->D1d.host_arr[fi], YY),
		dot(cuda_SD->D2d.host_arr[fi], XX),
		dot(cuda_SD->D2d.host_arr[fi], YY)
	);

	db1_dX = make_double3(-(-(pow(e10.y, 2) + pow(e10.z, 2)) / Norm_e10_3), -((e10.y * e10.x) / Norm_e10_3), -((e10.z * e10.x) / Norm_e10_3));
	XX = make_double3(dot(V0, db1_dX), dot(V1, db1_dX) + B1.x, dot(V2, db1_dX));
	YY = make_double3(dot(V0, db2_dX[1]), dot(V1, db2_dX[1]) + B2.x, dot(V2, db2_dX[1]));
	g[1] = make_double4(
		dot(cuda_SD->D1d.host_arr[fi], XX),
		dot(cuda_SD->D1d.host_arr[fi], YY),
		dot(cuda_SD->D2d.host_arr[fi], XX),
		dot(cuda_SD->D2d.host_arr[fi], YY)
	);

	XX = make_double3(0, 0, B1.x);
	YY = make_double3(dot(V0, db2_dX[2]), dot(V1, db2_dX[2]), dot(V2, db2_dX[2]) + B2.x);
	g[2] = make_double4(
		dot(cuda_SD->D1d.host_arr[fi], XX),
		dot(cuda_SD->D1d.host_arr[fi], YY),
		dot(cuda_SD->D2d.host_arr[fi], XX),
		dot(cuda_SD->D2d.host_arr[fi], YY)
	);

	db1_dX = make_double3(((e10.y * e10.x) / Norm_e10_3), (-(pow(e10.x, 2) + pow(e10.z, 2)) / Norm_e10_3), ((e10.z * e10.y) / Norm_e10_3));
	XX = make_double3(dot(V0, db1_dX) + B1.y, dot(V1, db1_dX), dot(V2, db1_dX));
	YY = make_double3(dot(V0, db2_dX[3]) + B2.y, dot(V1, db2_dX[3]), dot(V2, db2_dX[3]));
	g[3] = make_double4(
		dot(cuda_SD->D1d.host_arr[fi], XX),
		dot(cuda_SD->D1d.host_arr[fi], YY),
		dot(cuda_SD->D2d.host_arr[fi], XX),
		dot(cuda_SD->D2d.host_arr[fi], YY)
	);

	db1_dX = make_double3(-((e10.y * e10.x) / Norm_e10_3), -(-(pow(e10.x, 2) + pow(e10.z, 2)) / Norm_e10_3), -((e10.z * e10.y) / Norm_e10_3));
	XX = make_double3(dot(V0, db1_dX), dot(V1, db1_dX) + B1.y, dot(V2, db1_dX));
	YY = make_double3(dot(V0, db2_dX[4]), dot(V1, db2_dX[4]) + B2.y, dot(V2, db2_dX[4]));
	g[4] = make_double4(
		dot(cuda_SD->D1d.host_arr[fi], XX),
		dot(cuda_SD->D1d.host_arr[fi], YY),
		dot(cuda_SD->D2d.host_arr[fi], XX),
		dot(cuda_SD->D2d.host_arr[fi], YY)
	);

	XX = make_double3(0, 0, B1.y);
	YY = make_double3(dot(V0, db2_dX[5]), dot(V1, db2_dX[5]), dot(V2, db2_dX[5]) + B2.y);
	g[5] = make_double4(
		dot(cuda_SD->D1d.host_arr[fi], XX),
		dot(cuda_SD->D1d.host_arr[fi], YY),
		dot(cuda_SD->D2d.host_arr[fi], XX),
		dot(cuda_SD->D2d.host_arr[fi], YY)
	);

	db1_dX = make_double3(((e10.z * e10.x) / Norm_e10_3), ((e10.z * e10.y) / Norm_e10_3), (-(pow(e10.x, 2) + pow(e10.y, 2)) / Norm_e10_3));
	XX = make_double3(dot(V0, db1_dX) + B1.z, dot(V1, db1_dX), dot(V2, db1_dX));
	YY = make_double3(dot(V0, db2_dX[6]) + B2.z, dot(V1, db2_dX[6]), dot(V2, db2_dX[6]));
	g[6] = make_double4(
		dot(cuda_SD->D1d.host_arr[fi], XX),
		dot(cuda_SD->D1d.host_arr[fi], YY),
		dot(cuda_SD->D2d.host_arr[fi], XX),
		dot(cuda_SD->D2d.host_arr[fi], YY)
	);

	db1_dX = make_double3(-((e10.z * e10.x) / Norm_e10_3), -((e10.z * e10.y) / Norm_e10_3), -(-(pow(e10.x, 2) + pow(e10.y, 2)) / Norm_e10_3));
	XX = make_double3(dot(V0, db1_dX), dot(V1, db1_dX) + B1.z, dot(V2, db1_dX));
	YY = make_double3(dot(V0, db2_dX[7]), dot(V1, db2_dX[7]) + B2.z, dot(V2, db2_dX[7]));
	g[7] = make_double4(
		dot(cuda_SD->D1d.host_arr[fi], XX),
		dot(cuda_SD->D1d.host_arr[fi], YY),
		dot(cuda_SD->D2d.host_arr[fi], XX),
		dot(cuda_SD->D2d.host_arr[fi], YY)
	);

	XX = make_double3(0, 0, B1.z);
	YY = make_double3(dot(V0, db2_dX[8]), dot(V1, db2_dX[8]), dot(V2, db2_dX[8]) + B2.z);
	g[8] = make_double4(
		dot(cuda_SD->D1d.host_arr[fi], XX),
		dot(cuda_SD->D1d.host_arr[fi], YY),
		dot(cuda_SD->D2d.host_arr[fi], XX),
		dot(cuda_SD->D2d.host_arr[fi], YY)
	);
}

void SDenergy::dB2_dX(double3 (&g)[9], int fi, const double3 e10, const double3 e20)
{
	double3 B2_b2 = cross(cross(e10, e20), e10);
	double Norm_B2 = norm(B2_b2);
	double Norm_B2_2 = pow(Norm_B2, 2);
	double3 B2_dxyz0, B2_dxyz1;
	double B2_dnorm0, B2_dnorm1;

	B2_dxyz0 = make_double3(-e10.y * e20.y - e10.z * e20.z, 2 * e10.x * e20.y - e10.y * e20.x, -e10.z * e20.x + 2 * e10.x * e20.z);
	B2_dxyz1 = make_double3(pow(e10.y, 2) + pow(e10.z, 2), -e10.x * e10.y, -e10.x * e10.z);
	B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
	B2_dnorm1 = dot(B2_b2, B2_dxyz1) / Norm_B2;
	g[0] = make_double3(
		-((B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.x * Norm_B2 - B2_b2.x * B2_dnorm1) / Norm_B2_2),
		-((B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.y * Norm_B2 - B2_b2.y * B2_dnorm1) / Norm_B2_2),
		-((B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.z * Norm_B2 - B2_b2.z * B2_dnorm1) / Norm_B2_2)
	);

	B2_dxyz0 = make_double3(-e10.y * e20.y - e10.z * e20.z, 2 * e10.x * e20.y - e10.y * e20.x, -e10.z * e20.x + 2 * e10.x * e20.z);
	B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
	g[1] = make_double3(
		(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
		(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
		(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
	);

	B2_dxyz0 = make_double3(pow(e10.y, 2) + pow(e10.z, 2), -e10.x * e10.y, -e10.x * e10.z);
	B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
	g[2] = make_double3(
		(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
		(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
		(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
	);

	B2_dxyz0 = make_double3(-e10.x * e20.y + 2 * e10.y * e20.x, -e10.z * e20.z - e20.x * e10.x, 2 * e10.y * e20.z - e10.z * e20.y);
	B2_dxyz1 = make_double3(-e10.y * e10.x, pow(e10.z, 2) + pow(e10.x, 2), -e10.z * e10.y);
	B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
	B2_dnorm1 = dot(B2_b2, B2_dxyz1) / Norm_B2;
	g[3] = make_double3(
		-((B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.x * Norm_B2 - B2_b2.x * B2_dnorm1) / Norm_B2_2),
		-((B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.y * Norm_B2 - B2_b2.y * B2_dnorm1) / Norm_B2_2),
		-((B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.z * Norm_B2 - B2_b2.z * B2_dnorm1) / Norm_B2_2)
	);

	B2_dxyz0 = make_double3(-e10.x * e20.y + 2 * e10.y * e20.x, -e10.z * e20.z - e20.x * e10.x, 2 * e10.y * e20.z - e10.z * e20.y);
	B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
	g[4] = make_double3(
		(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
		(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
		(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
	);

	B2_dxyz0 = make_double3(-e10.y * e10.x, pow(e10.z, 2) + pow(e10.x, 2), -e10.z * e10.y);
	B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
	g[5] = make_double3(
		(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
		(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
		(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
	);

	B2_dxyz0 = make_double3(2 * e10.z * e20.x - e10.x * e20.z, -e10.y * e20.z + 2 * e10.z * e20.y, -e10.x * e20.x - e10.y * e20.y);
	B2_dxyz1 = make_double3(-e10.x * e10.z, -e10.z * e10.y, pow(e10.x, 2) + pow(e10.y, 2));
	B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
	B2_dnorm1 = dot(B2_b2, B2_dxyz1) / Norm_B2;
	g[6] = make_double3(
		-((B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.x * Norm_B2 - B2_b2.x * B2_dnorm1) / Norm_B2_2),
		-((B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.y * Norm_B2 - B2_b2.y * B2_dnorm1) / Norm_B2_2),
		-((B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2) - ((B2_dxyz1.z * Norm_B2 - B2_b2.z * B2_dnorm1) / Norm_B2_2)
	);

	B2_dxyz0 = make_double3(2 * e10.z * e20.x - e10.x * e20.z, -e10.y * e20.z + 2 * e10.z * e20.y, -e10.x * e20.x - e10.y * e20.y);
	B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
	g[7] = make_double3(
		(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
		(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
		(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
	);

	B2_dxyz0 = make_double3(-e10.x * e10.z, -e10.z * e10.y, pow(e10.x, 2) + pow(e10.y, 2));
	B2_dnorm0 = dot(B2_b2, B2_dxyz0) / Norm_B2;
	g[8] = make_double3(
		(B2_dxyz0.x * Norm_B2 - B2_b2.x * B2_dnorm0) / Norm_B2_2,
		(B2_dxyz0.y * Norm_B2 - B2_b2.y * B2_dnorm0) / Norm_B2_2,
		(B2_dxyz0.z * Norm_B2 - B2_b2.z * B2_dnorm0) / Norm_B2_2
	);
}