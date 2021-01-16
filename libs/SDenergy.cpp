#include "SDenergy.h"

double3 sub(const double3 a, const double3 b)
{
	return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}
double3 add(double3 a, double3 b)
{
	return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
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
Eigen::RowVector3d CastingToEigen(const double3 a) {
	return Eigen::RowVector3d(a.x, a.y, a.z);
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
	Cuda::MemCpyDeviceToHost(curr_x);
	cuda_SD->EnergyAtomic.host_arr[0] = 0;

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		int v0_index = restShapeF(fi, 0);
		int v1_index = restShapeF(fi, 1);
		int v2_index = restShapeF(fi, 2);
		double3 V0 = make_double3(
			curr_x.host_arr[v0_index + cuda_SD->mesh_indices.startVx],
			curr_x.host_arr[v0_index + cuda_SD->mesh_indices.startVy],
			curr_x.host_arr[v0_index + cuda_SD->mesh_indices.startVz]
		);
		double3 V1 = make_double3(
			curr_x.host_arr[v1_index + cuda_SD->mesh_indices.startVx],
			curr_x.host_arr[v1_index + cuda_SD->mesh_indices.startVy],
			curr_x.host_arr[v1_index + cuda_SD->mesh_indices.startVz]
		);
		double3 V2 = make_double3(
			curr_x.host_arr[v2_index + cuda_SD->mesh_indices.startVx],
			curr_x.host_arr[v2_index + cuda_SD->mesh_indices.startVy],
			curr_x.host_arr[v2_index + cuda_SD->mesh_indices.startVz]
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
		const double detJ2 = detJ * detJ;
		const double a2 = a * a;
		const double b2 = b * b;
		const double c2 = c * c;
		const double d2 = d * d;


		cuda_SD->Energy.host_arr[fi] = 0.5 * cuda_SD->restShapeArea.host_arr[fi] *
			(1 + 1 / detJ2) * (a2 + b2 + c2 + d2);
		cuda_SD->EnergyAtomic.host_arr[0] += cuda_SD->Energy.host_arr[fi];
	}
	Cuda::MemCpyHostToDevice(cuda_SD->Energy);
	Cuda::MemCpyHostToDevice(cuda_SD->EnergyAtomic);
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

		Eigen::Matrix<double, 1, 4> de_dJ;
		de_dJ <<
			a + a / det2 - d * Fnorm / det3,
			b + b / det2 + c * Fnorm / det3,
			c + c / det2 + b * Fnorm / det3,
			d + d / det2 - a * Fnorm / det3;
		de_dJ *= cuda_SD->restShapeArea.host_arr[fi];

		Eigen::Matrix<double, 1, 9> dE_dX = de_dJ * dJ_dX(fi, V0, V1, V2);
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

Eigen::Matrix<double, 4, 9> SDenergy::dJ_dX(
	int fi, 
	const double3 V0,
	const double3 V1,
	const double3 V2)
{
	Eigen::Matrix<double, 3, 9> dV0_dX, dV1_dX, dV2_dX;
	dV0_dX.setZero(); dV0_dX(0, 0) = 1; dV0_dX(1, 3) = 1; dV0_dX(2, 6) = 1;
	dV1_dX.setZero(); dV1_dX(0, 1) = 1; dV1_dX(1, 4) = 1; dV1_dX(2, 7) = 1;
	dV2_dX.setZero(); dV2_dX(0, 2) = 1; dV2_dX(1, 5) = 1; dV2_dX(2, 8) = 1;
	double3 e10 = sub(V1, V0);
	double3 e20 = sub(V2, V0);
	double3 B1 = normalize(e10);
	double3 B2 = normalize(cross(cross(B1, e20), B1));

	Eigen::Matrix<double, 3, 9> YY, XX, db1_dX = dB1_dX(fi, e10), db2_dX = dB2_dX(fi, e10, e20);
	XX <<
		(CastingToEigen(V0) * db1_dX + CastingToEigen(B1) * dV0_dX),
		(CastingToEigen(V1) * db1_dX + CastingToEigen(B1) * dV1_dX),
		(CastingToEigen(V2) * db1_dX + CastingToEigen(B1) * dV2_dX);
	YY <<
		(CastingToEigen(V0) * db2_dX + CastingToEigen(B2) * dV0_dX),
		(CastingToEigen(V1) * db2_dX + CastingToEigen(B2) * dV1_dX),
		(CastingToEigen(V2) * db2_dX + CastingToEigen(B2) * dV2_dX);

	Eigen::Matrix<double, 4, 9> dJ;
	dJ.row(0) = CastingToEigen(cuda_SD->D1d.host_arr[fi]) * XX;
	dJ.row(1) = CastingToEigen(cuda_SD->D1d.host_arr[fi]) * YY;
	dJ.row(2) = CastingToEigen(cuda_SD->D2d.host_arr[fi]) * XX;
	dJ.row(3) = CastingToEigen(cuda_SD->D2d.host_arr[fi]) * YY;
	return dJ;
}

Eigen::Matrix<double, 3, 9> SDenergy::dB1_dX(int fi, const double3 e10) 
{
	double Norm = norm(e10);
	double dB1x_dx0 = -(pow(e10.y, 2) + pow(e10.z, 2)) / pow(Norm, 3);
	double dB1y_dy0 = -(pow(e10.x, 2) + pow(e10.z, 2)) / pow(Norm, 3);
	double dB1z_dz0 = -(pow(e10.x, 2) + pow(e10.y, 2)) / pow(Norm, 3);
	double dB1x_dy0 = (e10.y * e10.x) / pow(Norm, 3);
	double dB1x_dz0 = (e10.z * e10.x) / pow(Norm, 3);
	double dB1y_dz0 = (e10.z * e10.y) / pow(Norm, 3);

	Eigen::Matrix<double, 3, 9> g;
	g <<
		dB1x_dx0, -dB1x_dx0, 0, dB1x_dy0, -dB1x_dy0, 0, dB1x_dz0, -dB1x_dz0, 0,
		dB1x_dy0, -dB1x_dy0, 0, dB1y_dy0, -dB1y_dy0, 0, dB1y_dz0, -dB1y_dz0, 0,
		dB1x_dz0, -dB1x_dz0, 0, dB1y_dz0, -dB1y_dz0, 0, dB1z_dz0, -dB1z_dz0, 0;
	return g;
}

Eigen::Matrix<double, 3, 9> SDenergy::dB2_dX(int fi, const double3 e10, const double3 e20)
{
	Eigen::Matrix<double, 3, 9> g;
	double3 b2 = cross(cross(e10, e20), e10);
	double NormB2 = norm(b2);
	double NormB2_2 = pow(NormB2, 2);

	Eigen::Matrix<double, 3, 6> dxyz;
	dxyz.row(0) <<
		-e10.y * e20.y - e10.z * e20.z,
		-e10.x * e20.y + 2 * e10.y * e20.x,
		2 * e10.z * e20.x - e10.x * e20.z,
		pow(e10.y, 2) + pow(e10.z, 2),
		-e10.y * e10.x,
		-e10.x * e10.z;
	dxyz.row(1) <<
		2 * e10.x * e20.y - e10.y * e20.x,
		-e10.z * e20.z - e20.x * e10.x,
		-e10.y * e20.z + 2 * e10.z * e20.y,
		-e10.x * e10.y,
		pow(e10.z, 2) + pow(e10.x, 2),
		-e10.z * e10.y;
	dxyz.row(2) <<
		-e10.z * e20.x + 2 * e10.x * e20.z,
		2 * e10.y * e20.z - e10.z * e20.y,
		-e10.x * e20.x - e10.y * e20.y,
		-e10.x * e10.z,
		-e10.z * e10.y,
		pow(e10.x, 2) + pow(e10.y, 2);

	Eigen::Matrix<double, 6, 1> dnorm;
	dnorm <<
		(b2.x * dxyz(0, 0) + b2.y * dxyz(1, 0) + b2.z * dxyz(2, 0)) / NormB2,
		(b2.x * dxyz(0, 1) + b2.y * dxyz(1, 1) + b2.z * dxyz(2, 1)) / NormB2,
		(b2.x * dxyz(0, 2) + b2.y * dxyz(1, 2) + b2.z * dxyz(2, 2)) / NormB2,
		(b2.x * dxyz(0, 3) + b2.y * dxyz(1, 3) + b2.z * dxyz(2, 3)) / NormB2,
		(b2.x * dxyz(0, 4) + b2.y * dxyz(1, 4) + b2.z * dxyz(2, 4)) / NormB2,
		(b2.x * dxyz(0, 5) + b2.y * dxyz(1, 5) + b2.z * dxyz(2, 5)) / NormB2;

	
	g.row(0) <<
		0,
		(dxyz(0, 0) * NormB2 - b2.x * dnorm[0]) / NormB2_2,
		(dxyz(0, 3) * NormB2 - b2.x * dnorm[3]) / NormB2_2,
		0,
		(dxyz(0, 1) * NormB2 - b2.x * dnorm[1]) / NormB2_2,
		(dxyz(0, 4) * NormB2 - b2.x * dnorm[4]) / NormB2_2,
		0,
		(dxyz(0, 2) * NormB2 - b2.x * dnorm[2]) / NormB2_2,
		(dxyz(0, 5) * NormB2 - b2.x * dnorm[5]) / NormB2_2;
	g.row(1) <<
		0,
		(dxyz(1, 0) * NormB2 - b2.y * dnorm[0]) / NormB2_2,
		(dxyz(1, 3) * NormB2 - b2.y * dnorm[3]) / NormB2_2,
		0,
		(dxyz(1, 1) * NormB2 - b2.y * dnorm[1]) / NormB2_2,
		(dxyz(1, 4) * NormB2 - b2.y * dnorm[4]) / NormB2_2,
		0,
		(dxyz(1, 2) * NormB2 - b2.y * dnorm[2]) / NormB2_2,
		(dxyz(1, 5) * NormB2 - b2.y * dnorm[5]) / NormB2_2;
	g.row(2) <<
		0,
		(dxyz(2, 0) * NormB2 - b2.z * dnorm[0]) / NormB2_2,
		(dxyz(2, 3) * NormB2 - b2.z * dnorm[3]) / NormB2_2,
		0,
		(dxyz(2, 1) * NormB2 - b2.z * dnorm[1]) / NormB2_2,
		(dxyz(2, 4) * NormB2 - b2.z * dnorm[4]) / NormB2_2,
		0,
		(dxyz(2, 2) * NormB2 - b2.z * dnorm[2]) / NormB2_2,
		(dxyz(2, 5) * NormB2 - b2.z * dnorm[5]) / NormB2_2;
	
	for (int c = 0; c < 9; c += 3)
		g.col(c) = -g.col(c + 1) - g.col(c + 2);
	return g;
}