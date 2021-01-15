#include "SDenergy.h"

SDenergy::SDenergy(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) {
	init_mesh(V, F);
	name = "Symmetric Dirichlet";
	w = 0.6;

	Efi.resize(F.rows());
	Efi.setZero();

	Eigen::MatrixX3d D1cols, D2cols;
	OptimizationUtils::computeSurfaceGradientPerFace(restShapeV, restShapeF, D1cols, D2cols);
	D1d = D1cols.transpose();
	D2d = D2cols.transpose();

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
	
	Cuda::MemCpyHostToDevice(cuda_SD->Energy);
	Cuda::MemCpyHostToDevice(cuda_SD->restShapeF);
	Cuda::MemCpyHostToDevice(cuda_SD->restShapeArea);
	Cuda::MemCpyHostToDevice(cuda_SD->grad);
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
		Eigen::RowVector3d V0(
			curr_x.host_arr[v0_index + cuda_SD->mesh_indices.startVx],
			curr_x.host_arr[v0_index + cuda_SD->mesh_indices.startVy],
			curr_x.host_arr[v0_index + cuda_SD->mesh_indices.startVz]
		);
		Eigen::RowVector3d V1(
			curr_x.host_arr[v1_index + cuda_SD->mesh_indices.startVx],
			curr_x.host_arr[v1_index + cuda_SD->mesh_indices.startVy],
			curr_x.host_arr[v1_index + cuda_SD->mesh_indices.startVz]
		);
		Eigen::RowVector3d V2(
			curr_x.host_arr[v2_index + cuda_SD->mesh_indices.startVx],
			curr_x.host_arr[v2_index + cuda_SD->mesh_indices.startVy],
			curr_x.host_arr[v2_index + cuda_SD->mesh_indices.startVz]
		);

		Eigen::RowVector3d B1 = (V1 - V0).normalized();
		Eigen::RowVector3d temp = B1.cross(V2 - V0).normalized();
		Eigen::RowVector3d B2 = temp.cross(B1).normalized();

		Eigen::Vector3d Xi, Yi;
		Xi <<
			V0 * B1.transpose(),
			V1* B1.transpose(),
			V2* B1.transpose();
		Yi <<
			V0 * B2.transpose(),
			V1* B2.transpose(),
			V2* B2.transpose();

		Eigen::Vector3d Dx = D1d.col(fi);
		Eigen::Vector3d Dy = D2d.col(fi);
		//prepare jacobian		
		const double a = Dx.transpose() * Xi;
		const double b = Dx.transpose() * Yi;
		const double c = Dy.transpose() * Xi;
		const double d = Dy.transpose() * Yi;
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

	/*if (update) {
		Efi = Energy;
		energy_value = total_energy;
	}*/
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
		Eigen::RowVector3d V0(
			X.host_arr[v0_index + cuda_SD->mesh_indices.startVx],
			X.host_arr[v0_index + cuda_SD->mesh_indices.startVy],
			X.host_arr[v0_index + cuda_SD->mesh_indices.startVz]
		);
		Eigen::RowVector3d V1(
			X.host_arr[v1_index + cuda_SD->mesh_indices.startVx],
			X.host_arr[v1_index + cuda_SD->mesh_indices.startVy],
			X.host_arr[v1_index + cuda_SD->mesh_indices.startVz]
		);
		Eigen::RowVector3d V2(
			X.host_arr[v2_index + cuda_SD->mesh_indices.startVx],
			X.host_arr[v2_index + cuda_SD->mesh_indices.startVy],
			X.host_arr[v2_index + cuda_SD->mesh_indices.startVz]
		);

		Eigen::RowVector3d B1 = (V1 - V0).normalized();
		Eigen::RowVector3d temp = B1.cross(V2 - V0).normalized();
		Eigen::RowVector3d B2 = temp.cross(B1).normalized();

		Eigen::Vector3d Xi, Yi;
		Xi <<
			V0 * B1.transpose(),
			V1* B1.transpose(),
			V2* B1.transpose();
		Yi <<
			V0 * B2.transpose(),
			V1* B2.transpose(),
			V2* B2.transpose();

		Eigen::Vector3d Dx = D1d.col(fi);
		Eigen::Vector3d Dy = D2d.col(fi);
		//prepare jacobian		
		const double a = Dx.transpose() * Xi;
		const double b = Dx.transpose() * Yi;
		const double c = Dy.transpose() * Xi;
		const double d = Dy.transpose() * Yi;
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
	const Eigen::RowVector3d V0,
	const Eigen::RowVector3d V1,
	const Eigen::RowVector3d V2) 
{
	Eigen::Vector3d Dx = D1d.col(fi);
	Eigen::Vector3d Dy = D2d.col(fi);
	Eigen::Matrix<double, 3, 9> dV0_dX, dV1_dX, dV2_dX;
	dV0_dX.setZero(); dV0_dX(0, 0) = 1; dV0_dX(1, 3) = 1; dV0_dX(2, 6) = 1;
	dV1_dX.setZero(); dV1_dX(0, 1) = 1; dV1_dX(1, 4) = 1; dV1_dX(2, 7) = 1;
	dV2_dX.setZero(); dV2_dX(0, 2) = 1; dV2_dX(1, 5) = 1; dV2_dX(2, 8) = 1;

	Eigen::RowVector3d B1 = (V1 - V0).normalized();
	Eigen::RowVector3d temp = B1.cross(V2 - V0).normalized();
	Eigen::RowVector3d B2 = temp.cross(B1).normalized();

	Eigen::Matrix<double, 3, 9> YY, XX, db1_dX = dB1_dX(fi, V1 - V0), db2_dX = dB2_dX(fi, V1 - V0, V2 - V0);
	XX <<
		(V0 * db1_dX + B1 * dV0_dX),
		(V1 * db1_dX + B1 * dV1_dX),
		(V2 * db1_dX + B1 * dV2_dX);
	YY <<
		(V0 * db2_dX + B2 * dV0_dX),
		(V1 * db2_dX + B2 * dV1_dX),
		(V2 * db2_dX + B2 * dV2_dX);

	Eigen::Matrix<double, 4, 9> dJ;
	dJ.row(0) = Dx.transpose() * XX;
	dJ.row(1) = Dx.transpose() * YY;
	dJ.row(2) = Dy.transpose() * XX;
	dJ.row(3) = Dy.transpose() * YY;
	return dJ;
}

Eigen::Matrix<double, 3, 9> SDenergy::dB1_dX(int fi, const Eigen::RowVector3d e10) 
{
	double Norm = e10.norm();
	double dB1x_dx0 = -(pow(e10(1), 2) + pow(e10(2), 2)) / pow(Norm, 3);
	double dB1y_dy0 = -(pow(e10(0), 2) + pow(e10(2), 2)) / pow(Norm, 3);
	double dB1z_dz0 = -(pow(e10(0), 2) + pow(e10(1), 2)) / pow(Norm, 3);
	double dB1x_dy0 = (e10(1) * e10(0)) / pow(Norm, 3);
	double dB1x_dz0 = (e10(2) * e10(0)) / pow(Norm, 3);
	double dB1y_dz0 = (e10(2) * e10(1)) / pow(Norm, 3);

	Eigen::Matrix<double, 3, 9> g;
	g <<
		dB1x_dx0, -dB1x_dx0, 0, dB1x_dy0, -dB1x_dy0, 0, dB1x_dz0, -dB1x_dz0, 0,
		dB1x_dy0, -dB1x_dy0, 0, dB1y_dy0, -dB1y_dy0, 0, dB1y_dz0, -dB1y_dz0, 0,
		dB1x_dz0, -dB1x_dz0, 0, dB1y_dz0, -dB1y_dz0, 0, dB1z_dz0, -dB1z_dz0, 0;
	return g;
}

Eigen::Matrix<double, 3, 9> SDenergy::dB2_dX(int fi, const Eigen::RowVector3d e10, const Eigen::RowVector3d e20)
{
	Eigen::Matrix<double, 3, 9> g;
	Eigen::Matrix<double, 3, 1> b2 = -(e10.cross(e10.cross(e20)));
	double NormB2 = b2.norm();
	double NormB2_2 = pow(NormB2, 2);

	Eigen::Matrix<double, 3, 6> dxyz;
	dxyz.row(0) <<
		-e10(1) * e20(1) - e10(2) * e20(2),
		-e10(0) * e20(1) + 2 * e10(1) * e20(0),
		2 * e10(2) * e20(0) - e10(0) * e20(2),
		pow(e10(1), 2) + pow(e10(2), 2),
		-e10(1) * e10(0),
		-e10(0) * e10(2);
	dxyz.row(1) <<
		2 * e10(0) * e20(1) - e10(1) * e20(0),
		-e10(2) * e20(2) - e20(0) * e10(0),
		-e10(1) * e20(2) + 2 * e10(2) * e20(1),
		-e10(0) * e10(1),
		pow(e10(2), 2) + pow(e10(0), 2),
		-e10(2) * e10(1);
	dxyz.row(2) <<
		-e10(2) * e20(0) + 2 * e10(0) * e20(2),
		2 * e10(1) * e20(2) - e10(2) * e20(1),
		-e10(0) * e20(0) - e10(1) * e20(1),
		-e10(0) * e10(2),
		-e10(2) * e10(1),
		pow(e10(0), 2) + pow(e10(1), 2);

	Eigen::Matrix<double, 6, 1> dnorm;
	dnorm <<
		(b2[0] * dxyz(0, 0) + b2[1] * dxyz(1, 0) + b2[2] * dxyz(2, 0)) / NormB2,
		(b2[0] * dxyz(0, 1) + b2[1] * dxyz(1, 1) + b2[2] * dxyz(2, 1)) / NormB2,
		(b2[0] * dxyz(0, 2) + b2[1] * dxyz(1, 2) + b2[2] * dxyz(2, 2)) / NormB2,
		(b2[0] * dxyz(0, 3) + b2[1] * dxyz(1, 3) + b2[2] * dxyz(2, 3)) / NormB2,
		(b2[0] * dxyz(0, 4) + b2[1] * dxyz(1, 4) + b2[2] * dxyz(2, 4)) / NormB2,
		(b2[0] * dxyz(0, 5) + b2[1] * dxyz(1, 5) + b2[2] * dxyz(2, 5)) / NormB2;

	for (int xyz = 0; xyz < 3; xyz++) {
		g.row(xyz) <<
			0,
			(dxyz(xyz, 0) * NormB2 - b2[xyz] * dnorm[0]) / NormB2_2,
			(dxyz(xyz, 3) * NormB2 - b2[xyz] * dnorm[3]) / NormB2_2,
			0,
			(dxyz(xyz, 1) * NormB2 - b2[xyz] * dnorm[1]) / NormB2_2,
			(dxyz(xyz, 4) * NormB2 - b2[xyz] * dnorm[4]) / NormB2_2,
			0,
			(dxyz(xyz, 2) * NormB2 - b2[xyz] * dnorm[2]) / NormB2_2,
			(dxyz(xyz, 5) * NormB2 - b2[xyz] * dnorm[5]) / NormB2_2;
	}
	for (int c = 0; c < 9; c += 3)
		g.col(c) = -g.col(c + 1) - g.col(c + 2);
	return g;
}