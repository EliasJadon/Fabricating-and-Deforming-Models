#include "SymmetricDirichlet.h"
#include "cuda_optimization_lib/Cuda_Basics.cuh"


SymmetricDirichlet::SymmetricDirichlet() {
	name = "Symmetric Dirichlet";
	//w = 0.6;
	w = 0;
	std::cout << "\t" << name << " constructor" << std::endl;
}

SymmetricDirichlet::~SymmetricDirichlet() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void SymmetricDirichlet::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if (restShapeV.size() == 0 || restShapeF.size() == 0)
		throw name + " must define members V,F before init()!";
	
	setRestShapeFromCurrentConfiguration();
}

void SymmetricDirichlet::setRestShapeFromCurrentConfiguration() {
	a.resize(restShapeF.rows());
	b.resize(restShapeF.rows());
	c.resize(restShapeF.rows());
	d.resize(restShapeF.rows());
	detJ.resize(restShapeF.rows());

	Eigen::MatrixX3d D1cols, D2cols;
	OptimizationUtils::computeSurfaceGradientPerFace(restShapeV, restShapeF, D1cols, D2cols);
	D1d = D1cols.transpose();
	D2d = D2cols.transpose();

	//compute the area for each triangle
	igl::doublearea(restShapeV, restShapeF, restShapeArea);
	restShapeArea /= 2;
}

void SymmetricDirichlet::updateX(Cuda::Array<double>& curr_x)
{
	Cuda::MemCpyDeviceToHost(curr_x);
	assert(curr_x.size == (restShapeV.size() + 7 * restShapeF.rows()));
	CurrV.resize(restShapeV.rows(), 3);
	for (int v = 0; v < restShapeV.rows(); v++) {
		CurrV(v, 0) = curr_x.host_arr[v];
		CurrV(v, 1) = curr_x.host_arr[v + restShapeV.rows()];
		CurrV(v, 2) = curr_x.host_arr[v + 2 * restShapeV.rows()];
	}
	//assert(X.rows() == (restShapeV.size() + 7*restShapeF.rows()));
	//CurrV = Eigen::Map<const Eigen::MatrixX3d>(X.middleRows(0, restShapeV.size()).data(), restShapeV.rows(), 3);
	
	OptimizationUtils::LocalBasis(CurrV, restShapeF, B1, B2);
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		Eigen::Vector3d Xi, Yi;
		Xi <<
			CurrV.row(restShapeF(fi, 0)) * B1.row(fi).transpose(),
			CurrV.row(restShapeF(fi, 1)) * B1.row(fi).transpose(),
			CurrV.row(restShapeF(fi, 2)) * B1.row(fi).transpose();
		Yi <<
			CurrV.row(restShapeF(fi, 0)) * B2.row(fi).transpose(),
			CurrV.row(restShapeF(fi, 1)) * B2.row(fi).transpose(),
			CurrV.row(restShapeF(fi, 2)) * B2.row(fi).transpose();

		Eigen::Vector3d Dx = D1d.col(fi);
		Eigen::Vector3d Dy = D2d.col(fi);
		//prepare jacobian		
		a(fi) = Dx.transpose() * Xi;
		b(fi) = Dx.transpose() * Yi;
		c(fi) = Dy.transpose() * Xi;
		d(fi) = Dy.transpose() * Yi;
		detJ(fi) = a(fi) * d(fi) - b(fi) * c(fi);
	}
}

double SymmetricDirichlet::value(Cuda::Array<double>& curr_x, const bool update) {
	return 15;
}

Eigen::Matrix<double, 1, 4> SymmetricDirichlet::dE_dJ(int fi) {
	Eigen::Matrix<double, 1, 4> de_dJ;
	double det2 = pow(detJ(fi), 2);
	double det3 = pow(detJ(fi), 3);
	double Fnorm = pow(a(fi), 2) + pow(b(fi), 2) + pow(c(fi), 2) + pow(d(fi), 2);
	de_dJ <<
		a(fi) + a(fi) / det2 - d(fi) * Fnorm / det3,
		b(fi) + b(fi) / det2 + c(fi) * Fnorm / det3,
		c(fi) + c(fi) / det2 + b(fi) * Fnorm / det3,
		d(fi) + d(fi) / det2 - a(fi) * Fnorm / det3;
	de_dJ *= restShapeArea[fi];
	return de_dJ;
}

Cuda::Array<double>* SymmetricDirichlet::gradient(Cuda::Array<double>& X, const bool update)
{
	return NULL;
}

Eigen::Matrix<double, 4, 9> SymmetricDirichlet::dJ_dX(int fi) {
	Eigen::Vector3d Dx = D1d.col(fi);
	Eigen::Vector3d Dy = D2d.col(fi);
	Eigen::Matrix<double, 1, 3> V0 = CurrV.row(restShapeF(fi, 0));
	Eigen::Matrix<double, 1, 3> V1 = CurrV.row(restShapeF(fi, 1));
	Eigen::Matrix<double, 1, 3> V2 = CurrV.row(restShapeF(fi, 2));
	Eigen::Matrix<double, 3, 9> dV0_dX, dV1_dX, dV2_dX;
	dV0_dX.setZero(); dV0_dX(0, 0) = 1; dV0_dX(1, 3) = 1; dV0_dX(2, 6) = 1;
	dV1_dX.setZero(); dV1_dX(0, 1) = 1; dV1_dX(1, 4) = 1; dV1_dX(2, 7) = 1;
	dV2_dX.setZero(); dV2_dX(0, 2) = 1; dV2_dX(1, 5) = 1; dV2_dX(2, 8) = 1;

	Eigen::Matrix<double, 3, 9> YY, XX, db1_dX = dB1_dX(fi), db2_dX = dB2_dX(fi);
	XX <<
		(V0 * db1_dX + B1.row(fi)*dV0_dX),
		(V1 * db1_dX + B1.row(fi)*dV1_dX),
		(V2 * db1_dX + B1.row(fi)*dV2_dX);
	YY <<
		(V0 * db2_dX + B2.row(fi)*dV0_dX),
		(V1 * db2_dX + B2.row(fi)*dV1_dX),
		(V2 * db2_dX + B2.row(fi)*dV2_dX);

	Eigen::Matrix<double, 4, 9> dJ;
	dJ.row(0) = Dx.transpose()*XX;
	dJ.row(1) = Dx.transpose()*YY;
	dJ.row(2) = Dy.transpose()*XX;
	dJ.row(3) = Dy.transpose()*YY;
	return dJ;
}

Eigen::Matrix<double, 3, 9> SymmetricDirichlet::dB1_dX(int fi) {
	Eigen::Matrix<double, 3, 9> g;
	Eigen::Matrix<double, 3, 1> V0 = CurrV.row(restShapeF(fi, 0));
	Eigen::Matrix<double, 3, 1> V1 = CurrV.row(restShapeF(fi, 1));
	Eigen::Matrix<double, 3, 1> V2 = CurrV.row(restShapeF(fi, 2));
	double Norm = (V1 - V0).norm();
	double Qx = V1[0] - V0[0]; // x1 - x0
	double Qy = V1[1] - V0[1]; // y1 - y0
	double Qz = V1[2] - V0[2]; // z1 - z0	
	double dB1x_dx0 = -(pow(Qy, 2) + pow(Qz, 2)) / pow(Norm, 3);
	double dB1y_dy0 = -(pow(Qx, 2) + pow(Qz, 2)) / pow(Norm, 3);
	double dB1z_dz0 = -(pow(Qx, 2) + pow(Qy, 2)) / pow(Norm, 3);
	double dB1x_dy0 = (Qy*Qx) / pow(Norm, 3);
	double dB1x_dz0 = (Qz*Qx) / pow(Norm, 3);
	double dB1y_dz0 = (Qz*Qy) / pow(Norm, 3);
	g <<
		dB1x_dx0, -dB1x_dx0, 0, dB1x_dy0, -dB1x_dy0, 0, dB1x_dz0, -dB1x_dz0, 0,
		dB1x_dy0, -dB1x_dy0, 0, dB1y_dy0, -dB1y_dy0, 0, dB1y_dz0, -dB1y_dz0, 0,
		dB1x_dz0, -dB1x_dz0, 0, dB1y_dz0, -dB1y_dz0, 0, dB1z_dz0, -dB1z_dz0, 0;
	return g;
}

Eigen::Matrix<double, 3, 9> SymmetricDirichlet::dB2_dX(int fi) {
	Eigen::Matrix<double, 3, 9> g;
	Eigen::Matrix<double, 3, 1> V0 = CurrV.row(restShapeF(fi, 0));
	Eigen::Matrix<double, 3, 1> V1 = CurrV.row(restShapeF(fi, 1));
	Eigen::Matrix<double, 3, 1> V2 = CurrV.row(restShapeF(fi, 2));
	double Qx = V1[0] - V0[0]; // Qx = x1 - x0
	double Qy = V1[1] - V0[1]; // Qy = y1 - y0
	double Qz = V1[2] - V0[2]; // Qz = z1 - z0
	double Wx = V2[0] - V0[0]; // Wx = x2 - x0
	double Wy = V2[1] - V0[1]; // Wy = y2 - y0
	double Wz = V2[2] - V0[2]; // Wz = z2 - z0	
	Eigen::Matrix<double, 3, 1> b2 = -((V1 - V0).cross((V1 - V0).cross(V2 - V0)));
	double NormB2 = b2.norm();
	double NormB2_2 = pow(NormB2, 2);

	Eigen::Matrix<double, 3, 6> dxyz;
	dxyz.row(0) <<
		-Qy * Wy - Qz * Wz,
		-Qx * Wy + 2 * Qy * Wx,
		2 * Qz*Wx - Qx * Wz,
		pow(Qy, 2) + pow(Qz, 2),
		-Qy * Qx,
		-Qx * Qz;
	dxyz.row(1) <<
		2 * Qx*Wy - Qy * Wx,
		-Qz * Wz - Wx * Qx,
		-Qy * Wz + 2 * Qz*Wy,
		-Qx * Qy,
		pow(Qz, 2) + pow(Qx, 2),
		-Qz * Qy;
	dxyz.row(2) <<
		-Qz * Wx + 2 * Qx*Wz,
		2 * Qy*Wz - Qz * Wy,
		-Qx * Wx - Qy * Wy,
		-Qx * Qz,
		-Qz * Qy,
		pow(Qx, 2) + pow(Qy, 2);

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
			(dxyz(xyz, 0)*NormB2 - b2[xyz] * dnorm[0]) / NormB2_2,
			(dxyz(xyz, 3)*NormB2 - b2[xyz] * dnorm[3]) / NormB2_2,
			0,
			(dxyz(xyz, 1)*NormB2 - b2[xyz] * dnorm[1]) / NormB2_2,
			(dxyz(xyz, 4)*NormB2 - b2[xyz] * dnorm[4]) / NormB2_2,
			0,
			(dxyz(xyz, 2)*NormB2 - b2[xyz] * dnorm[2]) / NormB2_2,
			(dxyz(xyz, 5)*NormB2 - b2[xyz] * dnorm[5]) / NormB2_2;
	}
	for (int c = 0; c < 9; c += 3)
		g.col(c) = -g.col(c + 1) - g.col(c + 2);
	return g;
}
