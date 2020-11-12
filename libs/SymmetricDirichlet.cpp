#include "SymmetricDirichlet.h"

extern int printCuda();
SymmetricDirichlet::SymmetricDirichlet() {
	printCuda();
	name = "Symmetric Dirichlet";
	w = 0.6;
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
	init_hessian();
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

void SymmetricDirichlet::updateX(const Eigen::VectorXd& X)
{
	assert(X.rows() == (restShapeV.size() + 7*restShapeF.rows()));
	CurrV = Eigen::Map<const Eigen::MatrixX3d>(X.middleRows(0, restShapeV.size()).data(), restShapeV.rows(), 3);
	
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

double SymmetricDirichlet::value(const bool update) {
	Eigen::VectorXd Energy(restShapeF.rows());
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		Energy(fi) = 0.5 * (1 + 1/ pow(detJ(fi),2)) * (pow(a(fi),2)+ pow(b(fi), 2)+ pow(c(fi), 2)+ pow(d(fi), 2));
	}
	double total_energy = restShapeArea.transpose() * Energy;
	
	if (update) {
		Efi = Energy;
		energy_value = total_energy;
	}
	return total_energy;
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

void SymmetricDirichlet::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(restShapeV.size() + 7*restShapeF.rows());
	g.setZero();

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		Eigen::Matrix<double, 1, 9> dE_dX = dE_dJ(fi) * dJ_dX(fi);
		for (int vi = 0; vi < 3; vi++)
			for (int xyz = 0; xyz < 3; xyz++)
				g[restShapeF(fi, vi) + (xyz*restShapeV.rows())] += dE_dX[xyz*3 + vi];
	}

	if (update)
		gradient_norm = g.norm();
}

void SymmetricDirichlet::init_hessian() {

}

void SymmetricDirichlet::hessian() {
	II.clear();
	JJ.clear();
	SS.clear();
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		Eigen::Matrix<double, 1, 4> de_dJ = dE_dJ(fi);
		Eigen::Matrix<double, 4, 4> dE_dJdJ;
		Eigen::Matrix<double, 9, 9> dE_dXdX;
	
		
		double det = detJ(fi);
		double det2 = pow(detJ(fi),2);
		double det3 = pow(detJ(fi),3);
		double det4 = pow(detJ(fi),4);
		double Fnorm = pow(a(fi),2) + pow(b(fi), 2) + pow(c(fi), 2) + pow(d(fi), 2);
		
	
		double aa = 1
			+ (1 / det2)
			- ((4 * a(fi)*d(fi)) / det3)
			+ ((3 * pow(d(fi), 2)*Fnorm) / det4);
	
		double bb = 1
			+ (1 / det2)
			+ ((4 * b(fi)*c(fi)) / det3)
			+ ((3 * pow(c(fi), 2)*Fnorm) / det4);
	
		double cc = 1
			+ (1 / det2)
			+ ((4 * b(fi)*c(fi)) / det3)
			+ ((3 * pow(b(fi), 2)*Fnorm) / det4);
	
		double dd = 1
			+ (1 / det2)
			- ((4 * a(fi)*d(fi)) / det3)
			+ ((3 * pow(a(fi), 2)*Fnorm) / det4);
	
		double ab = (-3 * c(fi)*d(fi)*Fnorm) + (2 * (a(fi)*c(fi) - b(fi) * d(fi))*det);
		ab /= det4;
	
		double ac = (-3 * b(fi)*d(fi)*Fnorm) + (2 * (a(fi)*b(fi) - c(fi) * d(fi))*det);
		ac /= det4;
	
		double ad = (3 * a(fi)*d(fi)*Fnorm) - ((2 * pow(a(fi), 2) + 2 * pow(d(fi), 2) + Fnorm)*det);
		ad /= det4;
	
		double bc = (3 * b(fi)*c(fi)*Fnorm) + ((2 * pow(b(fi), 2) + 2 * pow(c(fi), 2) + Fnorm)*det);
		bc /= det4;
	
		double bd = (-3 * a(fi)*c(fi)*Fnorm) + (2 * (c(fi)*d(fi) - a(fi) * b(fi))*det);
		bd /= det4;
	
		double cd = (-3 * a(fi)*b(fi)*Fnorm) + (2 * (b(fi)*d(fi) - a(fi) * c(fi))*det);
		cd /= det4;
	
		dE_dJdJ <<
			aa, ab, ac, ad,
			ab, bb, bc, bd,
			ac, bc, cc, cd,
			ad, bd, cd, dd;
		dE_dJdJ *= restShapeArea[fi];
		
		Eigen::Matrix<double, 4, 9> dj_dX = dJ_dX(fi);
		Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 4> ddj_dXdX = ddJ_dXdX(fi);
				
		dE_dXdX = dj_dX.transpose() * dE_dJdJ * dj_dX +
			de_dJ[0] * ddj_dXdX[0] +
			de_dJ[1] * ddj_dXdX[1] +
			de_dJ[2] * ddj_dXdX[2] +
			de_dJ[3] * ddj_dXdX[3];
							
		for (int v1 = 0; v1 < 3; v1++) {
			for (int v2 = 0; v2 < 3; v2++) {
				for (int xyz1 = 0; xyz1 < 3; xyz1++) {
					for (int xyz2 = 0; xyz2 < 3; xyz2++) {
						int global_i = restShapeF(fi, v1) + (xyz1*restShapeV.rows());
						int global_j = restShapeF(fi, v2) + (xyz2*restShapeV.rows());
						if (global_i <= global_j) {
							II.push_back(global_i);
							JJ.push_back(global_j);
							SS.push_back(dE_dXdX(3*xyz1 + v1, 3*xyz2 + v2));
						}
					}
				}
			}
		}
	}
	II.push_back(restShapeV.size() + 7*restShapeF.rows() - 1);
	JJ.push_back(restShapeV.size() + 7*restShapeF.rows() - 1);
	SS.push_back(0);
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

Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 4> SymmetricDirichlet::ddJ_dXdX(int fi) {
	Eigen::Vector3d Dx = D1d.col(fi);
	Eigen::Vector3d Dy = D2d.col(fi);

	Eigen::Matrix<double, 1, 3> V0 = CurrV.row(restShapeF(fi, 0));
	Eigen::Matrix<double, 1, 3> V1 = CurrV.row(restShapeF(fi, 1));
	Eigen::Matrix<double, 1, 3> V2 = CurrV.row(restShapeF(fi, 2));

	Eigen::Matrix<double, 3, 9> dV0_dX, dV1_dX, dV2_dX;
	dV0_dX.setZero(); dV0_dX(0, 0) = 1; dV0_dX(1, 3) = 1; dV0_dX(2, 6) = 1;
	dV1_dX.setZero(); dV1_dX(0, 1) = 1; dV1_dX(1, 4) = 1; dV1_dX(2, 7) = 1;
	dV2_dX.setZero(); dV2_dX(0, 2) = 1; dV2_dX(1, 5) = 1; dV2_dX(2, 8) = 1;

	Eigen::Matrix<double, 3, 9> db1_dX = dB1_dX(fi);
	Eigen::Matrix<double, 3, 9> db2_dX = dB2_dX(fi);
	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3>
		XX, YY,
		ddb1_dXdX = ddB1_dXdX(fi),
		ddb2_dXdX = ddB2_dXdX(fi);

	XX[0] =
		V0[0] * ddb1_dXdX[0] +
		V0[1] * ddb1_dXdX[1] +
		V0[2] * ddb1_dXdX[2] +
		dV0_dX.transpose()*db1_dX +
		db1_dX.transpose()*dV0_dX;

	XX[1] =
		V1[0] * ddb1_dXdX[0] +
		V1[1] * ddb1_dXdX[1] +
		V1[2] * ddb1_dXdX[2] +
		dV1_dX.transpose()*db1_dX +
		db1_dX.transpose()*dV1_dX;

	XX[2] =
		V2[0] * ddb1_dXdX[0] +
		V2[1] * ddb1_dXdX[1] +
		V2[2] * ddb1_dXdX[2] +
		dV2_dX.transpose()*db1_dX +
		db1_dX.transpose()*dV2_dX;


	YY[0] =
		V0[0] * ddb2_dXdX[0] +
		V0[1] * ddb2_dXdX[1] +
		V0[2] * ddb2_dXdX[2] +
		dV0_dX.transpose()*db2_dX +
		db2_dX.transpose()*dV0_dX;

	YY[1] =
		V1[0] * ddb2_dXdX[0] +
		V1[1] * ddb2_dXdX[1] +
		V1[2] * ddb2_dXdX[2] +
		dV1_dX.transpose()*db2_dX +
		db2_dX.transpose()*dV1_dX;

	YY[2] =
		V2[0] * ddb2_dXdX[0] +
		V2[1] * ddb2_dXdX[1] +
		V2[2] * ddb2_dXdX[2] +
		dV2_dX.transpose()*db2_dX +
		db2_dX.transpose()*dV2_dX;

	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 4> H;
	H[0] = Dx[0] * XX[0] + Dx[1] * XX[1] + Dx[2] * XX[2];
	H[1] = Dx[0] * YY[0] + Dx[1] * YY[1] + Dx[2] * YY[2];
	H[2] = Dy[0] * XX[0] + Dy[1] * XX[1] + Dy[2] * XX[2];
	H[3] = Dy[0] * YY[0] + Dy[1] * YY[1] + Dy[2] * YY[2];

	return H;
}

Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> SymmetricDirichlet::ddB1_dXdX(int fi) {
	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> H;
	Eigen::Matrix<double, 9, 9> H_x, H_y, H_z;

	Eigen::Matrix<double, 3, 1> V0 = CurrV.row(restShapeF(fi, 0));
	Eigen::Matrix<double, 3, 1> V1 = CurrV.row(restShapeF(fi, 1));
	Eigen::Matrix<double, 3, 1> V2 = CurrV.row(restShapeF(fi, 2));
	double NormB1 = (V1 - V0).norm();
	double Qx = V1[0] - V0[0]; // x1 - x0
	double Qy = V1[1] - V0[1]; // y1 - y0
	double Qz = V1[2] - V0[2]; // z1 - z0

	double xxx = (-3 * Qx*(pow(Qy, 2) + pow(Qz, 2))) / pow(NormB1, 5);
	double yyy = (-3 * Qy*(pow(Qx, 2) + pow(Qz, 2))) / pow(NormB1, 5);
	double zzz = (-3 * Qz*(pow(Qx, 2) + pow(Qy, 2))) / pow(NormB1, 5);

	double xxy = (Qy*(pow(NormB1, 2) - 3 * pow(Qx, 2))) / pow(NormB1, 5);
	double xxz = (Qz*(pow(NormB1, 2) - 3 * pow(Qx, 2))) / pow(NormB1, 5);
	double xyy = (Qx*(pow(NormB1, 2) - 3 * pow(Qy, 2))) / pow(NormB1, 5);
	double xzz = (Qx*(pow(NormB1, 2) - 3 * pow(Qz, 2))) / pow(NormB1, 5);
	double yyz = (Qz*(pow(NormB1, 2) - 3 * pow(Qy, 2))) / pow(NormB1, 5);
	double yzz = (Qy*(pow(NormB1, 2) - 3 * pow(Qz, 2))) / pow(NormB1, 5);

	double xyz = (3 * Qx*Qy*Qz) / pow(NormB1, 5);

	H_x <<
		xxx, -xxx, 0, -xxy, xxy, 0, -xxz, xxz, 0,
		-xxx, xxx, 0, xxy, -xxy, 0, xxz, -xxz, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		-xxy, xxy, 0, -xyy, xyy, 0, xyz, -xyz, 0,
		xxy, -xxy, 0, xyy, -xyy, 0, -xyz, xyz, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		-xxz, xxz, 0, xyz, -xyz, 0, -xzz, xzz, 0,
		xxz, -xxz, 0, -xyz, xyz, 0, xzz, -xzz, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0;

	H_y <<
		-xxy, xxy, 0, -xyy, xyy, 0, xyz, -xyz, 0,
		xxy, -xxy, 0, xyy, -xyy, 0, -xyz, xyz, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		-xyy, xyy, 0, yyy, -yyy, 0, -yyz, yyz, 0,
		xyy, -xyy, 0, -yyy, yyy, 0, yyz, -yyz, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		xyz, -xyz, 0, -yyz, yyz, 0, -yzz, yzz, 0,
		-xyz, xyz, 0, yyz, -yyz, 0, yzz, -yzz, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0;

	H_z <<
		-xxz, xxz, 0, xyz, -xyz, 0, -xzz, xzz, 0,
		xxz, -xxz, 0, -xyz, xyz, 0, xzz, -xzz, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		xyz, -xyz, 0, -yyz, yyz, 0, -yzz, yzz, 0,
		-xyz, xyz, 0, yyz, -yyz, 0, yzz, -yzz, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		-xzz, xzz, 0, -yzz, yzz, 0, zzz, -zzz, 0,
		xzz, -xzz, 0, yzz, -yzz, 0, -zzz, zzz, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0;

	H[0] = H_x;
	H[1] = H_y;
	H[2] = H_z;
	return H;
}

Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> SymmetricDirichlet::ddB2_dXdX(int fi) {
	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> H;

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
	double NormB2_3 = pow(NormB2, 3);
	double NormB2_6 = pow(NormB2, 6);


	Eigen::Matrix<double, 3, 6> dxyz;
	dxyz.row(0) <<
		-Qy * Wy - Qz * Wz,
		-Qx * Wy + 2 * Qy * Wx,
		2 * Qz*Wx - Qx * Wz,
		pow(Qy, 2) + pow(Qz, 2),
		-Qy * Qx,
		-Qx * Qz;

	Eigen::Matrix<Eigen::Matrix<double, 6, 6>, 1, 3> ddxyz;
	ddxyz[0] <<
		0, -Wy, -Wz, 0, -Qy, -Qz,
		-Wy, 2 * Wx, 0, 2 * Qy, -Qx, 0,
		-Wz, 0, 2 * Wx, 2 * Qz, 0, -Qx,
		0, 2 * Qy, 2 * Qz, 0, 0, 0,
		-Qy, -Qx, 0, 0, 0, 0,
		-Qz, 0, -Qx, 0, 0, 0;

	dxyz.row(1) <<
		2 * Qx*Wy - Qy * Wx,
		-Qz * Wz - Wx * Qx,
		-Qy * Wz + 2 * Qz*Wy,
		-Qx * Qy,
		pow(Qz, 2) + pow(Qx, 2),
		-Qz * Qy;

	ddxyz[1] <<
		2 * Wy, -Wx, 0, -Qy, 2 * Qx, 0,
		-Wx, 0, -Wz, -Qx, 0, -Qz,
		0, -Wz, 2 * Wy, 0, 2 * Qz, -Qy,
		-Qy, -Qx, 0, 0, 0, 0,
		2 * Qx, 0, 2 * Qz, 0, 0, 0,
		0, -Qz, -Qy, 0, 0, 0;

	dxyz.row(2) <<
		-Qz * Wx + 2 * Qx*Wz,
		2 * Qy*Wz - Qz * Wy,
		-Qx * Wx - Qy * Wy,
		-Qx * Qz,
		-Qz * Qy,
		pow(Qx, 2) + pow(Qy, 2);

	ddxyz[2] <<
		2 * Wz, 0, -Wx, -Qz, 0, 2 * Qx,
		0, 2 * Wz, -Wy, 0, -Qz, 2 * Qy,
		-Wx, -Wy, 0, -Qx, -Qy, 0,
		-Qz, 0, -Qx, 0, 0, 0,
		0, -Qz, -Qy, 0, 0, 0,
		2 * Qx, 2 * Qy, 0, 0, 0, 0;

	Eigen::Matrix<double, 6, 1> dnorm;
	dnorm <<
		(b2[0] * dxyz(0, 0) + b2[1] * dxyz(1, 0) + b2[2] * dxyz(2, 0)) / NormB2,
		(b2[0] * dxyz(0, 1) + b2[1] * dxyz(1, 1) + b2[2] * dxyz(2, 1)) / NormB2,
		(b2[0] * dxyz(0, 2) + b2[1] * dxyz(1, 2) + b2[2] * dxyz(2, 2)) / NormB2,
		(b2[0] * dxyz(0, 3) + b2[1] * dxyz(1, 3) + b2[2] * dxyz(2, 3)) / NormB2,
		(b2[0] * dxyz(0, 4) + b2[1] * dxyz(1, 4) + b2[2] * dxyz(2, 4)) / NormB2,
		(b2[0] * dxyz(0, 5) + b2[1] * dxyz(1, 5) + b2[2] * dxyz(2, 5)) / NormB2;


	////////gradient
	////X-coordinate of B2
	//double dB2xdQx = (dxdQx*NormB2 - b2[0] * dnormdQx) / NormB2_2;
	//double dB2xdQy = (dxdQy*NormB2 - b2[0] * dnormdQy) / NormB2_2;
	//double dB2xdQz = (dxdQz*NormB2 - b2[0] * dnormdQz) / NormB2_2;
	//double dB2xdWx = (dxdWx*NormB2 - b2[0] * dnormdWx) / NormB2_2;
	//double dB2xdWy = (dxdWy*NormB2 - b2[0] * dnormdWy) / NormB2_2;
	//double dB2xdWz = (dxdWz*NormB2 - b2[0] * dnormdWz) / NormB2_2;
	////Y-coordinate of B2
	//double dB2ydQx = (dydQx*NormB2 - b2[1] * dnormdQx) / NormB2_2;
	//double dB2ydQy = (dydQy*NormB2 - b2[1] * dnormdQy) / NormB2_2;
	//double dB2ydQz = (dydQz*NormB2 - b2[1] * dnormdQz) / NormB2_2;
	//double dB2ydWx = (dydWx*NormB2 - b2[1] * dnormdWx) / NormB2_2;
	//double dB2ydWy = (dydWy*NormB2 - b2[1] * dnormdWy) / NormB2_2;
	//double dB2ydWz = (dydWz*NormB2 - b2[1] * dnormdWz) / NormB2_2;
	////Z-coordinate of B2
	//double dB2zdQx = (dzdQx*NormB2 - b2[2] * dnormdQx) / NormB2_2;
	//double dB2zdQy = (dzdQy*NormB2 - b2[2] * dnormdQy) / NormB2_2;
	//double dB2zdQz = (dzdQz*NormB2 - b2[2] * dnormdQz) / NormB2_2;
	//double dB2zdWx = (dzdWx*NormB2 - b2[2] * dnormdWx) / NormB2_2;
	//double dB2zdWy = (dzdWy*NormB2 - b2[2] * dnormdWy) / NormB2_2;
	//double dB2zdWz = (dzdWz*NormB2 - b2[2] * dnormdWz) / NormB2_2;


	//hessian
	auto Hess = [&](int v, int d1, int d2) {
		return
			((ddxyz[v](d1, d2)*NormB2 - dnorm[d2] * dxyz(v, d1)) / NormB2_2)
			-
			(
			(
				NormB2_3 *
				(
					dxyz(v, d2) *dnorm[d1] * NormB2 +
					b2[v] * (
						dxyz(0, d2) * dxyz(0, d1) +
						dxyz(1, d2) * dxyz(1, d1) +
						dxyz(2, d2) * dxyz(2, d1) +
						ddxyz[0](d1, d2) * b2[0] +
						ddxyz[1](d1, d2) * b2[1] +
						ddxyz[2](d1, d2) * b2[2]
						)
					)
				- 3 * NormB2_2*dnorm[d2] * b2[v] * dnorm[d1] * NormB2
				) / NormB2_6
				);
	};

	for (int xyz = 0; xyz < 3; xyz++) {
		H[xyz] <<
			0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, Hess(xyz, 0, 0), Hess(xyz, 0, 3), 0, Hess(xyz, 0, 1), Hess(xyz, 0, 4), 0, Hess(xyz, 0, 2), Hess(xyz, 0, 5),
			0, 0, Hess(xyz, 3, 3), 0, Hess(xyz, 3, 1), Hess(xyz, 3, 4), 0, Hess(xyz, 3, 2), Hess(xyz, 3, 5),
			0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, Hess(xyz, 1, 1), Hess(xyz, 1, 4), 0, Hess(xyz, 1, 2), Hess(xyz, 1, 5),
			0, 0, 0, 0, 0, Hess(xyz, 4, 4), 0, Hess(xyz, 4, 2), Hess(xyz, 4, 5),
			0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, Hess(xyz, 2, 2), Hess(xyz, 2, 5),
			0, 0, 0, 0, 0, 0, 0, 0, Hess(xyz, 5, 5);

		H[xyz] = H[xyz].selfadjointView<Eigen::Upper>();
		for (int r = 0; r < 9; r += 3)
			H[xyz].row(r) = -H[xyz].row(r + 1) - H[xyz].row(r + 2);
		for (int c = 0; c < 9; c += 3)
			H[xyz].col(c) = -H[xyz].col(c + 1) - H[xyz].col(c + 2);
	}
	return H;
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
