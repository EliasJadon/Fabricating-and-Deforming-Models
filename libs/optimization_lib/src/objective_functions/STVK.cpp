#include "objective_functions/STVK.h"

STVK::STVK() {
	name = "STVK";
	w = 0;
	std::cout << name << " constructor" << std::endl;
}

STVK::~STVK() {
	std::cout << name << " destructor" << std::endl;
}

void STVK::init()
{
	std::cout << name << " initialization" << std::endl;
	if (restShapeV.size() == 0 || restShapeF.size() == 0)
		throw name + " must define members V,F before init()!";
	
	assert(restShapeV.col(2).isZero() && "Warning: Rest shape is assumed to be in the plane (z coordinate must be zero in the beginning)");
	shearModulus = 0.3;
	bulkModulus = 1.5;
	setRestShapeFromCurrentConfiguration();
	init_hessian();
}

void STVK::setRestShapeFromCurrentConfiguration() {
	dXInv.clear();
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		Eigen::VectorXd V1 = restShapeV.row(restShapeF(fi, 1)) - restShapeV.row(restShapeF(fi, 0));
		Eigen::VectorXd V2 = restShapeV.row(restShapeF(fi, 2)) - restShapeV.row(restShapeF(fi, 0));
		//matrix that holds three edge vectors
		Eigen::Matrix2d dX;
		dX <<
			V1[0], V2[0],
			V1[1], V2[1];
		dXInv.push_back(dX.inverse()); //TODO .inverse() is baaad
	}
	//compute the area for each triangle
	igl::doublearea(restShapeV, restShapeF, restShapeArea);
	restShapeArea /= 2;
}

void STVK::updateX(const Eigen::VectorXd& X)
{
	assert(X.rows() == (3 * restShapeV.rows()));
	CurrV = Eigen::Map<const Eigen::MatrixX3d>(X.data(), X.rows() / 3, 3);
	
	F.clear();
	strain.clear();
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		Eigen::VectorXd v1 = CurrV.row(restShapeF(fi, 1)) - CurrV.row(restShapeF(fi, 0));
		Eigen::VectorXd v2 = CurrV.row(restShapeF(fi, 2)) - CurrV.row(restShapeF(fi, 0));
		Eigen::Matrix<double, 3, 2> dx;
		dx <<
			v1(0), v2(0),
			v1(1), v2(1),
			v1(2), v2(2);
		F.push_back(dx * dXInv[fi]);

		//compute the Green Strain = 1/2 * (F'F-I)
		strain.push_back(F[fi].transpose() * F[fi]);
		strain[fi](0, 0) -= 1; strain[fi](1, 1) -= 1;
		strain[fi] *= 0.5;
	}
}

double STVK::value(const bool update) {
	Eigen::VectorXd Energy(restShapeF.rows());
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
			Energy(fi) = 
				shearModulus * strain[fi].squaredNorm() +
				(bulkModulus / 2) * pow(strain[fi].trace(), 2);
	}
	double total_energy = restShapeArea.transpose() * Energy;
	
	if (update) {
		Efi = Energy;
		energy_value = total_energy;
	}
	return total_energy;
}

void STVK::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(restShapeV.rows() * 3);
	g.setZero();

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		Eigen::Matrix<double, 6, 9> dF_dX;
		Eigen::Matrix<double, 4, 6> dstrain_dF;
		Eigen::Matrix<double, 1, 4> dE_dJ;
		dF_dX <<
			-dXInv[fi](0, 0) - dXInv[fi](1, 0), dXInv[fi](0, 0), dXInv[fi](1, 0), 0, 0, 0, 0, 0, 0,
			-dXInv[fi](0, 1) - dXInv[fi](1, 1), dXInv[fi](0, 1), dXInv[fi](1, 1), 0, 0, 0, 0, 0, 0,
			0, 0, 0, -dXInv[fi](0, 0) - dXInv[fi](1, 0), dXInv[fi](0, 0), dXInv[fi](1, 0), 0, 0, 0,
			0, 0, 0, -dXInv[fi](0, 1) - dXInv[fi](1, 1), dXInv[fi](0, 1), dXInv[fi](1, 1), 0, 0, 0,
			0, 0, 0, 0, 0, 0, -dXInv[fi](0, 0) - dXInv[fi](1, 0), dXInv[fi](0, 0), dXInv[fi](1, 0),
			0, 0, 0, 0, 0, 0, -dXInv[fi](0, 1) - dXInv[fi](1, 1), dXInv[fi](0, 1), dXInv[fi](1, 1);

		dstrain_dF <<
			F[fi](0, 0), 0, F[fi](1, 0), 0, F[fi](2, 0), 0,
			0.5*F[fi](0, 1), 0.5*F[fi](0, 0), 0.5*F[fi](1, 1), 0.5*F[fi](1, 0), 0.5*F[fi](2, 1), 0.5*F[fi](2, 0),
			0.5*F[fi](0, 1), 0.5*F[fi](0, 0), 0.5*F[fi](1, 1), 0.5*F[fi](1, 0), 0.5*F[fi](2, 1), 0.5*F[fi](2, 0),
			0, F[fi](0, 1), 0, F[fi](1, 1), 0, F[fi](2, 1);
		
		dE_dJ <<
			2 * shearModulus*strain[fi](0, 0) + bulkModulus * strain[fi].trace(),
			2 * shearModulus*strain[fi](0, 1),
			2 * shearModulus*strain[fi](1, 0),
			2 * shearModulus*strain[fi](1, 1) + bulkModulus * strain[fi].trace();
		dE_dJ *= restShapeArea[fi];
	
		Eigen::Matrix<double, 1, 9> dE_dX = dE_dJ * dstrain_dF * dF_dX;
		
		for (int vi = 0; vi < 3; vi++)
			for (int xyz = 0; xyz < 3; xyz++)
				g[restShapeF(fi, vi) + (xyz*restShapeV.rows())] += dE_dX[xyz*3 + vi];
	}

	if (update)
		gradient_norm = g.norm();
}

void STVK::init_hessian() {

}

void STVK::hessian() {
	II.clear();
	JJ.clear();
	SS.clear();
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		Eigen::Matrix<double, 1, 4> dE_dJ;
		Eigen::Matrix<double, 4, 4> dE_dJdJ;
		Eigen::Matrix<double, 9, 9> dE_dXdX;

		dE_dJ <<
			2 * shearModulus*strain[fi](0, 0) + bulkModulus * strain[fi].trace(),
			2 * shearModulus*strain[fi](0, 1),
			2 * shearModulus*strain[fi](1, 0),
			2 * shearModulus*strain[fi](1, 1) + bulkModulus * strain[fi].trace();
		dE_dJ *= restShapeArea[fi];

		dE_dJdJ <<
			2 * shearModulus + bulkModulus, 0, 0, bulkModulus,
			0, 2 * shearModulus, 0, 0,
			0, 0, 2 * shearModulus, 0,
			bulkModulus, 0, 0, 2 * shearModulus + bulkModulus;
		dE_dJdJ *= restShapeArea[fi];
		
		Eigen::Matrix<double, 6, 9> dF_dX;
		dF_dX <<
			-dXInv[fi](0, 0) - dXInv[fi](1, 0), dXInv[fi](0, 0), dXInv[fi](1, 0), 0, 0, 0, 0, 0, 0,
			-dXInv[fi](0, 1) - dXInv[fi](1, 1), dXInv[fi](0, 1), dXInv[fi](1, 1), 0, 0, 0, 0, 0, 0,
			0, 0, 0, -dXInv[fi](0, 0) - dXInv[fi](1, 0), dXInv[fi](0, 0), dXInv[fi](1, 0), 0, 0, 0,
			0, 0, 0, -dXInv[fi](0, 1) - dXInv[fi](1, 1), dXInv[fi](0, 1), dXInv[fi](1, 1), 0, 0, 0,
			0, 0, 0, 0, 0, 0, -dXInv[fi](0, 0) - dXInv[fi](1, 0), dXInv[fi](0, 0), dXInv[fi](1, 0),
			0, 0, 0, 0, 0, 0, -dXInv[fi](0, 1) - dXInv[fi](1, 1), dXInv[fi](0, 1), dXInv[fi](1, 1);

		Eigen::Matrix<double, 4, 6> dstrain_dF;
		dstrain_dF <<
			F[fi](0, 0), 0, F[fi](1, 0), 0, F[fi](2, 0), 0,
			0.5*F[fi](0, 1), 0.5*F[fi](0, 0), 0.5*F[fi](1, 1), 0.5*F[fi](1, 0), 0.5*F[fi](2, 1), 0.5*F[fi](2, 0),
			0.5*F[fi](0, 1), 0.5*F[fi](0, 0), 0.5*F[fi](1, 1), 0.5*F[fi](1, 0), 0.5*F[fi](2, 1), 0.5*F[fi](2, 0),
			0, F[fi](0, 1), 0, F[fi](1, 1), 0, F[fi](2, 1);

		Eigen::Matrix<double, 6, 6> ds_dFdF___dE_ds; // ds_dFdF * dE_ds
		ds_dFdF___dE_ds <<
			dE_dJ[0], 0.5*dE_dJ[1] + 0.5*dE_dJ[2], 0, 0, 0, 0,
			0.5*dE_dJ[1] + 0.5*dE_dJ[2], dE_dJ[3], 0, 0, 0, 0,
			0, 0, dE_dJ[0], 0.5*dE_dJ[1] + 0.5*dE_dJ[2], 0, 0,
			0, 0, 0.5*dE_dJ[1] + 0.5*dE_dJ[2], dE_dJ[3], 0, 0,
			0, 0, 0, 0, dE_dJ[0], 0.5*dE_dJ[1] + 0.5*dE_dJ[2],
			0, 0, 0, 0, 0.5*dE_dJ[1] + 0.5*dE_dJ[2], dE_dJ[3];

		dE_dXdX =
			dF_dX.transpose() * dstrain_dF.transpose() * dE_dJdJ * dstrain_dF * dF_dX
			+ dF_dX.transpose() * ds_dFdF___dE_ds * dF_dX;
		
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
}
