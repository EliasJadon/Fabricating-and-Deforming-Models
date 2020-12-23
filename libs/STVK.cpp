#include "STVK.h"

STVK::STVK(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) 
{
	init_mesh(V, F);
	name = "STVK";
	w = 0;
	if (restShapeV.size() == 0 || restShapeF.size() == 0)
		throw name + " must define members V,F before init()!";

	assert(restShapeV.col(2).isZero() && "Warning: Rest shape is assumed to be in the plane (z coordinate must be zero in the beginning)");
	shearModulus = 0.3;
	bulkModulus = 1.5;
	setRestShapeFromCurrentConfiguration();
	std::cout << "\t" << name << " constructor" << std::endl;
}

STVK::~STVK() {
	std::cout << "\t" << name << " destructor" << std::endl;
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

//void STVK::updateX(Cuda::Array<double>& curr_x)
//{
	//assert(X.rows() == (restShapeV.size()+ 7*restShapeF.rows()));
	//CurrV = Eigen::Map<const Eigen::MatrixX3d>(X.middleRows(0, restShapeV.size()).data(), restShapeV.rows(), 3);
	//
	//F.clear();
	//strain.clear();
	//for (int fi = 0; fi < restShapeF.rows(); fi++) {
	//	Eigen::VectorXd v1 = CurrV.row(restShapeF(fi, 1)) - CurrV.row(restShapeF(fi, 0));
	//	Eigen::VectorXd v2 = CurrV.row(restShapeF(fi, 2)) - CurrV.row(restShapeF(fi, 0));
	//	Eigen::Matrix<double, 3, 2> dx;
	//	dx <<
	//		v1(0), v2(0),
	//		v1(1), v2(1),
	//		v1(2), v2(2);
	//	F.push_back(dx * dXInv[fi]);

	//	//compute the Green Strain = 1/2 * (F'F-I)
	//	strain.push_back(F[fi].transpose() * F[fi]);
	//	strain[fi](0, 0) -= 1; strain[fi](1, 1) -= 1;
	//	strain[fi] *= 0.5;
	//}
//}

double STVK::value(Cuda::Array<double>& curr_x, const bool update) {
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

Cuda::Array<double>* STVK::gradient(Cuda::Array<double>& X, const bool update)
{
	return NULL;
	/*g.conservativeResize(restShapeV.size() + 7*restShapeF.rows());
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
		gradient_norm = g.norm();*/
}
