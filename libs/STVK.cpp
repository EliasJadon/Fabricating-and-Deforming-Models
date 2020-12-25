#include "STVK.h"

STVK::STVK(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F) 
{
	init_mesh(V, F);
	name = "STVK";
	w = 0;
	shearModulus = 0.3;
	bulkModulus = 1.5;
	Cuda::AllocateMemory(grad, 3 * V.rows() + 7 * F.rows());
	setRestShapeFromCurrentConfiguration();
	std::cout << "\t" << name << " constructor" << std::endl;
}

STVK::~STVK() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void STVK::setRestShapeFromCurrentConfiguration() {
	dXInv.clear();
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
		dXInv.push_back(dX.inverse()); //TODO .inverse() is baaad
	}
	//compute the area for each triangle
	igl::doublearea(restShapeV, restShapeF, restShapeArea);
	restShapeArea /= 2;
}

void STVK::updateX(Cuda::Array<double>& curr_x)
{
	Cuda::MemCpyDeviceToHost(curr_x);
	CurrV.resize(restShapeV.rows(), 3);
	for (int vi = 0; vi < restShapeV.rows(); vi++) {
		CurrV(vi, 0) = curr_x.host_arr[vi];
		CurrV(vi, 1) = curr_x.host_arr[vi + restShapeV.rows()];
		CurrV(vi, 2) = curr_x.host_arr[vi + 2 * restShapeV.rows()];
	}
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

double STVK::value(Cuda::Array<double>& curr_x, const bool update) {
	updateX(curr_x);
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
	updateX(X);
	for (int i = 0; i < grad.size; i++) {
		grad.host_arr[i] = 0;
	}

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
				grad.host_arr[restShapeF(fi, vi) + (xyz*restShapeV.rows())] += dE_dX[xyz*3 + vi];
	}
	Cuda::MemCpyHostToDevice(grad);
	return &grad;
	//if (update)
	//	gradient_norm = g.norm();
}
