﻿#include "BendingNormal.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <igl/triangle_triangle_adjacency.h>

BendingNormal::BendingNormal(FunctionType type) {
	functionType = type;
	name = "Bending Normal";
	w = 0;
	std::cout << "\t" << name << " constructor" << std::endl;
}

BendingNormal::~BendingNormal() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void BendingNormal::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if (restShapeV.size() == 0 || restShapeF.size() == 0)
		throw name + " must define members V,F before init()!";

	calculateHinges();
	
	restArea.resize(num_hinges);
	Eigen::VectorXd facesArea;
	igl::doublearea(restShapeV, restShapeF, facesArea);
	facesArea /= 2;
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		restArea(hi) = (facesArea(f0) + facesArea(f1)) / 2;
	}
	
	d_normals.resize(num_hinges);
	planarParameter = 1;
}

void BendingNormal::updateX(Cuda::Array<double>& curr_x)
{
	/*assert(X.rows() == (restShapeV.size() + 7*restShapeF.rows()));
	CurrV = Eigen::Map<const Eigen::MatrixX3d>(X.middleRows(0, restShapeV.size()).data(), restShapeV.rows(), 3);
	igl::per_face_normals(CurrV, restShapeF, normals);
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		d_normals(hi) = (normals.row(f1) - normals.row(f0)).squaredNorm();
	}*/
}

void BendingNormal::calculateHinges() {
	std::vector<std::vector<std::vector<int>>> TT;
	igl::triangle_triangle_adjacency(restShapeF, TT);
	assert(TT.size() == restShapeF.rows());
	hinges_faceIndex.clear();

	///////////////////////////////////////////////////////////
	//Part 1 - Find unique hinges
	for (int fi = 0; fi < TT.size(); fi++) {
		std::vector< std::vector<int>> CurrFace = TT[fi];
		assert(CurrFace.size() == 3 && "Each face should be a triangle (not square for example)!");
		for (std::vector<int> hinge : CurrFace) {
			if (hinge.size() == 1) {
				//add this "hinge"
				int FaceIndex1 = fi;
				int FaceIndex2 = hinge[0];

				if (FaceIndex2 < FaceIndex1) {
					//Skip
					//This hinge already exists!
					//Empty on purpose
				}
				else {
					hinges_faceIndex.push_back(Eigen::Vector2d(FaceIndex1, FaceIndex2));
				}
			}
			else if (hinge.size() == 0) {
				//Skip
				//This triangle has no another adjacent triangle on that edge
				//Empty on purpose
			}
			else {
				//We shouldn't get here!
				//The mesh is invalid
				assert("Each triangle should have only one adjacent triangle on each edge!");
			}

		}
	}
	num_hinges = hinges_faceIndex.size();

	///////////////////////////////////////////////////////////
	//Part 2 - Find x0,x1,x2,x3 indecis for each hinge
	x0_GlobInd.resize(num_hinges);
	x1_GlobInd.resize(num_hinges);
	x2_GlobInd.resize(num_hinges);
	x3_GlobInd.resize(num_hinges);
	x0_LocInd.resize(num_hinges, 2); x0_LocInd.setConstant(-1);
	x1_LocInd.resize(num_hinges, 2); x1_LocInd.setConstant(-1);
	x2_LocInd.resize(num_hinges, 2); x2_LocInd.setConstant(-1);
	x3_LocInd.resize(num_hinges, 2); x3_LocInd.setConstant(-1);

	for (int hi = 0; hi < num_hinges; hi++) {
		//first triangle vertices
		int v1 = restShapeF(hinges_faceIndex[hi](0), 0);
		int v2 = restShapeF(hinges_faceIndex[hi](0), 1);
		int v3 = restShapeF(hinges_faceIndex[hi](0), 2);
		//second triangle vertices
		int V1 = restShapeF(hinges_faceIndex[hi](1), 0);
		int V2 = restShapeF(hinges_faceIndex[hi](1), 1);
		int V3 = restShapeF(hinges_faceIndex[hi](1), 2);

		/*
		* Here we should find x0,x1,x2,x3
		* from the given two triangles (v1,v2,v3),(V1,V2,V3)
		*
		*	x0--x2
		*  / \ /
		* x3--x1
		*
		*/
		if (v1 != V1 && v1 != V2 && v1 != V3) {
			x2_GlobInd(hi) = v1; x2_LocInd(hi, 0) = 0;
			x0_GlobInd(hi) = v2; x0_LocInd(hi, 0) = 1;
			x1_GlobInd(hi) = v3; x1_LocInd(hi, 0) = 2;

			if (V1 != v1 && V1 != v2 && V1 != v3) {
				x3_GlobInd(hi) = V1; x3_LocInd(hi, 1) = 0;
			}
			else if (V2 != v1 && V2 != v2 && V2 != v3) {
				x3_GlobInd(hi) = V2; x3_LocInd(hi, 1) = 1;
			}
			else {
				x3_GlobInd(hi) = V3; x3_LocInd(hi, 1) = 2;
			}
		}
		else if (v2 != V1 && v2 != V2 && v2 != V3) {
			x2_GlobInd(hi) = v2; x2_LocInd(hi, 0) = 1;
			x0_GlobInd(hi) = v1; x0_LocInd(hi, 0) = 0;
			x1_GlobInd(hi) = v3; x1_LocInd(hi, 0) = 2;

			if (V1 != v1 && V1 != v2 && V1 != v3) {
				x3_GlobInd(hi) = V1; x3_LocInd(hi, 1) = 0;
			}	
			else if (V2 != v1 && V2 != v2 && V2 != v3) {
				x3_GlobInd(hi) = V2; x3_LocInd(hi, 1) = 1;
			}
			else {
				x3_GlobInd(hi) = V3; x3_LocInd(hi, 1) = 2;
			}
		}
		else {
			x2_GlobInd(hi) = v3; x2_LocInd(hi, 0) = 2;
			x0_GlobInd(hi) = v1; x0_LocInd(hi, 0) = 0;
			x1_GlobInd(hi) = v2; x1_LocInd(hi, 0) = 1;

			if (V1 != v1 && V1 != v2 && V1 != v3) {
				x3_GlobInd(hi) = V1; x3_LocInd(hi, 1) = 0;
			}
			else if (V2 != v1 && V2 != v2 && V2 != v3) {
				x3_GlobInd(hi) = V2; x3_LocInd(hi, 1) = 1;
			}
			else {
				x3_GlobInd(hi) = V3; x3_LocInd(hi, 1) = 2;
			}
		}

		if (V1 == x0_GlobInd(hi))
			x0_LocInd(hi, 1) = 0;
		else if (V2 == x0_GlobInd(hi))
			x0_LocInd(hi, 1) = 1;
		else if (V3 == x0_GlobInd(hi))
			x0_LocInd(hi, 1) = 2;

		if (V1 == x1_GlobInd(hi))
			x1_LocInd(hi, 1) = 0;
		else if (V2 == x1_GlobInd(hi))
			x1_LocInd(hi, 1) = 1;
		else if (V3 == x1_GlobInd(hi))
			x1_LocInd(hi, 1) = 2;
	}

	/*for (int hi = 0; hi < num_hinges; hi++) {
		std::cout << "--------------hinge " << hi << ":\n";
		std::cout << "x0 = " << x0_GlobInd(hi) << ", (" << x0_LocInd(hi, 0) << "," << x0_LocInd(hi, 1) << ")\n";
		std::cout << "x1 = " << x1_GlobInd(hi) << ", (" << x1_LocInd(hi, 0) << "," << x1_LocInd(hi, 1) << ")\n";
		std::cout << "x2 = " << x2_GlobInd(hi) << ", (" << x2_LocInd(hi, 0) << "," << x2_LocInd(hi, 1) << ")\n";
		std::cout << "x3 = " << x3_GlobInd(hi) << ", (" << x3_LocInd(hi, 0) << "," << x3_LocInd(hi, 1) << ")\n";
	}
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		std::cout << "--------------face " << fi << ":\n";
		std::cout << restShapeF.row(fi) << "\n";
	}*/
}

Eigen::VectorXd BendingNormal::Phi(Eigen::VectorXd x) {
	if(functionType == FunctionType::QUADRATIC)
		return x.cwiseAbs2();
	else if (functionType == FunctionType::EXPONENTIAL) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = std::exp(x(i)*x(i));
		}
		return res;
	}
	else if (functionType == FunctionType::SIGMOID) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			double x2 = pow(x(i), 2);
			res(i) = x2/(x2+planarParameter);
		}
		return res;
	}
}

Eigen::VectorXd BendingNormal::dPhi_dm(Eigen::VectorXd x) {
	if (functionType == FunctionType::QUADRATIC)
		return 2 * x;
	else if (functionType == FunctionType::EXPONENTIAL) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = 2 * x(i) * std::exp(x(i)*x(i));
		}
		return res;
	}
	else if (functionType == FunctionType::SIGMOID) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = (2*x(i)*planarParameter) / 
				pow(x(i)*x(i) + planarParameter, 2);
		}
		return res;
	}
}

Eigen::VectorXd BendingNormal::d2Phi_dmdm(Eigen::VectorXd x) {
	if (functionType == FunctionType::QUADRATIC)
		return Eigen::VectorXd::Constant(x.rows(),2);
	else if (functionType == FunctionType::EXPONENTIAL) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = (4 * x(i)*x(i) + 2) * std::exp(x(i)*x(i));
		}
		return res;
	}
	else if (functionType == FunctionType::SIGMOID) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			double x2 = pow(x(i), 2);
			res(i) = (2* planarParameter)*(-3*x2+ planarParameter) / 
				pow(x2 + planarParameter,3);
		}
		return res;
	}
}

double BendingNormal::value(Cuda::Array<double>& curr_x, const bool update)
{
	Eigen::VectorXd Energy = Phi(d_normals);
	double value = Energy.transpose()*restArea;
	if (update) {
		//TODO: calculate Efi (for coloring the faces)
		Efi.setZero();
		energy_value = value;
	}
	return value;
}

Cuda::Array<double>* BendingNormal::gradient(Cuda::Array<double>& X, const bool update)
{
	return NULL;
}


Eigen::Matrix< double, 6, 1> BendingNormal::dm_dN(int hi) {
	// m = ||n1 - n0||^2
	// m = (n1.x - n0.x)^2 + (n1.y - n0.y)^2 + (n1.z - n0.z)^2
	int f0 = hinges_faceIndex[hi](0);
	int f1 = hinges_faceIndex[hi](1);
	Eigen::Matrix< double, 6, 1> grad;
	grad <<
		-2 * (normals(f1, 0) - normals(f0, 0)),	//n0.x
		-2 * (normals(f1, 1) - normals(f0, 1)), //n0.y
		-2 * (normals(f1, 2) - normals(f0, 2)), //n0.z
		2 *  (normals(f1, 0) - normals(f0, 0)),	//n1.x
		2 *  (normals(f1, 1) - normals(f0, 1)),	//n1.y
		2 *  (normals(f1, 2) - normals(f0, 2));	//n1.z
	return grad;
}

Eigen::Matrix< double, 6, 6> BendingNormal::d2m_dNdN(int hi) {
	Eigen::Matrix< double, 6, 6> hess;
	hess <<
		2, 0, 0, -2, 0, 0, //n0.x
		0, 2, 0, 0, -2, 0, //n0.y
		0, 0, 2, 0, 0, -2, //n0.z
		-2, 0, 0, 2, 0, 0, //n1.x
		0, -2, 0, 0, 2, 0, //n1.y
		0, 0, -2, 0, 0, 2; //n1.z
	return hess;
}

Eigen::Matrix<double, 6, 12> BendingNormal::dN_dx_perhinge(int hi) {
	int f0 = hinges_faceIndex[hi](0);
	int f1 = hinges_faceIndex[hi](1);
	Eigen::Matrix<double, 3, 9> n0_x = dN_dx_perface(f0);
	Eigen::Matrix<double, 3, 9> n1_x = dN_dx_perface(f1);
	Eigen::Matrix<double, 6, 12> n_x;
	n_x.setZero();

	n_x.block<3, 3>(0, 0) = n0_x.block<3, 3>(0, x0_LocInd(hi, 0) * 3);
	n_x.block<3, 3>(0, 3) = n0_x.block<3, 3>(0, x1_LocInd(hi, 0) * 3);
	n_x.block<3, 3>(0, 6) = n0_x.block<3, 3>(0, x2_LocInd(hi, 0) * 3);
	
	n_x.block<3, 3>(3, 0) = n1_x.block<3, 3>(0, x0_LocInd(hi, 1) * 3);
	n_x.block<3, 3>(3, 3) = n1_x.block<3, 3>(0, x1_LocInd(hi, 1) * 3);
	n_x.block<3, 3>(3, 9) = n1_x.block<3, 3>(0, x3_LocInd(hi, 1) * 3);

	return n_x;
}

Eigen::Matrix<double, 3, 9> BendingNormal::dN_dx_perface(int fi) {
	// e1 = v1-v0
	// e2 = v2-v0
	//
	// N = e1 x e2
	// N.x = (y1-y0)*(z2-z0)-(z1-z0)*(y2-y0)
	// N.y = (z1-z0)*(x2-x0)-(x1-x0)*(z2-z0)
	// N.z = (x1-x0)*(y2-y0)-(y1-y0)*(x2-x0)
	//
	// NormalizedN = N / norm

	Eigen::Vector3d e0 = CurrV.row(restShapeF(fi, 1)) - CurrV.row(restShapeF(fi, 0));
	Eigen::Vector3d e1 = CurrV.row(restShapeF(fi, 2)) - CurrV.row(restShapeF(fi, 0));
	Eigen::Vector3d N = e0.cross(e1);
	double norm = N.norm();
	Eigen::Matrix<double, 9, 3> jacobian_N;
	Eigen::Matrix<double, 9, 1> grad_norm;
	jacobian_N <<
		0, -e0(2) + e1(2), -e1(1) + e0(1),	//x0
		-e1(2) + e0(2), 0, -e0(0) + e1(0),	//y0
		-e0(1) + e1(1), -e1(0) + e0(0), 0,	//z0
		0, -e1(2), e1(1),	//x1
		e1(2), 0, -e1(0),	//y1
		-e1(1), e1(0), 0,	//z1
		0, e0(2), -e0(1),	//x2
		-e0(2), 0, e0(0),	//y2
		e0(1), -e0(0), 0;			//z2
	
	grad_norm = (N(0)*jacobian_N.col(0) + N(1)*jacobian_N.col(1) + N(2)*jacobian_N.col(2))/norm;
	
	Eigen::Matrix<double, 3, 9> jacobian_normalizedN;
	for (int i = 0; i < 3; i++)
		jacobian_normalizedN.row(i) = (jacobian_N.col(i) / norm) - ((grad_norm*N(i)) / pow(norm, 2));
	return jacobian_normalizedN;
}

Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> BendingNormal::d2N_dxdx_perface(int fi) {
	// e1 = v1-v0
	// e2 = v2-v0
	//
	// N = e1 x e2
	// N.x = (y1-y0)*(z2-z0)-(z1-z0)*(y2-y0)
	// N.y = (z1-z0)*(x2-x0)-(x1-x0)*(z2-z0)
	// N.z = (x1-x0)*(y2-y0)-(y1-y0)*(x2-x0)
	//
	// NormalizedN = N / norm

	Eigen::Vector3d e0 = CurrV.row(restShapeF(fi, 1)) - CurrV.row(restShapeF(fi, 0));
	Eigen::Vector3d e1 = CurrV.row(restShapeF(fi, 2)) - CurrV.row(restShapeF(fi, 0));
	Eigen::Vector3d N = e0.cross(e1);
	double norm = N.norm();
	Eigen::Matrix<double, 9, 3> jacobian_N;
	Eigen::Matrix<double, 9, 1> grad_norm;
	jacobian_N <<
		0, -e0(2) + e1(2), -e1(1) + e0(1),	//x0
		-e1(2) + e0(2), 0, -e0(0) + e1(0),	//y0
		-e0(1) + e1(1), -e1(0) + e0(0), 0,	//z0
		0, -e1(2), e1(1),	//x1
		e1(2), 0, -e1(0),	//y1
		-e1(1), e1(0), 0,	//z1
		0, e0(2), -e0(1),	//x2
		-e0(2), 0, e0(0),	//y2
		e0(1), -e0(0), 0;	//z2


	grad_norm = (N(0)*jacobian_N.col(0) + N(1)*jacobian_N.col(1) + N(2)*jacobian_N.col(2)) / norm;

	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> hess_N;
	Eigen::Matrix<double, 9, 9> hess_norm;
	//	0			//x0
	//	-z2 + z1	//y0
	//	-y1 + y2	//z0
	//	0			//x1
	//	z2 - z0		//y1
	//	-y2 + y0	//z1
	//	0			//x2
	//	-z1 + z0	//y2
	//	y1 - y0		//z2

	hess_N[0] <<
		// x0 y0 z0 x1 y1 z1 x2 y2 z2
		0, 0, 0, 0, 0, 0, 0, 0, 0,	//x0
		0, 0, 0, 0, 0, 1, 0, 0, -1, //y0
		0, 0, 0, 0, -1, 0, 0, 1, 0,	//z0
		0, 0, 0, 0, 0, 0, 0, 0, 0,	//x1
		0, 0, -1, 0, 0, 0, 0, 0, 1,	//y1
		0, 1, 0, 0, 0, 0, 0, -1, 0,	//z1
		0, 0, 0, 0, 0, 0, 0, 0, 0,	//x2
		0, 0, 1, 0, 0, -1, 0, 0, 0,	//y2
		0, -1, 0, 0, 1, 0, 0, 0, 0;	//z2


	//	-z1 + z2	//x0			
	//	0			//y0			
	//	-x2 + x1	//z0		
	//	-z2 + z0	//x1		
	//	0			//y1		
	//	x2 - x0		//z1		
	//	z1 - z0		//x2		
	//	0			//y2		
	//	-x1 + x0)	//z2		

	hess_N[1] <<
		// x0 y0 z0 x1 y1 z1 x2 y2 z2
		0, 0, 0, 0, 0, -1, 0, 0, 1,	//x0
		0, 0, 0, 0, 0, 0, 0, 0, 0,	//y0
		0, 0, 0, 1, 0, 0, -1, 0, 0,	//z0
		0, 0, 1, 0, 0, 0, 0, 0, -1,	//x1
		0, 0, 0, 0, 0, 0, 0, 0, 0,	//y1
		-1, 0, 0, 0, 0, 0, 1, 0, 0,	//z1
		0, 0, -1, 0, 0, 1, 0, 0, 0,	//x2
		0, 0, 0, 0, 0, 0, 0, 0, 0,	//y2
		1, 0, 0, -1, 0, 0, 0, 0, 0;	//z2

	//	-y2 + y1	//x0
	//	-x1  + x2	//y0
	//	0			//z0
	//	y2 - y0		//x1
	//	-x2 + x0	//y1
	//	0			//z1
	//	-y1 + y0	//x2
	//	x1 - x0		//y2
	//	0			//z2

	hess_N[2] <<
		// x0 y0 z0 x1 y1 z1 x2 y2 z2
		0, 0, 0, 0, 1, 0, 0, -1, 0,	//x0
		0, 0, 0, -1, 0, 0, 1, 0, 0,	//y0
		0, 0, 0, 0, 0, 0, 0, 0, 0,	//z0
		0, -1, 0, 0, 0, 0, 0, 1, 0,	//x1
		1, 0, 0, 0, 0, 0, -1, 0, 0,	//y1
		0, 0, 0, 0, 0, 0, 0, 0, 0,	//z1
		0, 1, 0, 0, -1, 0, 0, 0, 0,	//x2
		-1, 0, 0, 1, 0, 0, 0, 0, 0,	//y2
		0, 0, 0, 0, 0, 0, 0, 0, 0;	//z2

	hess_norm =
		((N(0)* hess_N[0] +
			N(1)* hess_N[1] +
			N(2)* hess_N[2] +
			jacobian_N.col(0)*jacobian_N.col(0).transpose() +
			jacobian_N.col(1)*jacobian_N.col(1).transpose() +
			jacobian_N.col(2)*jacobian_N.col(2).transpose())
			- (grad_norm* grad_norm.transpose())) / norm;

	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> hess_normalizedN;
	for (int i = 0; i < 3; i++)
		hess_normalizedN[i] =
		(hess_N[i] / norm)
		- (jacobian_N.col(i)*grad_norm.transpose()) / pow(norm, 2)
		- (hess_norm*N(i) + grad_norm * jacobian_N.col(i).transpose()) / pow(norm, 2)
		+ (2 * N(i)*grad_norm*grad_norm.transpose()) / pow(norm, 3);

	return hess_normalizedN;
}

Eigen::Matrix<Eigen::Matrix<double, 12, 12>, 1, 6> BendingNormal::d2N_dxdx_perhinge(int hi) {
	int f0 = hinges_faceIndex[hi](0);
	int f1 = hinges_faceIndex[hi](1);
	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> n0_x = d2N_dxdx_perface(f0);
	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> n1_x = d2N_dxdx_perface(f1);
	Eigen::Matrix<Eigen::Matrix<double, 12, 12>, 1, 6> n_x;
	
	for (int i = 0; i < 3; i++) {
		//first face
		n_x[i].setZero();
		n_x[i].block<3, 3>(0, 0) = n0_x[i].block<3, 3>(x0_LocInd(hi, 0) * 3, x0_LocInd(hi, 0) * 3);
		n_x[i].block<3, 3>(0, 3) = n0_x[i].block<3, 3>(x0_LocInd(hi, 0) * 3, x1_LocInd(hi, 0) * 3);
		n_x[i].block<3, 3>(0, 6) = n0_x[i].block<3, 3>(x0_LocInd(hi, 0) * 3, x2_LocInd(hi, 0) * 3);

		n_x[i].block<3, 3>(3, 0) = n0_x[i].block<3, 3>(x1_LocInd(hi, 0) * 3, x0_LocInd(hi, 0) * 3);
		n_x[i].block<3, 3>(3, 3) = n0_x[i].block<3, 3>(x1_LocInd(hi, 0) * 3, x1_LocInd(hi, 0) * 3);
		n_x[i].block<3, 3>(3, 6) = n0_x[i].block<3, 3>(x1_LocInd(hi, 0) * 3, x2_LocInd(hi, 0) * 3);

		n_x[i].block<3, 3>(6, 0) = n0_x[i].block<3, 3>(x2_LocInd(hi, 0) * 3, x0_LocInd(hi, 0) * 3);
		n_x[i].block<3, 3>(6, 3) = n0_x[i].block<3, 3>(x2_LocInd(hi, 0) * 3, x1_LocInd(hi, 0) * 3);
		n_x[i].block<3, 3>(6, 6) = n0_x[i].block<3, 3>(x2_LocInd(hi, 0) * 3, x2_LocInd(hi, 0) * 3);
		
		// second face
		n_x[3 + i].setZero();
		n_x[3 + i].block<3, 3>(0, 0) = n1_x[0 + i].block<3, 3>(x0_LocInd(hi, 1) * 3, x0_LocInd(hi, 1) * 3);
		n_x[3 + i].block<3, 3>(0, 3) = n1_x[0 + i].block<3, 3>(x0_LocInd(hi, 1) * 3, x1_LocInd(hi, 1) * 3);
		n_x[3 + i].block<3, 3>(0, 9) = n1_x[0 + i].block<3, 3>(x0_LocInd(hi, 1) * 3, x3_LocInd(hi, 1) * 3);

		n_x[3 + i].block<3, 3>(3, 0) = n1_x[0 + i].block<3, 3>(x1_LocInd(hi, 1) * 3, x0_LocInd(hi, 1) * 3);
		n_x[3 + i].block<3, 3>(3, 3) = n1_x[0 + i].block<3, 3>(x1_LocInd(hi, 1) * 3, x1_LocInd(hi, 1) * 3);
		n_x[3 + i].block<3, 3>(3, 9) = n1_x[0 + i].block<3, 3>(x1_LocInd(hi, 1) * 3, x3_LocInd(hi, 1) * 3);

		n_x[3 + i].block<3, 3>(9, 0) = n1_x[0 + i].block<3, 3>(x3_LocInd(hi, 1) * 3, x0_LocInd(hi, 1) * 3);
		n_x[3 + i].block<3, 3>(9, 3) = n1_x[0 + i].block<3, 3>(x3_LocInd(hi, 1) * 3, x1_LocInd(hi, 1) * 3);
		n_x[3 + i].block<3, 3>(9, 9) = n1_x[0 + i].block<3, 3>(x3_LocInd(hi, 1) * 3, x3_LocInd(hi, 1) * 3);
	}
	return n_x;
}

int BendingNormal::x_GlobInd(int index, int hi) {
	if (index == 0)
		return x0_GlobInd(hi);
	if (index == 1)
		return x1_GlobInd(hi);
	if (index == 2)
		return x2_GlobInd(hi);
	if (index == 3)
		return x3_GlobInd(hi);
}
