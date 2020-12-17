﻿#include "AuxSpherePerHinge.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <igl/triangle_triangle_adjacency.h>


AuxSpherePerHinge::AuxSpherePerHinge(FunctionType type) {
	Cuda::AuxSpherePerHinge::functionType = type;
	name = "Aux Sphere Per Hinge";
	w = 0;
	std::cout << "\t" << name << " constructor" << std::endl;
}

AuxSpherePerHinge::~AuxSpherePerHinge() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void AuxSpherePerHinge::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if (restShapeV.size() == 0 || restShapeF.size() == 0)
		throw name + " must define members V,F before init()!";

	calculateHinges();
	
	restAreaPerHinge.resize(num_hinges);
	igl::doublearea(restShapeV, restShapeF, restAreaPerFace);
	restAreaPerFace /= 2;
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		restAreaPerHinge(hi) = (restAreaPerFace(f0) + restAreaPerFace(f1)) / 2;
	}
	
	d_center.resize(num_hinges);
	d_radius.resize(num_hinges);
	Cuda::AuxSpherePerHinge::planarParameter = 1;
	internalInitCuda();
}

void AuxSpherePerHinge::internalInitCuda() {
	const unsigned int numF = restShapeF.rows();
	const unsigned int numV = restShapeV.rows();
	const unsigned int numH = num_hinges;

	Cuda::initIndices(Cuda::AuxSpherePerHinge::mesh_indices, numF, numV, numH);

	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::restShapeF, numF);
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::restAreaPerFace, numF);
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::restAreaPerHinge, numH);
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::grad, (3 * numV) + (7 * numF));
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::EnergyAtomic, 1);
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::hinges_faceIndex, numH);
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::x0_GlobInd, numH);
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::x1_GlobInd, numH);
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::x2_GlobInd, numH);
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::x3_GlobInd, numH);
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::x0_LocInd, numH);
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::x1_LocInd, numH);
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::x2_LocInd, numH);
	Cuda::AllocateMemory(Cuda::AuxSpherePerHinge::x3_LocInd, numH);

	//init host buffers...
	for (int i = 0; i < Cuda::AuxSpherePerHinge::grad.size; i++) {
		Cuda::AuxSpherePerHinge::grad.host_arr[i] = 0;
	}
	for (int f = 0; f < restShapeF.rows(); f++) {
		Cuda::AuxSpherePerHinge::restShapeF.host_arr[f] = make_int3(restShapeF(f, 0), restShapeF(f, 1), restShapeF(f, 2));
		Cuda::AuxSpherePerHinge::restAreaPerFace.host_arr[f] = restAreaPerFace[f];
	}
	for (int h = 0; h < num_hinges; h++) {
		Cuda::AuxSpherePerHinge::restAreaPerHinge.host_arr[h] = restAreaPerHinge[h];
		Cuda::AuxSpherePerHinge::hinges_faceIndex.host_arr[h] = Cuda::newHinge(hinges_faceIndex[h][0], hinges_faceIndex[h][1]);
		Cuda::AuxSpherePerHinge::x0_GlobInd.host_arr[h] = x0_GlobInd[h];
		Cuda::AuxSpherePerHinge::x1_GlobInd.host_arr[h] = x1_GlobInd[h];
		Cuda::AuxSpherePerHinge::x2_GlobInd.host_arr[h] = x2_GlobInd[h];
		Cuda::AuxSpherePerHinge::x3_GlobInd.host_arr[h] = x3_GlobInd[h];
		Cuda::AuxSpherePerHinge::x0_LocInd.host_arr[h] = Cuda::newHinge(x0_LocInd(h, 0), x0_LocInd(h, 1));
		Cuda::AuxSpherePerHinge::x1_LocInd.host_arr[h] = Cuda::newHinge(x1_LocInd(h, 0), x1_LocInd(h, 1));
		Cuda::AuxSpherePerHinge::x2_LocInd.host_arr[h] = Cuda::newHinge(x2_LocInd(h, 0), x2_LocInd(h, 1));
		Cuda::AuxSpherePerHinge::x3_LocInd.host_arr[h] = Cuda::newHinge(x3_LocInd(h, 0), x3_LocInd(h, 1));
	}

	// Copy input vectors from host memory to GPU buffers.
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::grad);
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::restShapeF);
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::restAreaPerFace);
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::restAreaPerHinge);
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::hinges_faceIndex);
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::x0_GlobInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::x1_GlobInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::x2_GlobInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::x3_GlobInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::x0_LocInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::x1_LocInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::x2_LocInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxSpherePerHinge::x3_LocInd);
}

void AuxSpherePerHinge::updateX(Cuda::Array<double>& curr_x)
{
	////X = [
	////		x(0), ... ,x(#V-1), 
	////		y(0), ... ,y(#V-1), 
	////		z(0), ... ,z(#V-1), 
	////		Nx(0), ... ,Nx(#F-1),
	////		Ny(0), ... ,Ny(#F-1),
	////		Nz(0), ... ,Nz(#F-1),
	////		Cx,
	////		Cy,
	////		Cz,
	////		R
	////	  ]
	//assert(X.rows() == (restShapeV.size() + 7 * restShapeF.rows()));
	//CurrV =		Eigen::Map<const Eigen::MatrixX3d>(X.middleRows(0,3 * restShapeV.rows()).data(), restShapeV.rows(), 3);
	//CurrCenter = Eigen::Map<const Eigen::MatrixX3d>(X.middleRows(3 * restShapeV.rows()+ 3 * restShapeF.rows(),3 * restShapeF.rows()).data(), restShapeF.rows(), 3);
	//CurrRadius = Eigen::Map<const Eigen::VectorXd>(X.middleRows(3 * restShapeV.rows() + 6 * restShapeF.rows(), restShapeF.rows()).data(), restShapeF.rows(), 1);
	//
	//for (int hi = 0; hi < num_hinges; hi++) {
	//	int f0 = hinges_faceIndex[hi](0);
	//	int f1 = hinges_faceIndex[hi](1);
	//	d_center(hi) = (CurrCenter.row(f1) - CurrCenter.row(f0)).squaredNorm();
	//	d_radius(hi) = pow(CurrRadius(f1) - CurrRadius(f0), 2);
	//}
}

void AuxSpherePerHinge::calculateHinges() {
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

Eigen::VectorXd AuxSpherePerHinge::Phi(Eigen::VectorXd x) {
	if(Cuda::AuxSpherePerHinge::functionType == FunctionType::QUADRATIC)
		return x.cwiseAbs2();
	else if (Cuda::AuxSpherePerHinge::functionType == FunctionType::EXPONENTIAL) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = std::exp(x(i)*x(i));
		}
		return res;
	}
	else if (Cuda::AuxSpherePerHinge::functionType == FunctionType::SIGMOID) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			double x2 = pow(x(i), 2);
			res(i) = x2/(x2+ Cuda::AuxSpherePerHinge::planarParameter);
		}
		return res;
	}
}

Eigen::VectorXd AuxSpherePerHinge::dPhi_dm(Eigen::VectorXd x) {
	if (Cuda::AuxSpherePerHinge::functionType == FunctionType::QUADRATIC)
		return 2 * x;
	else if (Cuda::AuxSpherePerHinge::functionType == FunctionType::EXPONENTIAL) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = 2 * x(i) * std::exp(x(i)*x(i));
		}
		return res;
	}
	else if (Cuda::AuxSpherePerHinge::functionType == FunctionType::SIGMOID) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = (2*x(i)* Cuda::AuxSpherePerHinge::planarParameter) /
				pow(x(i)*x(i) + Cuda::AuxSpherePerHinge::planarParameter, 2);
		}
		return res;
	}
}

Eigen::VectorXd AuxSpherePerHinge::d2Phi_dmdm(Eigen::VectorXd x) {
	if (Cuda::AuxSpherePerHinge::functionType == FunctionType::QUADRATIC)
		return Eigen::VectorXd::Constant(x.rows(),2);
	else if (Cuda::AuxSpherePerHinge::functionType == FunctionType::EXPONENTIAL) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = (4 * x(i)*x(i) + 2) * std::exp(x(i)*x(i));
		}
		return res;
	}
	else if (Cuda::AuxSpherePerHinge::functionType == FunctionType::SIGMOID) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			double x2 = pow(x(i), 2);
			res(i) = (2* Cuda::AuxSpherePerHinge::planarParameter)*(-3*x2+ Cuda::AuxSpherePerHinge::planarParameter) /
				pow(x2 + Cuda::AuxSpherePerHinge::planarParameter,3);
		}
		return res;
	}
}

double AuxSpherePerHinge::value(Cuda::Array<double>& curr_x, const bool update)
{
#ifdef USING_CUDA
	double value = Cuda::AuxSpherePerHinge::value(curr_x);
#else
	//per hinge
	Eigen::VectorXd Energy1 = Phi(d_center + d_radius);
	
	//per face
	double Energy2 = 0; 
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		int x0 = restShapeF(fi, 0);
		int x1 = restShapeF(fi, 1);
		int x2 = restShapeF(fi, 2);
		Eigen::RowVector3d c = CurrCenter.row(fi);
		double r = CurrRadius(fi);
		Energy2 += pow((CurrV.row(x0) - c).squaredNorm() - pow(r, 2), 2);
		Energy2 += pow((CurrV.row(x1) - c).squaredNorm() - pow(r, 2), 2);
		Energy2 += pow((CurrV.row(x2) - c).squaredNorm() - pow(r, 2), 2);
	}

	double value =
		Cuda::AuxSpherePerHinge::w1 * Energy1.transpose()*restAreaPerHinge +
		Cuda::AuxSpherePerHinge::w2 * Energy2;
#endif
	if (update) {
		//TODO: calculate Efi (for coloring the faces)
		//Efi.setZero();
		energy_value = value;
	}
	return value;
}

void AuxSpherePerHinge::gradient(Cuda::Array<double>& X, Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(restShapeV.size() + 7*restShapeF.rows());
	g.setZero();
	
	//Energy 1: per hinge
	Eigen::VectorXd dphi_dm = dPhi_dm(d_center + d_radius);
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		Eigen::Matrix<double, 1, 8> dE_dx = Cuda::AuxSpherePerHinge::w1 *restAreaPerHinge(hi)*dphi_dm(hi) * dm_dN(hi).transpose();
		for (int xyz = 0; xyz < 3; ++xyz) {
			int start = 3 * restShapeV.rows() + 3 * restShapeF.rows();
			g[f0 + start + (xyz * restShapeF.rows())] += dE_dx(xyz);
			g[f1 + start + (xyz * restShapeF.rows())] += dE_dx(3 + xyz);
		}
		int start = 3 * restShapeV.rows() + 6 * restShapeF.rows();
		g[f0 + start] += dE_dx(6);
		g[f1 + start] += dE_dx(7);
	}


	//Energy 2: per face
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		int x0 = restShapeF(fi, 0);
		int x1 = restShapeF(fi, 1);
		int x2 = restShapeF(fi, 2);
		Eigen::RowVector3d c = CurrCenter.row(fi);
		double r = CurrRadius(fi);
		double sqrtE0 = (CurrV.row(x0) - c).squaredNorm() - pow(r, 2);
		double sqrtE1 = (CurrV.row(x1) - c).squaredNorm() - pow(r, 2);
		double sqrtE2 = (CurrV.row(x2) - c).squaredNorm() - pow(r, 2);
		
		Eigen::Matrix<double, 1, 13> g_sqrtE0, g_sqrtE1, g_sqrtE2;
		g_sqrtE0 <<
			2 * (CurrV(x0, 0) - c(0)), // V0x
			2 * (CurrV(x0, 1) - c(1)), // V0y
			2 * (CurrV(x0, 2) - c(2)), // V0z
			0, // V1x
			0, // V1y
			0, // V1z
			0, // V2x
			0, // V2y
			0, // V2z
			-2 * (CurrV(x0, 0) - c(0)), // Cx
			-2 * (CurrV(x0, 1) - c(1)), // Cy
			-2 * (CurrV(x0, 2) - c(2)), // Cz
			-2 * r; //r
		g_sqrtE1 <<
			0, // V0x
			0, // V0y
			0, // V0z
			2 * (CurrV(x1, 0) - c(0)), // V1x
			2 * (CurrV(x1, 1) - c(1)), // V1y
			2 * (CurrV(x1, 2) - c(2)), // V1z
			0, // V2x
			0, // V2y
			0, // V2z
			-2 * (CurrV(x1, 0) - c(0)), // Cx
			-2 * (CurrV(x1, 1) - c(1)), // Cy
			-2 * (CurrV(x1, 2) - c(2)), // Cz
			-2 * r; //r
		g_sqrtE2 <<
			0, // V0x
			0, // V0y
			0, // V0z
			0, // V1x
			0, // V1y
			0, // V1z
			2 * (CurrV(x2, 0) - c(0)), // V2x
			2 * (CurrV(x2, 1) - c(1)), // V2y
			2 * (CurrV(x2, 2) - c(2)), // V2z
			-2 * (CurrV(x2, 0) - c(0)), // Cx
			-2 * (CurrV(x2, 1) - c(1)), // Cy
			-2 * (CurrV(x2, 2) - c(2)), // Cz
			-2 * r; //r
		Eigen::Matrix<double, 1, 13> dE_dx = Cuda::AuxSpherePerHinge::w2 * 2 *
			(sqrtE0*g_sqrtE0 + sqrtE1 * g_sqrtE1 + sqrtE2 * g_sqrtE2);
		
		int startC = 3 * restShapeV.rows() + 3 * restShapeF.rows();
		int startR = 3 * restShapeV.rows() + 6 * restShapeF.rows();
		for (int xyz = 0; xyz < 3; ++xyz) {
			g[x0 + (xyz * restShapeV.rows())] += dE_dx(0+xyz);
			g[x1 + (xyz * restShapeV.rows())] += dE_dx(3+xyz);
			g[x2 + (xyz * restShapeV.rows())] += dE_dx(6+xyz);
			g[fi + startC + (xyz * restShapeF.rows())] += dE_dx(9+xyz);
		}
		g[fi + startR] += dE_dx(12);
	}
		
	if (update)
		gradient_norm = g.norm();
}

Eigen::Matrix< double, 8, 1> AuxSpherePerHinge::dm_dN(int hi) {
	int f0 = hinges_faceIndex[hi](0);
	int f1 = hinges_faceIndex[hi](1);
	Eigen::Matrix< double, 8, 1> grad;
	grad <<
		-2 * (CurrCenter(f1, 0) - CurrCenter(f0, 0)),	//C0.x
		-2 * (CurrCenter(f1, 1) - CurrCenter(f0, 1)),	//C0.y
		-2 * (CurrCenter(f1, 2) - CurrCenter(f0, 2)),	//C0.z
		2 * (CurrCenter(f1, 0) - CurrCenter(f0, 0)),	//C1.x
		2 * (CurrCenter(f1, 1) - CurrCenter(f0, 1)),	//C1.y
		2 * (CurrCenter(f1, 2) - CurrCenter(f0, 2)),	//C1.z
		-2 * (CurrRadius(f1) - CurrRadius(f0)),			//r0
		2 * (CurrRadius(f1) - CurrRadius(f0));			//r1
	return grad;
}
