#include "AuxBendingNormal.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <igl/triangle_triangle_adjacency.h>

AuxBendingNormal::AuxBendingNormal(FunctionType type) {
	Cuda::AuxBendingNormal::functionType = type;
	name = "Aux Bending Normal";
	w = 1;
	std::cout << "\t" << name << " constructor" << std::endl;
}

AuxBendingNormal::~AuxBendingNormal() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void AuxBendingNormal::init()
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
	
	d_normals.resize(num_hinges);
	Cuda::AuxBendingNormal::planarParameter = 1;
	Efi.setZero();
	internalInitCuda();
}


void AuxBendingNormal::internalInitCuda() {
	const unsigned int numF = restShapeF.rows();
	const unsigned int numV = restShapeV.rows();
	const unsigned int numH = num_hinges;

	Cuda::initIndices(Cuda::AuxBendingNormal::mesh_indices, numF, numV, numH);
	
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::restShapeF, numF);
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::restAreaPerFace,numF);
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::restAreaPerHinge,numH);
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::grad, (3 * numV) + (7 * numF));
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::EnergyAtomic,1);
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::hinges_faceIndex,numH);
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::x0_GlobInd,numH);
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::x1_GlobInd,numH);
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::x2_GlobInd,numH);
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::x3_GlobInd,numH);
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::x0_LocInd,numH);
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::x1_LocInd,numH);
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::x2_LocInd,numH);
	Cuda::AllocateMemory(Cuda::AuxBendingNormal::x3_LocInd,numH);

	//init host buffers...
	for (int i = 0; i < Cuda::AuxBendingNormal::grad.size; i++) {
		Cuda::AuxBendingNormal::grad.host_arr[i] = 0;
	}
	for (int f = 0; f < restShapeF.rows(); f++) {
		Cuda::AuxBendingNormal::restShapeF.host_arr[f] = make_int3(restShapeF(f, 0), restShapeF(f, 1), restShapeF(f, 2));
		Cuda::AuxBendingNormal::restAreaPerFace.host_arr[f] = restAreaPerFace[f];
	}
	for (int h = 0; h < num_hinges; h++) {
		Cuda::AuxBendingNormal::restAreaPerHinge.host_arr[h] = restAreaPerHinge[h];
		Cuda::AuxBendingNormal::hinges_faceIndex.host_arr[h] = Cuda::newHinge(hinges_faceIndex[h][0], hinges_faceIndex[h][1]);
		Cuda::AuxBendingNormal::x0_GlobInd.host_arr[h] = x0_GlobInd[h];
		Cuda::AuxBendingNormal::x1_GlobInd.host_arr[h] = x1_GlobInd[h];
		Cuda::AuxBendingNormal::x2_GlobInd.host_arr[h] = x2_GlobInd[h];
		Cuda::AuxBendingNormal::x3_GlobInd.host_arr[h] = x3_GlobInd[h];
		Cuda::AuxBendingNormal::x0_LocInd.host_arr[h] = Cuda::newHinge(x0_LocInd(h, 0), x0_LocInd(h, 1));
		Cuda::AuxBendingNormal::x1_LocInd.host_arr[h] = Cuda::newHinge(x1_LocInd(h, 0), x1_LocInd(h, 1));
		Cuda::AuxBendingNormal::x2_LocInd.host_arr[h] = Cuda::newHinge(x2_LocInd(h, 0), x2_LocInd(h, 1));
		Cuda::AuxBendingNormal::x3_LocInd.host_arr[h] = Cuda::newHinge(x3_LocInd(h, 0), x3_LocInd(h, 1));
	}

	// Copy input vectors from host memory to GPU buffers.
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::grad);
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::restShapeF);
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::restAreaPerFace);
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::restAreaPerHinge);
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::hinges_faceIndex);
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::x0_GlobInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::x1_GlobInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::x2_GlobInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::x3_GlobInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::x0_LocInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::x1_LocInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::x2_LocInd);
	Cuda::MemCpyHostToDevice(Cuda::AuxBendingNormal::x3_LocInd);
}

void AuxBendingNormal::updateX(Cuda::Array<double>& curr_x)
{
	//X = [
	//		x(0), ... ,x(#V-1), 
	//		y(0), ... ,y(#V-1), 
	//		z(0), ... ,z(#V-1), 
	//		Nx(0), ... ,Nx(#F-1),
	//		Ny(0), ... ,Ny(#F-1),
	//		Nz(0), ... ,Nz(#F-1)
	//	  ]
	//assert(X.rows() == (restShapeV.size() + 7*restShapeF.rows()));
	
#ifdef USING_CUDA
	/*int numV = restShapeV.rows();
	int num2V = 2 * numV;
	int numF = restShapeF.rows();
	int startNx = 3 * numV;
	int startNy = 3 * numV + numF;
	int startNz = 3 * numV + 2 * numF;

	for (int v = 0; v < numV; v++)
		Cuda::AuxBendingNormal::CurrV.host_arr[v] =
		Cuda::newRowVector<double>(X[v], X[v + numV], X[v + num2V]);
	for (int f = 0; f < numF; f++)
		Cuda::AuxBendingNormal::CurrN.host_arr[f] =
		Cuda::newRowVector<double>(X[f + startNx], X[f + startNy], X[f + startNz]);
	*///Cuda::AuxBendingNormal::updateX();
#else
	CurrV = Eigen::Map<const Eigen::MatrixX3d>(X.middleRows(0, 3 * restShapeV.rows()).data(), restShapeV.rows(), 3);
	CurrN = Eigen::Map<const Eigen::MatrixX3d>(X.middleRows(3 * restShapeV.rows(), 3 * restShapeF.rows()).data(), restShapeF.rows(), 3);

	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		d_normals(hi) = (CurrN.row(f1) - CurrN.row(f0)).squaredNorm();
	}
#endif
}

void AuxBendingNormal::calculateHinges() {
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

Eigen::VectorXd AuxBendingNormal::Phi(Eigen::VectorXd x) {
	if(Cuda::AuxBendingNormal::functionType == FunctionType::QUADRATIC)
		return x.cwiseAbs2();
	else if (Cuda::AuxBendingNormal::functionType == FunctionType::EXPONENTIAL) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = std::exp(x(i)*x(i));
		}
		return res;
	}
	else if (Cuda::AuxBendingNormal::functionType == FunctionType::SIGMOID) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			double x2 = pow(x(i), 2);
			res(i) = x2/(x2+ Cuda::AuxBendingNormal::planarParameter);
		}
		return res;
	}
}

Eigen::VectorXd AuxBendingNormal::dPhi_dm(Eigen::VectorXd x) {
	if (Cuda::AuxBendingNormal::functionType == FunctionType::QUADRATIC)
		return 2 * x;
	else if (Cuda::AuxBendingNormal::functionType == FunctionType::EXPONENTIAL) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = 2 * x(i) * std::exp(x(i)*x(i));
		}
		return res;
	}
	else if (Cuda::AuxBendingNormal::functionType == FunctionType::SIGMOID) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = (2*x(i)* Cuda::AuxBendingNormal::planarParameter) /
				pow(x(i)*x(i) + Cuda::AuxBendingNormal::planarParameter, 2);
		}
		return res;
	}
}

Eigen::VectorXd AuxBendingNormal::d2Phi_dmdm(Eigen::VectorXd x) {
	if (Cuda::AuxBendingNormal::functionType == FunctionType::QUADRATIC)
		return Eigen::VectorXd::Constant(x.rows(),2);
	else if (Cuda::AuxBendingNormal::functionType == FunctionType::EXPONENTIAL) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = (4 * x(i)*x(i) + 2) * std::exp(x(i)*x(i));
		}
		return res;
	}
	else if (Cuda::AuxBendingNormal::functionType == FunctionType::SIGMOID) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			double x2 = pow(x(i), 2);
			res(i) = (2* Cuda::AuxBendingNormal::planarParameter)*(-3*x2+ Cuda::AuxBendingNormal::planarParameter) /
				pow(x2 + Cuda::AuxBendingNormal::planarParameter,3);
		}
		return res;
	}
}

double AuxBendingNormal::value(Cuda::Array<double>& curr_x, const bool update)
{
#ifdef USING_CUDA
	double value = Cuda::AuxBendingNormal::value(curr_x);
#else
	//per hinge
	Eigen::VectorXd Energy1 = Phi(d_normals);
	
	//per face
	double Energy2 = 0; // (||N||^2 - 1)^2
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		Energy2 += pow(CurrN.row(fi).squaredNorm() - 1, 2);
	}

	double Energy3 = 0; // (N^T*(x1-x0))^2 + (N^T*(x2-x1))^2 + (N^T*(x0-x2))^2
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		int x0 = restShapeF(fi, 0);
		int x1 = restShapeF(fi, 1);
		int x2 = restShapeF(fi, 2);
		Eigen::VectorXd e21 = CurrV.row(x2) - CurrV.row(x1);
		Eigen::VectorXd e10 = CurrV.row(x1) - CurrV.row(x0);
		Eigen::VectorXd e02 = CurrV.row(x0) - CurrV.row(x2);
		Energy3 += pow(CurrN.row(fi) * e21, 2);
		Energy3 += pow(CurrN.row(fi) * e10, 2);
		Energy3 += pow(CurrN.row(fi) * e02, 2);
	}

	double value =
		Cuda::AuxBendingNormal::w1 * Energy1.transpose()*restAreaPerHinge +
		Cuda::AuxBendingNormal::w2 * Energy2 +
		Cuda::AuxBendingNormal::w3 * Energy3;

	std::cout << "value1 = " << value1 << std::endl;
	std::cout << "value = " << value << std::endl;
#endif
	if (update)
		energy_value = value;
	return value;
}

void AuxBendingNormal::gradient(Cuda::Array<double>& X, Eigen::VectorXd& g, const bool update)
{
#ifdef USING_CUDA
	//g.conservativeResize(restShapeV.size() + 7 * restShapeF.rows());
	Cuda::AuxBendingNormal::gradient(X);
	/*for (int i = 0; i < Cuda::AuxBendingNormal::grad.size; i++) {
		g(i) = Cuda::AuxBendingNormal::grad.host_arr[i];
	}
	if (update)
		gradient_norm = g.norm();*/
#else
	g.conservativeResize(restShapeV.size() + 7*restShapeF.rows());
	g.setZero();
	
	//Energy 1: per hinge
	Eigen::VectorXd dphi_dm = dPhi_dm(d_normals);
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		Eigen::Matrix<double, 1, 6> dE_dx = Cuda::AuxBendingNormal::w1*restAreaPerHinge(hi)* dphi_dm(hi) * dm_dN(hi).transpose();
		for (int xyz = 0; xyz < 3; ++xyz) {
			g[f0 + (3 * restShapeV.rows()) + (xyz * restShapeF.rows())] += dE_dx(xyz);
			g[f1 + (3 * restShapeV.rows()) + (xyz * restShapeF.rows())] += dE_dx(3 + xyz);
		}
	}

	//Energy 2: per face
	for (int fi = 0; fi < restShapeF.rows(); fi++)
		for (int xyz = 0; xyz < 3; ++xyz)
			g[fi + (3 * restShapeV.rows()) + (xyz * restShapeF.rows())] += 
			Cuda::AuxBendingNormal::w2 * 4 * (CurrN.row(fi).squaredNorm() - 1)*CurrN(fi, xyz);
		
	//Energy 3: per face
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		int x0 = restShapeF(fi, 0);
		int x1 = restShapeF(fi, 1);
		int x2 = restShapeF(fi, 2);
		Eigen::VectorXd e21 = CurrV.row(x2) - CurrV.row(x1);
		Eigen::VectorXd e10 = CurrV.row(x1) - CurrV.row(x0);
		Eigen::VectorXd e02 = CurrV.row(x0) - CurrV.row(x2);
		
		Eigen::Matrix<double, 12, 1> dE_dx;
		dE_dx <<
			2 * CurrN(fi, 0) * (CurrN.row(fi) * e02 - CurrN.row(fi) * e10),//x0
			2 * CurrN(fi, 1) * (CurrN.row(fi) * e02 - CurrN.row(fi) * e10),//y0
			2 * CurrN(fi, 2) * (CurrN.row(fi) * e02 - CurrN.row(fi) * e10),//z0
			2 * CurrN(fi, 0) * (CurrN.row(fi) * e10 - CurrN.row(fi) * e21),//x1
			2 * CurrN(fi, 1) * (CurrN.row(fi) * e10 - CurrN.row(fi) * e21),//y1
			2 * CurrN(fi, 2) * (CurrN.row(fi) * e10 - CurrN.row(fi) * e21),//z1
			2 * CurrN(fi, 0) * (CurrN.row(fi) * e21 - CurrN.row(fi) * e02),//x2
			2 * CurrN(fi, 1) * (CurrN.row(fi) * e21 - CurrN.row(fi) * e02),//y2
			2 * CurrN(fi, 2) * (CurrN.row(fi) * e21 - CurrN.row(fi) * e02),//z2
			2 * CurrN.row(fi) * e10*e10(0) + 2 * CurrN.row(fi) * e21*e21(0) + 2 * CurrN.row(fi) * e02*e02(0),//Nx
			2 * CurrN.row(fi) * e10*e10(1) + 2 * CurrN.row(fi) * e21*e21(1) + 2 * CurrN.row(fi) * e02*e02(1),//Ny
			2 * CurrN.row(fi) * e10*e10(2) + 2 * CurrN.row(fi) * e21*e21(2) + 2 * CurrN.row(fi) * e02*e02(2);//Nz
		dE_dx *= Cuda::AuxBendingNormal::w3;

		for (int xyz = 0; xyz < 3; ++xyz) {
			g[x0 + (xyz * restShapeV.rows())] += dE_dx(xyz);
			g[x1 + (xyz * restShapeV.rows())] += dE_dx(3 + xyz);
			g[x2 + (xyz * restShapeV.rows())] += dE_dx(6 + xyz);
			g[fi + (3 * restShapeV.rows()) + (xyz * restShapeF.rows())] += dE_dx(9 + xyz);
		}
	}


	Eigen::VectorXd compareG(restShapeV.size() + 7 * restShapeF.rows());
	for (int i = 0; i < Cuda::AuxBendingNormal::grad.size; i++) {
		compareG(i) = Cuda::AuxBendingNormal::grad.host_arr[i];
	}
	std::cout << "g.norm() = " << g.norm() << std::endl;
	std::cout << "compareG.norm() = " << compareG.norm() << std::endl;
	
	if (update)
		gradient_norm = g.norm();
#endif
}

Eigen::Matrix< double, 6, 1> AuxBendingNormal::dm_dN(int hi) {
	// m = ||n1 - n0||^2
	// m = (n1.x - n0.x)^2 + (n1.y - n0.y)^2 + (n1.z - n0.z)^2
	int f0 = hinges_faceIndex[hi](0);
	int f1 = hinges_faceIndex[hi](1);
	Eigen::Matrix< double, 6, 1> grad;
	grad <<
		-2 * (CurrN(f1, 0) - CurrN(f0, 0)),	//n0.x
		-2 * (CurrN(f1, 1) - CurrN(f0, 1)), //n0.y
		-2 * (CurrN(f1, 2) - CurrN(f0, 2)), //n0.z
		2 *  (CurrN(f1, 0) - CurrN(f0, 0)),	//n1.x
		2 *  (CurrN(f1, 1) - CurrN(f0, 1)),	//n1.y
		2 *  (CurrN(f1, 2) - CurrN(f0, 2));	//n1.z
	return grad;
}

