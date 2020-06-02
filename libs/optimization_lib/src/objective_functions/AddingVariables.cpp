#include "objective_functions/AddingVariables.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <igl/triangle_triangle_adjacency.h>

AddingVariables::AddingVariables(OptimizationUtils::FunctionType type) {
	functionType = type;
	name = "Adding Variables";
	w = 0;
	std::cout << "\t" << name << " constructor" << std::endl;
}

AddingVariables::~AddingVariables() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void AddingVariables::init()
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
	planarParameter = 1;
	init_hessian();
}

void AddingVariables::updateX(const Eigen::VectorXd& X)
{
	//X = [
	//		x(0), ... ,x(#V-1), 
	//		y(0), ... ,y(#V-1), 
	//		z(0), ... ,z(#V-1), 
	//		Nx(0), ... ,Nx(#F-1),
	//		Ny(0), ... ,Ny(#F-1),
	//		Nz(0), ... ,Nz(#F-1)
	//	  ]
	assert(X.rows() == (restShapeV.size() + restShapeF.size()));
	CurrV = Eigen::Map<const Eigen::MatrixX3d>(X.middleRows(0,3 * restShapeV.rows()).data(), restShapeV.rows(), 3);
	CurrN = Eigen::Map<const Eigen::MatrixX3d>(X.bottomRows(3 * restShapeF.rows()).data(), restShapeF.rows(), 3);
	
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		d_normals(hi) = (CurrN.row(f1) - CurrN.row(f0)).squaredNorm();
	}
}

void AddingVariables::calculateHinges() {
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

Eigen::VectorXd AddingVariables::Phi(Eigen::VectorXd x) {
	if(functionType == OptimizationUtils::Quadratic)
		return x.cwiseAbs2();
	else if (functionType == OptimizationUtils::Exponential) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = std::exp(x(i)*x(i));
		}
		return res;
	}
	else if (functionType == OptimizationUtils::PlanarL) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			double x2 = pow(x(i), 2);
			res(i) = x2/(x2+planarParameter);
		}
		return res;
	}
}

Eigen::VectorXd AddingVariables::dPhi_dm(Eigen::VectorXd x) {
	if (functionType == OptimizationUtils::Quadratic)
		return 2 * x;
	else if (functionType == OptimizationUtils::Exponential) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = 2 * x(i) * std::exp(x(i)*x(i));
		}
		return res;
	}
	else if (functionType == OptimizationUtils::PlanarL) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = (2*x(i)*planarParameter) / 
				pow(x(i)*x(i) + planarParameter, 2);
		}
		return res;
	}
}

Eigen::VectorXd AddingVariables::d2Phi_dmdm(Eigen::VectorXd x) {
	if (functionType == OptimizationUtils::Quadratic)
		return Eigen::VectorXd::Constant(x.rows(),2);
	else if (functionType == OptimizationUtils::Exponential) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			res(i) = (4 * x(i)*x(i) + 2) * std::exp(x(i)*x(i));
		}
		return res;
	}
	else if (functionType == OptimizationUtils::PlanarL) {
		Eigen::VectorXd res(x.rows());
		for (int i = 0; i < x.rows(); i++) {
			double x2 = pow(x(i), 2);
			res(i) = (2* planarParameter)*(-3*x2+ planarParameter) / 
				pow(x2 + planarParameter,3);
		}
		return res;
	}
}

double AddingVariables::value(const bool update)
{
	//per hinge
	Eigen::VectorXd Energy1 = Phi(d_normals);
	
	//per face
	double Energy2 = CurrN.squaredNorm(); // ||N||^2
	double Energy3 = 0; // (N^T x)^2
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		int x0 = restShapeF(fi, 0);
		int x1 = restShapeF(fi, 1);
		int x2 = restShapeF(fi, 2);
		Energy3 += pow(CurrN.row(fi) * CurrV.row(x0).transpose(), 2);
		Energy3 += pow(CurrN.row(fi) * CurrV.row(x1).transpose(), 2);
		Energy3 += pow(CurrN.row(fi) * CurrV.row(x2).transpose(), 2);
	}

	double value =
		w1 * Energy1.transpose()*restAreaPerHinge +
		w2 * Energy2 + 
		w3 * Energy3;
	if (update) {
		//TODO: calculate Efi (for coloring the faces)
		Efi.setZero();
		energy_value = value;
	}
	return value;
}

void AddingVariables::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(restShapeV.size() + restShapeF.size());
	g.setZero();
	
	//Energy 1: per hinge
	Eigen::VectorXd dphi_dm = dPhi_dm(d_normals);
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		Eigen::Matrix<double, 1, 6> dE_dx = restAreaPerHinge(hi)* dphi_dm(hi) * dm_dN(hi).transpose();
		for (int xyz = 0; xyz < 3; ++xyz) {
			g[f0 + (3 * restShapeV.rows()) + (xyz * restShapeF.rows())] += dE_dx(xyz);
			g[f1 + (3 * restShapeV.rows()) + (xyz * restShapeF.rows())] += dE_dx(3 + xyz);
		}
	}

	//Energy 2: per face
	for (int fi = 0; fi < restShapeF.rows(); fi++)
		for (int xyz = 0; xyz < 3; ++xyz)
			g[fi + (3 * restShapeV.rows()) + (xyz * restShapeF.rows())] += w2 * 2 * CurrN(fi, xyz);
		
	//Energy 3: per face

	if (update)
		gradient_norm = g.norm();
}

Eigen::Matrix< double, 6, 1> AddingVariables::dm_dN(int hi) {
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

Eigen::Matrix< double, 6, 6> AddingVariables::d2m_dNdN(int hi) {
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

void AddingVariables::hessian() {
	II.clear();
	JJ.clear();
	SS.clear();
	
	Eigen::VectorXd phi_m = dPhi_dm(d_normals);
	Eigen::VectorXd phi2_mm = d2Phi_dmdm(d_normals);

	for (int hi = 0; hi < num_hinges; hi++) {
		Eigen::Matrix<double, 6, 1> m_n = dm_dN(hi);
		Eigen::Matrix<double, 6, 6> m2_nn = d2m_dNdN(hi);

		Eigen::Matrix<double, 6, 6> dE_dx =
			phi_m(hi) * m2_nn +
			m_n * phi2_mm(hi) * m_n.transpose();
		dE_dx *= restAreaPerHinge(hi);

		for (int fk = 0; fk < 2; fk++) {
			for (int fj = 0; fj < 2; fj++) {
				for (int xyz1 = 0; xyz1 < 3; ++xyz1) {
					for (int xyz2 = 0; xyz2 < 3; ++xyz2) {
						int f0 = hinges_faceIndex[hi](fk);
						int f1 = hinges_faceIndex[hi](fj);
						int Grow = f0 + (3 * restShapeV.rows()) + (xyz1 * restShapeF.rows());
						int Gcol = f1 + (3 * restShapeV.rows()) + (xyz2 * restShapeF.rows());
						if (Grow <= Gcol) {
							II.push_back(Grow);
							JJ.push_back(Gcol);
							SS.push_back(dE_dx(3 * fk + xyz1, 3 * fj + xyz2));
						}
					}
				}
			}
		}	
	}
}

int AddingVariables::x_GlobInd(int index, int hi) {
	if (index == 0)
		return x0_GlobInd(hi);
	if (index == 1)
		return x1_GlobInd(hi);
	if (index == 2)
		return x2_GlobInd(hi);
	if (index == 3)
		return x3_GlobInd(hi);
}

void AddingVariables::init_hessian()
{
	
}
