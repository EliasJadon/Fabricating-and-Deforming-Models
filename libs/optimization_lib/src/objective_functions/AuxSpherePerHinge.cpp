#include "objective_functions/AuxSpherePerHinge.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <igl/triangle_triangle_adjacency.h>

AuxSpherePerHinge::AuxSpherePerHinge(OptimizationUtils::FunctionType type) {
	functionType = type;
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
	planarParameter = 1;
	init_hessian();
}

void AuxSpherePerHinge::updateX(const Eigen::VectorXd& X)
{
	//X = [
	//		x(0), ... ,x(#V-1), 
	//		y(0), ... ,y(#V-1), 
	//		z(0), ... ,z(#V-1), 
	//		Nx(0), ... ,Nx(#F-1),
	//		Ny(0), ... ,Ny(#F-1),
	//		Nz(0), ... ,Nz(#F-1),
	//		Cx,
	//		Cy,
	//		Cz,
	//		R
	//	  ]
	assert(X.rows() == (restShapeV.size() + 7 * restShapeF.rows()));
	CurrV =		Eigen::Map<const Eigen::MatrixX3d>(X.middleRows(0,3 * restShapeV.rows()).data(), restShapeV.rows(), 3);
	CurrCenter = Eigen::Map<const Eigen::MatrixX3d>(X.middleRows(3 * restShapeV.rows()+ 3 * restShapeF.rows(),3 * restShapeF.rows()).data(), restShapeF.rows(), 3);
	CurrRadius = Eigen::Map<const Eigen::VectorXd>(X.middleRows(3 * restShapeV.rows() + 6 * restShapeF.rows(), restShapeF.rows()).data(), restShapeF.rows(), 1);
	
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		d_center(hi) = (CurrCenter.row(f1) - CurrCenter.row(f0)).squaredNorm();
		d_radius(hi) = pow(CurrRadius(f1) - CurrRadius(f0), 2);
	}
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

Eigen::VectorXd AuxSpherePerHinge::dPhi_dm(Eigen::VectorXd x) {
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

Eigen::VectorXd AuxSpherePerHinge::d2Phi_dmdm(Eigen::VectorXd x) {
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

double AuxSpherePerHinge::value(const bool update)
{
	//per hinge
	Eigen::VectorXd Energy1 = Phi(d_center + d_radius);
	
	//per face
	double Energy2 = 0; 
	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		int x0 = restShapeF(fi, 0);
		int x1 = restShapeF(fi, 1);
		int x2 = restShapeF(fi, 2);
		Eigen::RowVector3d x = ((CurrV.row(x0) + CurrV.row(x1) + CurrV.row(x2)) / 3);
		Eigen::RowVector3d c = CurrCenter.row(fi);
		double r = CurrRadius(fi);
		Energy2 += pow((x - c).squaredNorm() - pow(r, 2), 2);
	}

	double value =
		w1 * Energy1.transpose()*restAreaPerHinge +
		w2 * Energy2;

	if (update) {
		//TODO: calculate Efi (for coloring the faces)
		Efi.setZero();
		energy_value = value;
	}
	return value;
}

void AuxSpherePerHinge::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(restShapeV.size() + 7*restShapeF.rows());
	g.setZero();
	
	//Energy 1: per hinge
	Eigen::VectorXd dphi_dm = dPhi_dm(d_center + d_radius);
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		Eigen::Matrix<double, 1, 8> dE_dx = w1*restAreaPerHinge(hi)*dphi_dm(hi) * dm_dN(hi).transpose();
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
		Eigen::RowVector3d x = ((CurrV.row(x0) + CurrV.row(x1) + CurrV.row(x2)) / 3);
		Eigen::RowVector3d c = CurrCenter.row(fi);
		double r = CurrRadius(fi);
		double sqrtE = (x - c).squaredNorm() - pow(r, 2);
		
		Eigen::Matrix<double, 1, 13> dE_dx;
		dE_dx <<
			(4 / 3.0f)*sqrtE*(x(0) - c(0)), // V0x
			(4 / 3.0f)*sqrtE*(x(1) - c(1)), // V0y
			(4 / 3.0f)*sqrtE*(x(2) - c(2)), // V0z
			(4 / 3.0f)*sqrtE*(x(0) - c(0)), // V1x
			(4 / 3.0f)*sqrtE*(x(1) - c(1)), // V1y
			(4 / 3.0f)*sqrtE*(x(2) - c(2)), // V1z
			(4 / 3.0f)*sqrtE*(x(0) - c(0)), // V2x
			(4 / 3.0f)*sqrtE*(x(1) - c(1)), // V2y
			(4 / 3.0f)*sqrtE*(x(2) - c(2)), // V2z
			-4 * sqrtE*(x(0) - c(0)), // Cx
			-4 * sqrtE*(x(1) - c(1)), // Cy
			-4 * sqrtE*(x(2) - c(2)), // Cz
			-4 * sqrtE*r; //r
		dE_dx *= w2;
		
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

Eigen::Matrix< double, 8, 8> AuxSpherePerHinge::d2m_dNdN(int hi) {
	Eigen::Matrix< double, 8, 8> hess;
	hess <<
		2, 0, 0, -2, 0, 0, 0, 0,	//C0.x
		0, 2, 0, 0, -2, 0, 0, 0,	//C0.y
		0, 0, 2, 0, 0, -2, 0, 0,	//C0.z
		-2, 0, 0, 2, 0, 0, 0, 0,	//C1.x
		0, -2, 0, 0, 2, 0, 0, 0,	//C1.y
		0, 0, -2, 0, 0, 2, 0, 0,	//C1.z
		0, 0, 0, 0, 0, 0, 2, -2,	//r0
		0, 0, 0, 0, 0, 0, -2, 2;	//r1
	return hess;
}

void AuxSpherePerHinge::hessian() {
	II.clear();
	JJ.clear();
	SS.clear();
	
	Eigen::VectorXd phi_m = dPhi_dm(d_center + d_radius);
	Eigen::VectorXd phi2_mm = d2Phi_dmdm(d_center + d_radius);

	//Energy 1: per hinge
	for (int hi = 0; hi < num_hinges; hi++) {
		Eigen::Matrix<double, 8, 1> m_n = dm_dN(hi);
		Eigen::Matrix<double, 8, 8> m2_nn = d2m_dNdN(hi);
		Eigen::Matrix<double, 8, 8> dE_dx =
			phi_m(hi) * m2_nn +
			m_n * phi2_mm(hi) * m_n.transpose();
		dE_dx *= restAreaPerHinge(hi);
		dE_dx *= w1;

		for (int fk = 0; fk < 2; fk++) {
			for (int fj = 0; fj < 2; fj++) {
				int f0 = hinges_faceIndex[hi](fk);
				int f1 = hinges_faceIndex[hi](fj);
				int start = 3 * restShapeV.rows() + 6 * restShapeF.rows();
				int Grow = f0 + start;
				int Gcol = f1 + start;
				if (Grow <= Gcol) {
					II.push_back(Grow);
					JJ.push_back(Gcol);
					SS.push_back(dE_dx(fk + 6, fj + 6));
				}
				for (int xyz1 = 0; xyz1 < 3; ++xyz1) {
					int startc = 3 * restShapeV.rows() + 6 * restShapeF.rows();
					int startr = 3 * restShapeV.rows() + 3 * restShapeF.rows();

					int Grow = f0 + startr + (xyz1 * restShapeF.rows());
					int Gcol = f1 + startc;
					if (Grow <= Gcol) {
						II.push_back(Grow);
						JJ.push_back(Gcol);
						SS.push_back(dE_dx(3 * fk + xyz1, fj + 6));
					}
					for (int xyz2 = 0; xyz2 < 3; ++xyz2) {
						int start = 3 * restShapeV.rows() + 3 * restShapeF.rows();
						int Grow = f0 + start + (xyz1 * restShapeF.rows());
						int Gcol = f1 + start + (xyz2 * restShapeF.rows());
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

	////Energy 2: per face
	//for (int fi = 0; fi < restShapeF.rows(); fi++) {
	//	
	//	Eigen::Matrix<double, 3, 3> dE_dx;
	//	dE_dx <<
	//		4 * (CurrN.row(fi).squaredNorm() - 1) + 8 * pow(CurrN(fi, 0), 2), 8 * CurrN(fi, 0)*CurrN(fi, 1)										, 8 * CurrN(fi, 0)*CurrN(fi, 2),
	//		8 * CurrN(fi, 0)*CurrN(fi, 1)									, 4 * (CurrN.row(fi).squaredNorm() - 1) + 8 * pow(CurrN(fi, 1), 2)	, 8 * CurrN(fi, 1)*CurrN(fi, 2),
	//		8 * CurrN(fi, 0)*CurrN(fi, 2)									, 8 * CurrN(fi, 1)*CurrN(fi, 2)										, 4 * (CurrN.row(fi).squaredNorm() - 1) + 8 * pow(CurrN(fi, 2), 2);
	//	dE_dx *= w2;

	//	for (int xyz1 = 0; xyz1 < 3; ++xyz1) {
	//		for (int xyz2 = 0; xyz2 < 3; ++xyz2) {
	//			int Grow = fi + (3 * restShapeV.rows()) + (xyz1 * restShapeF.rows());
	//			int Gcol = fi + (3 * restShapeV.rows()) + (xyz2 * restShapeF.rows());
	//			if (Grow <= Gcol) {
	//				II.push_back(Grow);
	//				JJ.push_back(Gcol);
	//				SS.push_back(dE_dx(xyz1, xyz2));
	//			}
	//		}
	//	}
	//}

	////Energy 3: per face
	//for (int fi = 0; fi < restShapeF.rows(); fi++) {
	//	int x0 = restShapeF(fi, 0);
	//	int x1 = restShapeF(fi, 1);
	//	int x2 = restShapeF(fi, 2);
	//	Eigen::Matrix<double,3,1> e21 = CurrV.row(x2) - CurrV.row(x1);
	//	Eigen::Matrix<double, 3, 1> e10 = CurrV.row(x1) - CurrV.row(x0);
	//	Eigen::Matrix<double, 3, 1> e02 = CurrV.row(x0) - CurrV.row(x2);
	//	Eigen::Matrix<double, 1, 3> N = CurrN.row(fi);
	//	
	//	double NxNx = N(0)*N(0);
	//	double NxNy = N(0)*N(1);
	//	double NxNz = N(0)*N(2);
	//	double NyNy = N(1)*N(1);
	//	double NyNz = N(1)*N(2);
	//	double NzNz = N(2)*N(2);
	//	
	//	double Ne02 = N * e02;
	//	double Ne21 = N * e21;
	//	double Ne10 = N * e10;

	//	
	//	double d_x0Nx = 2 * (Ne02 - Ne10) + 2 * N(0)*(e02(0) - e10(0));
	//	double d_x0Ny = 2 * N(0)*(e02(1) - e10(1));
	//	double d_x0Nz = 2 * N(0)*(e02(2) - e10(2));

	//	double d_y0Nx = 2 * N(1)*(e02(0) - e10(0)); 
	//	double d_y0Ny = 2 * (Ne02 - Ne10) + 2 * N(1)*(e02(1) - e10(1));
	//	double d_y0Nz = 2 * N(1)*(e02(2) - e10(2));

	//	double d_z0Nx = 2 * N(2)*(e02(0) - e10(0));
	//	double d_z0Ny = 2 * N(2)*(e02(1) - e10(1));
	//	double d_z0Nz = 2 * (Ne02 - Ne10) + 2 * N(2)*(e02(2) - e10(2));

	//	//////////////////////////////////////////
	//	double d_x1Nx = 2 * (Ne10 - Ne21) + 2 * N(0)*(e10(0) - e21(0));
	//	double d_x1Ny = 2 * N(0)*(e10(1) - e21(1));
	//	double d_x1Nz = 2 * N(0)*(e10(2) - e21(2));

	//	double d_y1Nx = 2 * N(1)*(e10(0) - e21(0)); 
	//	double d_y1Ny = 2 * (Ne10 - Ne21) + 2 * N(1)*(e10(1) - e21(1));
	//	double d_y1Nz = 2 * N(1)*(e10(2) - e21(2));

	//	double d_z1Nx = 2 * N(2)*(e10(0) - e21(0));
	//	double d_z1Ny = 2 * N(2)*(e10(1) - e21(1));
	//	double d_z1Nz = 2 * (Ne10 - Ne21) + 2 * N(2)*(e10(2) - e21(2));
	//	//////////////////////////////////////////
	//	double d_x2Nx = 2 * (Ne21 - Ne02) + 2 * N(0)*(e21(0) - e02(0));
	//	double d_x2Ny = 2 * N(0)*(e21(1) - e02(1));
	//	double d_x2Nz = 2 * N(0)*(e21(2) - e02(2));

	//	double d_y2Nx = 2 * N(1)*(e21(0) - e02(0));
	//	double d_y2Ny = 2 * (Ne21 - Ne02) + 2 * N(1)*(e21(1) - e02(1));
	//	double d_y2Nz = 2 * N(1)*(e21(2) - e02(2));

	//	double d_z2Nx = 2 * N(2)*(e21(0) - e02(0));
	//	double d_z2Ny = 2 * N(2)*(e21(1) - e02(1));
	//	double d_z2Nz = 2 * (Ne21 - Ne02) + 2 * N(2)*(e21(2) - e02(2));
	//	////////////////////////////////////////////////
	//	double d_NxNx = 2 * e10(0)*e10(0) + 2 * e21(0)*e21(0) + 2 * e02(0)*e02(0);
	//	double d_NxNy = 2 * e10(1)*e10(0) + 2 * e21(1)*e21(0) + 2 * e02(1)*e02(0);
	//	double d_NxNz = 2 * e10(2)*e10(0) + 2 * e21(2)*e21(0) + 2 * e02(2)*e02(0);

	//	double d_NyNy = 2 * e10(1)*e10(1) + 2 * e21(1)*e21(1) + 2 * e02(1)*e02(1);
	//	double d_NyNz = 2 * e10(2)*e10(1) + 2 * e21(2)*e21(1) + 2 * e02(2)*e02(1);
	//	
	//	double d_NzNz = 2 * e10(2)*e10(2) + 2 * e21(2)*e21(2) + 2 * e02(2)*e02(2);
	//	
	//	Eigen::Matrix<double, 12, 12> dE_dx;
	//	dE_dx <<
	//		NxNx * 4, NxNy * 4, NxNz * 4, -2 * NxNx, -2 * NxNy, -2 * NxNz, -2 * NxNx, -2 * NxNy, -2 * NxNz, d_x0Nx, d_x0Ny, d_x0Nz,
	//		NxNy * 4, NyNy * 4, NyNz * 4, -2 * NxNy, -2 * NyNy, -2 * NyNz, -2 * NxNy, -2 * NyNy, -2 * NyNz, d_y0Nx, d_y0Ny, d_y0Nz,
	//		NxNz * 4, NyNz * 4, NzNz * 4, -2 * NxNz, -2 * NyNz, -2 * NzNz, -2 * NxNz, -2 * NyNz, -2 * NzNz, d_z0Nx, d_z0Ny, d_z0Nz,
	//		-2 * NxNx, -2 * NxNy, -2 * NxNz, 4 * NxNx, 4 * NxNy, 4 * NxNz, -2 * NxNx, -2 * NxNy, -2 * NxNz, d_x1Nx, d_x1Ny, d_x1Nz,
	//		-2 * NxNy, -2 * NyNy, -2 * NyNz, 4 * NxNy, 4 * NyNy, 4 * NyNz, -2 * NxNy, -2 * NyNy, -2 * NyNz, d_y1Nx, d_y1Ny, d_y1Nz,
	//		-2 * NxNz, -2 * NyNz, -2 * NzNz, 4 * NxNz, 4 * NyNz, 4 * NzNz, -2 * NxNz, -2 * NyNz, -2 * NzNz, d_z1Nx, d_z1Ny, d_z1Nz,
	//		-2 * NxNx, -2 * NxNy, -2 * NxNz, -2 * NxNx, -2 * NxNy, -2 * NxNz, 4 * NxNx, 4 * NxNy, 4 * NxNz, d_x2Nx, d_x2Ny, d_x2Nz,
	//		-2 * NxNy, -2 * NyNy, -2 * NyNz, -2 * NxNy, -2 * NyNy, -2 * NyNz, 4 * NxNy, 4 * NyNy, 4 * NyNz, d_y2Nx, d_y2Ny, d_y2Nz,
	//		-2 * NxNz, -2 * NyNz, -2 * NzNz, -2 * NxNz, -2 * NyNz, -2 * NzNz, 4 * NxNz, 4 * NyNz, 4 * NzNz, d_z2Nx, d_z2Ny, d_z2Nz,
	//		d_x0Nx, d_y0Nx, d_z0Nx, d_x1Nx, d_y1Nx, d_z1Nx, d_x2Nx, d_y2Nx, d_z2Nx, d_NxNx, d_NxNy, d_NxNz,
	//		d_x0Ny, d_y0Ny, d_z0Ny, d_x1Ny, d_y1Ny, d_z1Ny, d_x2Ny, d_y2Ny, d_z2Ny, d_NxNy, d_NyNy, d_NyNz,
	//		d_x0Nz, d_y0Nz, d_z0Nz, d_x1Nz, d_y1Nz, d_z1Nz, d_x2Nz, d_y2Nz, d_z2Nz, d_NxNz, d_NyNz, d_NzNz;
	//	dE_dx *= w3;

	//	
	//	for (int xyz1 = 0; xyz1 < 3; ++xyz1) {
	//		for (int xyz2 = 0; xyz2 < 3; ++xyz2) {
	//			
	//			int Grow = fi + (3 * restShapeV.rows()) + (xyz1 * restShapeF.rows());
	//			int Gcol = fi + (3 * restShapeV.rows()) + (xyz2 * restShapeF.rows());
	//			if (Grow <= Gcol) {
	//				II.push_back(Grow);
	//				JJ.push_back(Gcol);
	//				SS.push_back(dE_dx(9+xyz1, 9+xyz2));
	//			}

	//			for (int xi = 0; xi < 3; ++xi) {
	//				Gcol = fi + (3 * restShapeV.rows()) + (xyz2 * restShapeF.rows());
	//				Grow = restShapeF(fi, xi) + (xyz1 * restShapeV.rows());
	//				if (Grow <= Gcol) {
	//					II.push_back(Grow);
	//					JJ.push_back(Gcol);
	//					SS.push_back(dE_dx(3 * xi + xyz1, 9 + xyz2));
	//				}
	//			}

	//			for (int xi = 0; xi < 3; ++xi) {
	//				for (int xj = 0; xj < 3; ++xj) {
	//					Grow = restShapeF(fi, xi) + (xyz1 * restShapeV.rows());
	//					Gcol = restShapeF(fi, xj) + (xyz2 * restShapeV.rows());
	//					if (Grow <= Gcol) {
	//						II.push_back(Grow);
	//						JJ.push_back(Gcol);
	//						SS.push_back(dE_dx(3 * xi + xyz1, 3 * xj + xyz2));
	//					}
	//				}
	//			}		
	//		}
	//	}
	//}
		
	II.push_back(restShapeV.size() + 7 * restShapeF.rows() - 1);
	JJ.push_back(restShapeV.size() + 7 * restShapeF.rows() - 1);
	SS.push_back(0);
}

void AuxSpherePerHinge::init_hessian()
{
	
}
