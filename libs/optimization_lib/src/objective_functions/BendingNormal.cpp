#include "objective_functions/BendingNormal.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <igl/triangle_triangle_adjacency.h>

BendingNormal::BendingNormal(OptimizationUtils::FunctionType type) {
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
	init_hessian();
}

void BendingNormal::updateX(const Eigen::VectorXd& X)
{
	assert(X.rows() == (3 * restShapeV.rows()));
	CurrV = Eigen::Map<const Eigen::MatrixX3d>(X.data(), X.rows() / 3, 3);
	igl::per_face_normals(CurrV, restShapeF, normals);
	for (int hi = 0; hi < num_hinges; hi++) {
		int f0 = hinges_faceIndex[hi](0);
		int f1 = hinges_faceIndex[hi](1);
		d_normals(hi) = (normals.row(f1) - normals.row(f0)).squaredNorm();
	}
}

void BendingNormal::calculateHinges() {
	std::vector<std::vector<std::vector<int>>> TT;
	igl::triangle_triangle_adjacency(restShapeF, TT);
	assert(TT.size() == restShapeF.rows());
	hinges_faceIndex.clear();
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
}

Eigen::VectorXd BendingNormal::Phi(Eigen::VectorXd x) {
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

Eigen::VectorXd BendingNormal::dPhi_dm(Eigen::VectorXd x) {
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

Eigen::VectorXd BendingNormal::d2Phi_dmdm(Eigen::VectorXd x) {
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

double BendingNormal::value(const bool update)
{
	//Eigen::VectorXd Energy = Phi(d_normals);
	//double value = Energy.transpose()*restArea;

	double value = normals.sum();
	
	if (update) {
		//TODO: calculate Efi (for coloring the faces)
		Efi.setZero();
		energy_value = value;
	}
	return value;
}

void BendingNormal::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(restShapeV.rows() * 3);
	g.setZero();

	// m = ||n1-n0||^2
	// E = Phi( ||n1-n0||^2 )
	// 
	// dE/dx = dPhi/dx
	// 
	// using chain rule:
	// dPhi/dx = dPhi/dm * dm/dn * dn/dx

	Eigen::VectorXd dphi_dm = dPhi_dm(d_normals);

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		Eigen::Matrix<double, 3, 9> asd = dN_dx(fi);
		int v0 = restShapeF(fi, 0);
		int v1 = restShapeF(fi, 1);
		int v2 = restShapeF(fi, 2);

		for (int ddd = 0; ddd < 3; ddd++) {
			for (int xyz = 0; xyz < 3; ++xyz) {
				g[v0 + (xyz*restShapeV.rows())] += asd(ddd, 0 + 3 * xyz);
				g[v1 + (xyz*restShapeV.rows())] += asd(ddd, 1 + 3 * xyz);
				g[v2 + (xyz*restShapeV.rows())] += asd(ddd, 2 + 3 * xyz);
			}
		}
		
			
	}

	/*for (int hi = 0; hi < num_hinges; hi++) {
		Eigen::Matrix<double, 4, 3> dE_dx = dphi_dm(hi) * dm_dN(hi) * dN_dx(hi);

		for(int i=0;i<4;i++)
			for (int xyz = 0; xyz < 3; ++xyz)
				g[x_index(i,hi) + (xyz*restShapeV.rows())] += dE_dx(i, xyz);
	}*/

	if (update)
		gradient_norm = g.norm();
}


Eigen::Matrix< double, 6, 1> BendingNormal::dm_dN(int hi) {
	// m = ||n1 - n0||^2
	// m = (n1.x - n0.x)^2 + (n1.y - n0.y)^2 + (n1.z - n0.z)^2
	int f0 = hinges_faceIndex[hi](0);
	int f1 = hinges_faceIndex[hi](1);
	Eigen::Matrix< double, 6, 1> grad;
	grad <<
		-2 * (normals(f1, 0) - normals(f0, 0)),	//n0.x
		2 * (normals(f1, 0) - normals(f0, 0)),	//n1.x
		-2 * (normals(f1, 1) - normals(f0, 1)), //n0.y
		2 * (normals(f1, 1) - normals(f0, 1)),	//n1.y
		-2 * (normals(f1, 2) - normals(f0, 2)), //n0.z
		2 * (normals(f1, 2) - normals(f0, 2));	//n1.z
	return grad;
}

Eigen::Matrix< double, 6, 6> BendingNormal::d2m_dNdN(int hi) {
	Eigen::Matrix< double, 6, 6> hess;
	hess << 
		2, -2, 0, 0, 0, 0,
		-2, 2, 0, 0, 0, 0,
		0, 0, 2, -2, 0, 0,
		0, 0, -2, 2, 0, 0,
		0, 0, 0, 0, 2, -2,
		0, 0, 0, 0, -2, 2;
	return hess;
}

Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> BendingNormal::d2N_dxdx(int fi) {
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
		0			, -e0(2) + e1(2), -e1(1) + e0(1),	//x0
		0			, -e1(2)		, e1(1),	//x1
		0			, e0(2)			, -e0(1),	//x2
		-e1(2) + e0(2), 0				, -e0(0) + e1(0),	//y0
		e1(2), 0				, -e1(0),	//y1
		-e0(2), 0				, e0(0),	//y2
		-e0(1) + e1(1), -e1(0) + e0(0), 0				,	//z0
		-e1(1), e1(0), 0				,	//z1
		e0(1), -e0(0), 0;				//z2

	grad_norm = (N(0)*jacobian_N.col(0) + N(1)*jacobian_N.col(1) + N(2)*jacobian_N.col(2)) / norm;
	
	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> hess_N;
	Eigen::Matrix<double, 9, 9> hess_norm;
	hess_N[0] <<
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 1, -1,
		0, 0, 0, 0, 0, 0, -1, 0, 1,
		0, 0, 0, 0, 0, 0, 1, -1, 0,
		0, 0, 0, 0, -1, 1, 0, 0, 0,
		0, 0, 0, 1, 0, -1, 0, 0, 0,
		0, 0, 0, -1, 1, 0, 0, 0, 0;
	hess_N[1] <<
		0, 0, 0, 0, 0, 0, 0, -1, 1,
		0, 0, 0, 0, 0, 0, 1, 0, -1,
		0, 0, 0, 0, 0, 0, -1, 1, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 1, -1, 0, 0, 0, 0, 0, 0,
		-1, 0, 1, 0, 0, 0, 0, 0, 0,
		1, -1, 0, 0, 0, 0, 0, 0, 0;
	hess_N[2] <<
		0, 0, 0, 0, 1, -1, 0, 0, 0,
		0, 0, 0, -1, 0, 1, 0, 0, 0,
		0, 0, 0, 1, -1, 0, 0, 0, 0,
		0, -1, 1, 0, 0, 0, 0, 0, 0,
		1, 0, -1, 0, 0, 0, 0, 0, 0,
		-1, 1, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 0;

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

Eigen::Matrix<double, 3, 9> BendingNormal::dN_dx(int fi) {
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
		0			, -e0(2) + e1(2), -e1(1) + e0(1),	//x0
		0			, -e1(2), e1(1),	//x1
		0			, e0(2), -e0(1),	//x2
		-e1(2) + e0(2), 0				, -e0(0) + e1(0),	//y0
		e1(2), 0				, -e1(0),	//y1
		-e0(2), 0				, e0(0),	//y2
		-e0(1) + e1(1), -e1(0) + e0(0), 0				,	//z0
		-e1(1), e1(0), 0				,	//z1
		e0(1), -e0(0), 0;				//z2
	
	grad_norm = (N(0)*jacobian_N.col(0) + N(1)*jacobian_N.col(1) + N(2)*jacobian_N.col(2))/norm;
	
	Eigen::Matrix<double, 3, 9> jacobian_normalizedN;
	for (int i = 0; i < 3; i++)
		jacobian_normalizedN.row(i) = (jacobian_N.col(i) / norm) - ((grad_norm*N(i)) / pow(norm, 2));
	return jacobian_normalizedN;
}

void BendingNormal::hessian() {
	II.clear();
	JJ.clear();
	SS.clear();
	//// constant factors
	//int index = 0;
	//Eigen::VectorXd d_angle = angle - restAngle;
	//Eigen::VectorXd dE_df = k * restConst;
	//Eigen::VectorXd df_d0 = dF_d0(d_angle);
	//Eigen::VectorXd d2f_d0d0 = d2F_d0d0(d_angle);

	for (int fi = 0; fi < restShapeF.rows(); fi++) {
		Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> asd = d2N_dxdx(fi);
		int v0 = restShapeF(fi, 0);
		int v1 = restShapeF(fi, 1);
		int v2 = restShapeF(fi, 2);
		
		for (int ddd = 0; ddd < 3; ++ddd)
			for (int i = 0; i < 3; ++i)
				for (int j = 0; j < 3; ++j)
					for (int ii = 0; ii < 3; ++ii)
						for (int jj = 0; jj < 3; ++jj) {
							int global_i = restShapeF(fi, i) + (ii*restShapeV.rows());
							int global_j = restShapeF(fi, j) + (jj*restShapeV.rows());
				
							if (global_i <= global_j) {
								II.push_back(global_i);
								JJ.push_back(global_j);
								SS.push_back(asd[ddd](i + 3 * ii, j + 3 * jj));
							}	
						}		
	}

	//for (int hi = 0; hi < num_hinges; hi++) {
	//	Eigen::Matrix< Eigen::Matrix3d, 4, 4> H_angle = d20_dxdx(hi);
	//	Eigen::Matrix<double, 4, 3> grad_angle = d0_dx(hi);
	//	//dE/dx =	dE/dF * dF/d0 * d20/dxdx + 
	//	//			dE/dF * d2F/d0d0 * d0/dx * (d0/dx)^T
	//	Eigen::Matrix3d H[4][4];
	//	for (int i = 0; i < 4; ++i)
	//		for (int j = 0; j < 4; ++j)
	//			H[i][j] = 
	//			dE_df(hi) * df_d0(hi) * H_angle(i,j) + 
	//			dE_df(hi) * d2f_d0d0(hi) * grad_angle.row(i).transpose() * grad_angle.row(j);

	//	//Finally, convert to triplets
	//	for (int i = 0; i < 4; ++i)
	//		for (int j = 0; j < 4; ++j)
	//			for (int ii = 0; ii < 3; ++ii)
	//				for (int jj = 0; jj < 3; ++jj) {
	//					int global_j = x_index(i, hi) + (ii*restShapeV.rows());
	//					int global_i = x_index(j, hi) + (jj*restShapeV.rows());

	//					if (global_i <= global_j) {
	//						//hesEntries.push_back(Eigen::Triplet<double>(global_i, global_j, H[i][j](ii, jj)));
	//						SS[index++] = H[i][j](ii, jj);
	//					}	
	//				}
	//}
}

void BendingNormal::init_hessian()
{
	
}
