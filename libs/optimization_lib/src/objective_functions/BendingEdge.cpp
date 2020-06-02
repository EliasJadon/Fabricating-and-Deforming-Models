#include "objective_functions/BendingEdge.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <igl/triangle_triangle_adjacency.h>
#define SGN(x) (((x)<0)?(-1):(1))

BendingEdge::BendingEdge(OptimizationUtils::FunctionType type) {
	functionType = type;
	name = "Bending Edge";
	w = 0;
	std::cout << "\t" << name << " constructor" << std::endl;
}

BendingEdge::~BendingEdge() {
	std::cout << "\t" << name << " destructor" << std::endl;
}

void BendingEdge::init()
{
	std::cout << "\t" << name << " initialization" << std::endl;
	if (restShapeV.size() == 0 || restShapeF.size() == 0)
		throw name + " must define members V,F before init()!";

	calculateHinges();
	restAngle.resize(num_hinges);
	restEdgeLength.resize(num_hinges);
	restConst.resize(num_hinges);
	restArea.resize(num_hinges);
	angle.resize(num_hinges);
	x0.resize(num_hinges, 3);
	x1.resize(num_hinges, 3);
	x2.resize(num_hinges, 3);
	x3.resize(num_hinges, 3);
	planarParameter = 1;

	setRestShapeFromCurrentConfiguration();
	init_hessian();
}

void BendingEdge::updateX(const Eigen::VectorXd& X)
{
	assert(X.rows() == (restShapeV.size()+ restShapeF.size()));
	for (int hi = 0; hi < num_hinges; hi++) {
		//X = [x,x, ... ,x,y,y, ... ,y,z,z, ... ,z]
		x0.row(hi) = Eigen::Vector3d(
			X(x0_index(hi) + (0 * restShapeV.rows())),	//x-coordinate
			X(x0_index(hi) + (1 * restShapeV.rows())),	//Y-coordinate
			X(x0_index(hi) + (2 * restShapeV.rows()))	//Z-coordinate
		); 
		x1.row(hi) = Eigen::Vector3d(
			X(x1_index(hi) + (0 * restShapeV.rows())),	//x-coordinate
			X(x1_index(hi) + (1 * restShapeV.rows())),	//Y-coordinate
			X(x1_index(hi) + (2 * restShapeV.rows()))	//Z-coordinate
		);
		x2.row(hi) = Eigen::Vector3d(
			X(x2_index(hi) + (0 * restShapeV.rows())),	//x-coordinate
			X(x2_index(hi) + (1 * restShapeV.rows())),	//Y-coordinate
			X(x2_index(hi) + (2 * restShapeV.rows()))	//Z-coordinate
		);
		x3.row(hi) = Eigen::Vector3d(
			X(x3_index(hi) + (0 * restShapeV.rows())),	//x-coordinate
			X(x3_index(hi) + (1 * restShapeV.rows())),	//Y-coordinate
			X(x3_index(hi) + (2 * restShapeV.rows()))	//Z-coordinate
		);
	}
	getAngle();
}

void BendingEdge::calculateHinges() {
	std::vector<std::vector<std::vector<int>>> TT;
	std::vector<Eigen::Vector2d> hinges;
	igl::triangle_triangle_adjacency(restShapeF, TT);
	assert(TT.size() == restShapeF.rows());
	
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
					hinges.push_back(Eigen::Vector2d(FaceIndex1, FaceIndex2));
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

	///////////////////////////////////////////////////////////
	//Part 2 - Find x0,x1,x2,x3 indecis for each hinge
	num_hinges = hinges.size();
	x0_index.resize(num_hinges);
	x1_index.resize(num_hinges);
	x2_index.resize(num_hinges);
	x3_index.resize(num_hinges);
	
	for (int hi = 0; hi < num_hinges; hi++) {
		//first triangle vertices
		int v1 = restShapeF(hinges[hi](0), 0);
		int v2 = restShapeF(hinges[hi](0), 1);
		int v3 = restShapeF(hinges[hi](0), 2);
		//second triangle vertices
		int V1 = restShapeF(hinges[hi](1), 0);
		int V2 = restShapeF(hinges[hi](1), 1);
		int V3 = restShapeF(hinges[hi](1), 2);

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
			x2_index(hi) = v1;
			x0_index(hi) = v2;
			x1_index(hi) = v3;

			if (V1 != v1 && V1 != v2 && V1 != v3)
				x3_index(hi) = V1;
			else if (V2 != v1 && V2 != v2 && V2 != v3)
				x3_index(hi) = V2;
			else
				x3_index(hi) = V3;
		}
		else if (v2 != V1 && v2 != V2 && v2 != V3) {
			x2_index(hi) = v2;
			x0_index(hi) = v1;
			x1_index(hi) = v3;

			if (V1 != v1 && V1 != v2 && V1 != v3)
				x3_index(hi) = V1;
			else if (V2 != v1 && V2 != v2 && V2 != v3)
				x3_index(hi) = V2;
			else
				x3_index(hi) = V3;
		}
		else {
			x2_index(hi) = v3;
			x0_index(hi) = v1;
			x1_index(hi) = v2;

			if (V1 != v1 && V1 != v2 && V1 != v3)
				x3_index(hi) = V1;
			else if (V2 != v1 && V2 != v2 && V2 != v3)
				x3_index(hi) = V2;
			else
				x3_index(hi) = V3;
		}
	}
}

void BendingEdge::setRestShapeFromCurrentConfiguration() {
	for (int hi = 0; hi < num_hinges; hi++) {
		Eigen::Vector3d x0 = restShapeV.row(x0_index(hi));
		Eigen::Vector3d x1 = restShapeV.row(x1_index(hi));
		Eigen::Vector3d x2 = restShapeV.row(x2_index(hi));
		Eigen::Vector3d x3 = restShapeV.row(x3_index(hi));
		Eigen::Vector3d e0 = x1 - x0;
		Eigen::Vector3d n1 = e0.cross(x2 - x0);
		Eigen::Vector3d n2 = (x3 - x0).cross(e0);
		int sign = SGN(n1.cross(n2).dot(e0));
		double l_n1 = n1.norm();
		double l_n2 = n2.norm();

		//update rest variables
		//restAngle(hi) = acos(n1.dot(n2) / (l_n1*l_n2))*sign;
		restAngle(hi) = 0;
		restEdgeLength(hi) = e0.norm();
		restArea(hi) = 0.5 * (l_n1 + l_n2);
	}
	
	restConst = 3 * restEdgeLength.cwiseProduct(restEdgeLength).cwiseProduct(restArea.cwiseInverse());
}

void BendingEdge::getAngle() {
	for (int hi = 0; hi < num_hinges; hi++) {
		Eigen::Vector3d e0 = x1.row(hi) - x0.row(hi);
		Eigen::Vector3d n1 = e0.cross(x2.row(hi) - x0.row(hi));
		Eigen::Vector3d n2 = (x3.row(hi) - x0.row(hi)).cross(e0);
		int sign = SGN(n1.cross(n2).dot(e0));
		double ratio = std::max(std::min(n1.dot(n2) / (n1.norm()*n2.norm()), 1.0), -1.0);
		angle(hi) = acos(ratio)*sign;
	}
}

Eigen::VectorXd BendingEdge::AngleFunctionvalue(Eigen::VectorXd x) {
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

Eigen::VectorXd BendingEdge::dF_d0(Eigen::VectorXd x) {
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

Eigen::VectorXd BendingEdge::d2F_d0d0(Eigen::VectorXd x) {
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

double BendingEdge::value(const bool update)
{
	Eigen::VectorXd d_angle = angle - restAngle;
	Eigen::VectorXd f = AngleFunctionvalue(d_angle);
	Eigen::VectorXd Energy = k * f.cwiseProduct(restConst);

	double value = Energy.sum();
	
	if (update) {
		Efi.setZero();
		energy_value = value;
	}
	return value;
}

void BendingEdge::gradient(Eigen::VectorXd& g, const bool update)
{
	g.conservativeResize(restShapeV.size() + restShapeF.size());
	g.setZero();

	Eigen::VectorXd d_angle = angle - restAngle;
	Eigen::VectorXd dE_df = k * restConst;
	Eigen::VectorXd df_d0 = dF_d0(d_angle);

	for (int hi = 0; hi < num_hinges; hi++) {
		Eigen::Matrix<double, 4, 3> dE_dx = dE_df(hi) * df_d0(hi) * d0_dx(hi);

		for(int i=0;i<4;i++)
			for (int xyz = 0; xyz < 3; ++xyz)
				g[x_index(i,hi) + (xyz*restShapeV.rows())] += dE_dx(i, xyz);
	}

	if (update)
		gradient_norm = g.norm();
}

Eigen::Matrix< Eigen::Matrix3d, 4, 4> BendingEdge::d20_dxdx(int hi) {
	//start copied code from gradient
	Eigen::Vector3d e0 = x1.row(hi) - x0.row(hi);
	Eigen::Vector3d e1 = x2.row(hi) - x0.row(hi);
	Eigen::Vector3d e2 = x3.row(hi) - x0.row(hi);
	Eigen::Vector3d e3 = x2.row(hi) - x1.row(hi);
	Eigen::Vector3d e4 = x3.row(hi) - x1.row(hi);
	Eigen::Vector3d n1 = e0.cross(e1);
	Eigen::Vector3d n2 = e2.cross(e0);
	double l_e0 = e0.norm(); e0 /= l_e0;
	double l_e1 = e1.norm(); e1 /= l_e1;
	double l_e2 = e2.norm(); e2 /= l_e2;
	double l_e3 = e3.norm(); e3 /= l_e3;
	double l_e4 = e4.norm(); e4 /= l_e4;
	double l_n1 = n1.norm(); n1 /= l_n1;
	double l_n2 = n2.norm(); n2 /= l_n2;
	double angle_1 = acos(e0.dot(e1));
	double angle_2 = acos(e2.dot(e0));
	double angle_3 = acos(e3.dot(-e0));
	double angle_4 = acos((-e0).dot(e4));
	double h_1 = l_n1 / l_e1;
	double h_2 = l_n2 / l_e2;
	double h_3 = l_n1 / l_e3;
	double h_4 = l_n2 / l_e4;
	double h_01 = l_n1 / l_e0;
	double h_02 = l_n2 / l_e0;
	//end copied code from gradient
	
	// vectors m
	Eigen::Vector3d m1 = -e1.cross(n1);
	Eigen::Vector3d m2 = e2.cross(n2);
	Eigen::Vector3d m3 = e3.cross(n1);
	Eigen::Vector3d m4 = -e4.cross(n2);
	Eigen::Vector3d m01 = e0.cross(n1);
	Eigen::Vector3d m02 = -e0.cross(n2);

	//Hessian of angle
	Eigen::Matrix< Eigen::Matrix3d, 4, 4> H_angle;
	//Eigen is making v1 * v2.T harder than it should be
	Eigen::RowVector3d n1_t = n1.transpose();
	Eigen::RowVector3d n2_t = n2.transpose();
	Eigen::RowVector3d m1_t = m1.transpose();
	Eigen::RowVector3d m2_t = m2.transpose();
	Eigen::RowVector3d m3_t = m3.transpose();
	Eigen::RowVector3d m4_t = m4.transpose();
	Eigen::RowVector3d m01_t = m01.transpose();
	Eigen::RowVector3d m02_t = m02.transpose();
	Eigen::Matrix3d B1 = n1 * m01_t / (l_e0*l_e0);
	Eigen::Matrix3d B2 = n2 * m02_t / (l_e0*l_e0);

	H_angle(0, 0) = cos(angle_3) / (h_3*h_3) * (m3 * n1_t + n1 * m3_t) - B1
		+ cos(angle_4) / (h_4*h_4) * (m4 * n2_t + n2 * m4_t) - B2;
	H_angle(0, 1) = cos(angle_3) / (h_1*h_3) * m1*n1_t + cos(angle_1) / (h_1*h_3)*n1*m3_t + B1
		+ cos(angle_4) / (h_2*h_4)*m2*n2_t + cos(angle_2) / (h_2*h_4)*n2*m4_t + B2;
	H_angle(0, 2) = cos(angle_3) / (h_3*h_01)*m01*n1_t - n1 * m3_t / (h_01*h_3);
	H_angle(0, 3) = cos(angle_4) / (h_4*h_02)*m02*n2_t - n2 * m4_t / (h_02*h_4);
	H_angle(1, 1) = cos(angle_1) / (h_1*h_1)*(m1*n1_t + n1 * m1_t) - B1
		+ cos(angle_2) / (h_2*h_2)*(m2*n2_t + n2 * m2_t) - B2;
	H_angle(1, 2) = cos(angle_1) / (h_1*h_01)*m01*n1_t - n1 * m1_t / (h_01*h_1);
	H_angle(1, 3) = cos(angle_2) / (h_2*h_02)*m02*n2_t - n2 * m2_t / (h_02*h_2);
	H_angle(2, 2) = -(n1*m01_t + m01 * n1_t) / (h_01*h_01);
	H_angle(2, 3).setZero();
	H_angle(3, 3) = -(n2*m02_t + m02 * n2_t) / (h_02*h_02);
	for (int i = 1; i < 4; ++i)
		for (int j = i - 1; j >= 0; --j)
			H_angle(i, j) = H_angle(j, i).transpose();

	return H_angle;
}

Eigen::Matrix<double, 4, 3> BendingEdge::d0_dx(int hi) {
	//start copied code from gradient
	Eigen::Vector3d e0 = x1.row(hi) - x0.row(hi);
	Eigen::Vector3d e1 = x2.row(hi) - x0.row(hi);
	Eigen::Vector3d e2 = x3.row(hi) - x0.row(hi);
	Eigen::Vector3d e3 = x2.row(hi) - x1.row(hi);
	Eigen::Vector3d e4 = x3.row(hi) - x1.row(hi);
	Eigen::Vector3d n1 = e0.cross(e1);
	Eigen::Vector3d n2 = e2.cross(e0);
	double l_e0 = e0.norm(); e0 /= l_e0;
	double l_e1 = e1.norm(); e1 /= l_e1;
	double l_e2 = e2.norm(); e2 /= l_e2;
	double l_e3 = e3.norm(); e3 /= l_e3;
	double l_e4 = e4.norm(); e4 /= l_e4;
	double l_n1 = n1.norm(); n1 /= l_n1;
	double l_n2 = n2.norm(); n2 /= l_n2;
	double angle_1 = acos(e0.dot(e1));
	double angle_2 = acos(e2.dot(e0));
	double angle_3 = acos(e3.dot(-e0));
	double angle_4 = acos((-e0).dot(e4));
	double h_1 = l_n1 / l_e1;
	double h_2 = l_n2 / l_e2;
	double h_3 = l_n1 / l_e3;
	double h_4 = l_n2 / l_e4;
	double h_01 = l_n1 / l_e0;
	double h_02 = l_n2 / l_e0;
	//end copied code from gradient

	//Gradient of angle
	Eigen::Matrix<double, 4, 3> grad_angle;
	grad_angle.row(0) = n1 * cos(angle_3) / h_3 + n2 * cos(angle_4) / h_4;
	grad_angle.row(1) = n1 * cos(angle_1) / h_1 + n2 * cos(angle_2) / h_2;
	grad_angle.row(2) = -n1 / h_01;
	grad_angle.row(3) = -n2 / h_02;

	return grad_angle;
}

int BendingEdge::x_index(int i,int hi) {
	if (i == 0)
		return x0_index(hi);
	else if (i == 1)
		return x1_index(hi);
	else if (i == 2)
		return x2_index(hi);
	else if (i == 3)
		return x3_index(hi);
	else
		assert("Error!");
}

void BendingEdge::hessian() {
	// constant factors
	int index = 0;
	Eigen::VectorXd d_angle = angle - restAngle;
	Eigen::VectorXd dE_df = k * restConst;
	Eigen::VectorXd df_d0 = dF_d0(d_angle);
	Eigen::VectorXd d2f_d0d0 = d2F_d0d0(d_angle);

	for (int hi = 0; hi < num_hinges; hi++) {
		Eigen::Matrix< Eigen::Matrix3d, 4, 4> H_angle = d20_dxdx(hi);
		Eigen::Matrix<double, 4, 3> grad_angle = d0_dx(hi);
		//dE/dx =	dE/dF * dF/d0 * d20/dxdx + 
		//			dE/dF * d2F/d0d0 * d0/dx * (d0/dx)^T
		Eigen::Matrix3d H[4][4];
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				H[i][j] = 
				dE_df(hi) * df_d0(hi) * H_angle(i,j) + 
				dE_df(hi) * d2f_d0d0(hi) * grad_angle.row(i).transpose() * grad_angle.row(j);

		//Finally, convert to triplets
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				for (int ii = 0; ii < 3; ++ii)
					for (int jj = 0; jj < 3; ++jj) {
						int global_j = x_index(i, hi) + (ii*restShapeV.rows());
						int global_i = x_index(j, hi) + (jj*restShapeV.rows());

						if (global_i <= global_j) {
							//hesEntries.push_back(Eigen::Triplet<double>(global_i, global_j, H[i][j](ii, jj)));
							SS[index++] = H[i][j](ii, jj);
						}	
					}
	}
	SS[index++] = 0;
}

void BendingEdge::init_hessian()
{
	II.clear();
	JJ.clear();
	
	for (int hi = 0; hi < num_hinges; hi++) {
		//Finally, convert to triplets
		for (int i = 0; i < 4; ++i)
			for (int j = 0; j < 4; ++j)
				for (int ii = 0; ii < 3; ++ii)
					for (int jj = 0; jj < 3; ++jj) {
						int global_j = x_index(i, hi) + (ii*restShapeV.rows());
						int global_i = x_index(j, hi) + (jj*restShapeV.rows());

						if (global_i <= global_j) {
							II.push_back(global_i);
							JJ.push_back(global_j);
						}
					}
	}
	II.push_back(restShapeV.size() + restShapeF.size() - 1);
	JJ.push_back(restShapeV.size() + restShapeF.size() - 1);
	SS = std::vector<double>(II.size(), 0.);
}
