//#pragma once
//#include "libs/optimization_lib/include/objective_functions/ObjectiveFunction.h"
//
///*
//	This class implements Constant Strain Triangles elements in 3D
//	Mostly the same as CSTElement2D except for matrix sizes
//*/
//
//
//class MembraneConstraints : public ObjectiveFunction {
//	
//private:
//	// Jacobian of the parameterization per face
//	Eigen::VectorXd a;
//	Eigen::VectorXd b;
//	Eigen::VectorXd c;
//	Eigen::VectorXd d;
//	Eigen::VectorXd detJ;
//	
//	Eigen::MatrixX3d B1, B2;
//
//	//dense mesh derivative matrices
//	Eigen::Matrix3Xd D1d, D2d;
//
//	//material type
//	Utils::Material type;
//	Utils::Jacobian JacType;
//
//	//material parameters...
//	double shearModulus, bulkModulus;
//	//keep track of the rest shape area
//	Eigen::VectorXd restShapeArea;
//	Eigen::MatrixX3d CurrV;
//	//tmp matrices used to speed up computation of the deformation gradient, green strain, etc
//	std::vector<Eigen::Matrix2d> dXInv, strain ,Jac;
//	std::vector<Eigen::Matrix<double, 3, 2>> F;
//
//	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> ddB1_dXdX(int fi);
//	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> ddB2_dXdX(int fi);
//	Eigen::Matrix<double, 3, 9> dB1_dX(int fi);
//	Eigen::Matrix<double, 3, 9> dB2_dX(int fi);
//
//	Eigen::Matrix<double, 4, 9> dJ_dX(int fi);
//	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 4> ddJ_dXdX(int fi);
//
//	//sets important properties of the rest shape using the set of points passed in as parameters
//	void setRestShapeFromCurrentConfiguration();
//	virtual void init_hessian();
//public:
//	MembraneConstraints(Utils::Material type, Utils::Jacobian JacType);
//	~MembraneConstraints();
//	virtual void init();
//	
//	virtual void updateX(const Eigen::VectorXd& X);
//	virtual double value(const bool update);
//	virtual void gradient(Eigen::VectorXd& g, const bool update);
//	virtual void hessian();
//};