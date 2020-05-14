#pragma once
#include "libs/optimization_lib/include/objective_functions/ObjectiveFunction.h"

class SymmetricDirichlet : public ObjectiveFunction {
private:
	Eigen::VectorXd a;
	Eigen::VectorXd b;
	Eigen::VectorXd c;
	Eigen::VectorXd d;
	Eigen::VectorXd detJ;
	Eigen::MatrixX3d B1, B2;
	Eigen::Matrix3Xd D1d, D2d;
	Eigen::VectorXd restShapeArea;
	Eigen::MatrixX3d CurrV;
	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> ddB1_dXdX(int fi);
	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> ddB2_dXdX(int fi);
	Eigen::Matrix<double, 3, 9> dB1_dX(int fi);
	Eigen::Matrix<double, 3, 9> dB2_dX(int fi);
	Eigen::Matrix<double, 4, 9> dJ_dX(int fi);
	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 4> ddJ_dXdX(int fi);
	Eigen::Matrix<double, 1, 4> dE_dJ(int fi);

	//sets important properties of the rest shape using the set of points passed in as parameters
	void setRestShapeFromCurrentConfiguration();
	virtual void init_hessian();
public:
	SymmetricDirichlet();
	~SymmetricDirichlet();
	virtual void init();
	virtual void updateX(const Eigen::VectorXd& X);
	virtual double value(const bool update);
	virtual void gradient(Eigen::VectorXd& g, const bool update);
	virtual void hessian();
};