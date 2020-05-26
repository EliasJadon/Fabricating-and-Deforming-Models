#pragma once
#include "libs/optimization_lib/include/objective_functions/ObjectiveFunction.h"

class BendingNormal : public ObjectiveFunction
{	
private:
	Eigen::MatrixXd normals;
	Eigen::MatrixX3d CurrV;
	Eigen::VectorXd restArea, d_normals;
	int num_hinges = -1;
	std::vector<Eigen::Vector2d> hinges_faceIndex;

	void calculateHinges();	
	virtual void init_hessian() override;

	Eigen::VectorXd Phi(Eigen::VectorXd);
	Eigen::VectorXd dPhi_df(Eigen::VectorXd);
	Eigen::VectorXd d2Phi_dfdf(Eigen::VectorXd);
	Eigen::Matrix< Eigen::Matrix3d, 4, 4> d2N_dxdx(int hi);
	Eigen::Matrix<double, 4, 3> dN_dx(int hi);
public:
	OptimizationUtils::FunctionType functionType;
	float planarParameter;

	BendingNormal(OptimizationUtils::FunctionType type);
	~BendingNormal();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
};

