#pragma once
#include "libs/optimization_lib/include/objective_functions/ObjectiveFunction.h"

class BendingNormal : public ObjectiveFunction
{	
private:
	Eigen::MatrixX3d normals;
	Eigen::MatrixX3d CurrV;
	Eigen::VectorXd restArea, d_normals;
	int num_hinges = -1;
	std::vector<Eigen::Vector2d> hinges_faceIndex;

	void calculateHinges();	
	virtual void init_hessian() override;

	Eigen::VectorXd Phi(Eigen::VectorXd);
	Eigen::VectorXd dPhi_dm(Eigen::VectorXd);
	Eigen::VectorXd d2Phi_dmdm(Eigen::VectorXd);
	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> d2N_dxdx(int hi);
	Eigen::Matrix<double, 3, 9> dN_dx(int hi);
	Eigen::Matrix< double, 6, 1> dm_dN(int hi);
	Eigen::Matrix< double, 6, 6> d2m_dNdN(int hi);
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

