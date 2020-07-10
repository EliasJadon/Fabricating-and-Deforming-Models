#pragma once
#include "libs/optimization_lib/include/objective_functions/ObjectiveFunction.h"

class AuxBendingNormal : public ObjectiveFunction
{	
private:
	
	Eigen::MatrixX3d CurrV, CurrN;
	Eigen::VectorXd restAreaPerFace, restAreaPerHinge, d_normals;
	int num_hinges = -1;
	std::vector<Eigen::Vector2d> hinges_faceIndex;
	Eigen::VectorXi x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd;
	Eigen::MatrixXi x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd;

	void calculateHinges();	
	virtual void init_hessian() override;

	Eigen::VectorXd Phi(Eigen::VectorXd);
	Eigen::VectorXd dPhi_dm(Eigen::VectorXd);
	Eigen::VectorXd d2Phi_dmdm(Eigen::VectorXd);
	Eigen::Matrix< double, 6, 1> dm_dN(int hi);
	Eigen::Matrix< double, 6, 6> d2m_dNdN(int hi);
	
public:
	float w1 = 1, w2 = 100, w3 = 100;
	OptimizationUtils::FunctionType functionType;
	double planarParameter;

	AuxBendingNormal(OptimizationUtils::FunctionType type);
	~AuxBendingNormal();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
};

