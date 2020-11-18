#pragma once
#include "ObjectiveFunction.h"
#include "cudaLibrary/CudaBasics.h"

class AuxSpherePerHinge : public ObjectiveFunction
{	
private:
	
	Eigen::MatrixX3d CurrV, CurrCenter;
	Eigen::VectorXd  d_center,d_radius, CurrRadius;
	Eigen::VectorXd restAreaPerFace, restAreaPerHinge;
	int num_hinges = -1;
	std::vector<Eigen::Vector2d> hinges_faceIndex;
	Eigen::VectorXi x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd;
	Eigen::MatrixXi x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd;

	void calculateHinges();	
	virtual void init_hessian() override;

	Eigen::VectorXd Phi(Eigen::VectorXd);
	Eigen::VectorXd dPhi_dm(Eigen::VectorXd);
	Eigen::VectorXd d2Phi_dmdm(Eigen::VectorXd);
	Eigen::Matrix< double, 8, 1> dm_dN(int hi);
	Eigen::Matrix< double, 8, 8> d2m_dNdN(int hi);
	
public:
	std::vector<double> w_aux = { 1,100 }; // w1 = 1, w2 = 100;
	FunctionType functionType;
	double planarParameter;

	AuxSpherePerHinge(FunctionType type);
	~AuxSpherePerHinge();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
};

