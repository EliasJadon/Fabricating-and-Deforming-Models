#pragma once
#include "ObjectiveFunction.h"
#include "cudaLibrary/CudaBasics.cuh"

class BendingNormal : public ObjectiveFunction
{	
private:
	Eigen::MatrixX3d normals;
	Eigen::MatrixX3d CurrV;
	Eigen::VectorXd restArea, d_normals;
	int num_hinges = -1;
	std::vector<Eigen::Vector2d> hinges_faceIndex;
	Eigen::VectorXi x0_GlobInd, x1_GlobInd, x2_GlobInd, x3_GlobInd;
	Eigen::MatrixXi x0_LocInd, x1_LocInd, x2_LocInd, x3_LocInd;

	void calculateHinges();	
	virtual void init_hessian() override;

	Eigen::VectorXd Phi(Eigen::VectorXd);
	Eigen::VectorXd dPhi_dm(Eigen::VectorXd);
	Eigen::VectorXd d2Phi_dmdm(Eigen::VectorXd);
	Eigen::Matrix<Eigen::Matrix<double, 9, 9>, 1, 3> d2N_dxdx_perface(int hi);
	Eigen::Matrix<Eigen::Matrix<double, 12, 12>, 1, 6> d2N_dxdx_perhinge(int hi);
	Eigen::Matrix<double, 3, 9> dN_dx_perface(int hi);
	Eigen::Matrix< double, 6, 1> dm_dN(int hi);
	Eigen::Matrix< double, 6, 6> d2m_dNdN(int hi);
	Eigen::Matrix<double, 6, 12> dN_dx_perhinge(int hi);

	int x_GlobInd(int index, int hi);
public:
	FunctionType functionType;
	float planarParameter;

	BendingNormal(FunctionType type);
	~BendingNormal();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
};

