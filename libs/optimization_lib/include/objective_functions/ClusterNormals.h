#pragma once
#include "libs/optimization_lib/include/objective_functions/ObjectiveFunction.h"

class ClusterNormals : public ObjectiveFunction
{
private:
	virtual void init_hessian() override;
	int getNumberOfClusters();
	int CheckInputValidation();
public:
	ClusterNormals();
	~ClusterNormals();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
	
	std::vector < std::vector<int>> ClustersInd;
	std::vector < Eigen::MatrixX3d> NormalPos;
	int numV=0;
	int numF=0;
};