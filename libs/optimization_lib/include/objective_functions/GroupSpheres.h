#pragma once
#include "libs/optimization_lib/include/objective_functions/ObjectiveFunction.h"
#include <mutex>

class GroupSpheres : public ObjectiveFunction
{
private:
	virtual void init_hessian() override;
	int startC;
	int startC_x;
	int startC_y;
	int startC_z;
	int startR;
	std::vector < std::vector<int>> GroupsInd;
	std::vector < std::vector<int>> currGroupsInd;
	std::vector < Eigen::MatrixX3d> SphereCenterPos;
	std::vector < Eigen::VectorXd> SphereRadiusLen;
	std::mutex m;
public:
	GroupSpheres();
	~GroupSpheres();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
	void updateExtConstraints(std::vector < std::vector<int>>& CInd);
	int numV=0;
	int numF=0;
};