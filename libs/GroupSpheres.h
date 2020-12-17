#pragma once
#include "ObjectiveFunction.h"
#include <mutex>

class GroupSpheres : public ObjectiveFunction
{
private:
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
	virtual void updateX(Cuda::Array<double>& curr_x) override;
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual void gradient(Cuda::Array<double>& X, Eigen::VectorXd& g, const bool update) override;
	void updateExtConstraints(std::vector < std::vector<int>>& CInd);
	int numV=0;
	int numF=0;
};