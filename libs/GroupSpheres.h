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
	GroupSpheres(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixX3i& F);
	~GroupSpheres();
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual Cuda::Array<double>* gradient(Cuda::Array<double>& X, const bool update) override;
	void updateExtConstraints(std::vector < std::vector<int>>& CInd);
	int numV=0;
	int numF=0;
};