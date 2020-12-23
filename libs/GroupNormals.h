#pragma once
#include "ObjectiveFunction.h"
#include <mutex>

class GroupNormals : public ObjectiveFunction
{
private:
	int startN;
	int startN_x;
	int startN_y;
	int startN_z;
	std::vector < std::vector<int>> GroupsInd;
	std::vector < std::vector<int>> currGroupsInd;
	std::vector < Eigen::MatrixX3d> NormalPos;
	std::mutex m;
public:
	GroupNormals(
		const Eigen::MatrixXd& V,
		const Eigen::MatrixX3i& F);
	~GroupNormals();
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual Cuda::Array<double>* gradient(Cuda::Array<double>& X, const bool update) override;
	void updateExtConstraints(std::vector < std::vector<int>>& CInd);
	int numV=0;
	int numF=0;
};