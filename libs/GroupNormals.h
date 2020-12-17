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
	GroupNormals();
	~GroupNormals();
	virtual void init() override;
	virtual void updateX(Cuda::Array<double>& curr_x) override;
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual void gradient(Cuda::Array<double>& X, Eigen::VectorXd& g, const bool update) override;
	void updateExtConstraints(std::vector < std::vector<int>>& CInd);
	int numV=0;
	int numF=0;
};