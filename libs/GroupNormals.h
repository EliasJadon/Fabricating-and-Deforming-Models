#pragma once
#include "ObjectiveFunction.h"
#include <mutex>

class GroupNormals : public ObjectiveFunction
{
private:
	virtual void init_hessian() override;
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
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
	void updateExtConstraints(std::vector < std::vector<int>>& CInd);
	int numV=0;
	int numF=0;
};