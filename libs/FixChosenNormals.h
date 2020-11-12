#pragma once
#include "ObjectiveFunction.h"
#include <mutex>

class FixChosenNormals : public ObjectiveFunction
{
private:
	virtual void init_hessian() override;
	int startN_x;
	int startN_y;
	int startN_z;
	Eigen::MatrixX3d diff;
	std::mutex m;
	std::vector<int> ConstrainedNormalsInd;
	std::vector<int> currConstrainedNormalsInd;
	Eigen::MatrixX3d ConstrainedNormalsPos;
public:
	FixChosenNormals();
	~FixChosenNormals();
	virtual void init() override;
	virtual void updateX(const Eigen::VectorXd& X) override;
	virtual double value(const bool update) override;
	virtual void gradient(Eigen::VectorXd& g, const bool update) override;
	virtual void hessian() override;
	void updateExtConstraints(std::vector<int>& CNormalsInd, Eigen::MatrixX3d& CNormalsPos);
	int numV=0;
	int numF=0;
};