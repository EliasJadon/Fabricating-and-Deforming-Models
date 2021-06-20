#pragma once
#include "ObjectiveFunction.h"
#include <mutex>

class FixChosenConstraints : public ObjectiveFunction
{
private:
	std::mutex m_value, m_gradient;
	std::vector<int> Constraints_indices;
	Eigen::MatrixX3d Constraints_Position;
public:
	FixChosenConstraints(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F);
	~FixChosenConstraints();
	
	virtual double value(Cuda::Array<double>& curr_x, const bool update) override;
	virtual void gradient(Cuda::Array<double>& X, const bool update) override;
	void updateExtConstraints(std::vector<int>& CVInd, Eigen::MatrixX3d& CVPos);
};