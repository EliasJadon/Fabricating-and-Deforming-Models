#pragma once
#include "ObjectiveFunction.h"
#include <mutex>

class GroupNormals : public ObjectiveFunction
{
private:
	unsigned int startN_x, startN_y, startN_z;
	Cuda::Array<int> GroupInd;
	unsigned int num_clusters, max_face_per_cluster;
	std::mutex m_value, m_gradient;
public:
	Cuda::Array<double> grad;

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