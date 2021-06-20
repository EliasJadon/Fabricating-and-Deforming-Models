#include "FixChosenConstraints.h"

FixChosenConstraints::FixChosenConstraints(const Eigen::MatrixXd& V, const Eigen::MatrixX3i& F)
{
	init_mesh(V, F);
    name = "Fix Chosen Vertices";
	w = 100000;
	std::cout << "\t" << name << " constructor" << std::endl;
}

FixChosenConstraints::~FixChosenConstraints() 
{
	std::cout << "\t" << name << " destructor" << std::endl;
}

void FixChosenConstraints::updateExtConstraints(
	std::vector<int>& CVInd,
	Eigen::MatrixX3d& CVPos)
{
	m_value.lock();
	m_gradient.lock();

	Constraints_indices = CVInd;
	Constraints_Position = CVPos;

	m_gradient.unlock();
	m_value.unlock();
}

double FixChosenConstraints::value(Cuda::Array<double>& curr_x, const bool update)
{
	m_value.lock();
	double value = 0;
	for (int i = 0; i < Constraints_indices.size(); i++) {
		const unsigned int v_index = Constraints_indices[i];
		double3 Vi = getV(curr_x, v_index);
		value += squared_norm(sub(Vi, Constraints_Position.row(i)));
	}
	m_value.unlock();

	if (update)
		energy_value = value;
	return value;
}

void FixChosenConstraints::gradient(Cuda::Array<double>& X, const bool update)
{
	m_gradient.lock();
	for (int i = 0; i < Constraints_indices.size(); i++) {
		const unsigned int v_index = Constraints_indices[i];
		grad.host_arr[v_index + mesh_indices.startVx] = 2 * (X.host_arr[v_index + mesh_indices.startVx] - Constraints_Position(i,0));
		grad.host_arr[v_index + mesh_indices.startVy] = 2 * (X.host_arr[v_index + mesh_indices.startVy] - Constraints_Position(i,1));
		grad.host_arr[v_index + mesh_indices.startVz] = 2 * (X.host_arr[v_index + mesh_indices.startVz] - Constraints_Position(i,2));
	}
	m_gradient.unlock();

	if (update) {
		gradient_norm = 0;
		for (int i = 0; i < grad.size; i++)
			gradient_norm += pow(grad.host_arr[i], 2);
	}
}