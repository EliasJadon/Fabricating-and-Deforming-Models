#pragma once
#include "ObjectiveFunction.h"
#include "FixChosenConstraints.h"
#include "FixAllVertices.h"
#include "Grouping.h"
#include "AuxSpherePerHinge.h"
#include "AuxBendingNormal.h"
#include "STVK.h"
#include "SDenergy.h"
#include "cuda_optimization_lib/Cuda_FixAllVertices.cuh"
#include "cuda_optimization_lib/Cuda_AuxSpherePerHinge.cuh"
#include "cuda_optimization_lib/Cuda_Minimizer.cuh"
#include <string>

class TotalObjective
{
public:
	std::string name;
	std::vector<std::shared_ptr<ObjectiveFunction>> objectiveList;
	std::shared_ptr<Cuda_Minimizer> cuda_Minimizer;
	double energy_value = 0, gradient_norm = 0;
	
	TotalObjective();
	~TotalObjective();
	void FDGradient(const Cuda::Array<double>& X, Cuda::Array<double>& grad);
	void checkGradient(const Eigen::VectorXd& X);
	double value(Cuda::Array<double>& curr_x, const bool update);
	void gradient(Cuda::Array<double>& X, const bool update);
};