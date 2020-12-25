#pragma once
#include "ObjectiveFunction.h"
#include "FixChosenConstraints.h"
#include "FixAllVertices.h"
#include "Grouping.h"
#include "AuxSpherePerHinge.h"
#include "AuxBendingNormal.h"
#include "STVK.h"
#include "cuda_optimization_lib/Cuda_FixAllVertices.cuh"
#include "cuda_optimization_lib/Cuda_AuxSpherePerHinge.cuh"
#include "cuda_optimization_lib/Cuda_Minimizer.cuh"
#include <string>

class TotalObjective
{
public:
	std::string name;
	std::vector<std::shared_ptr<ObjectiveFunction>> objectiveList;
	double energy_value, gradient_norm;
	
	TotalObjective();
	~TotalObjective();
	double value(Cuda::Array<double>& curr_x, const bool update);
	void gradient(
		std::shared_ptr<Cuda_Minimizer> cuda_Minimizer,
		Cuda::Array<double>& X,  
		const bool update);
};