#pragma once
#include "ObjectiveFunction.h"
#include "FixChosenConstraints.h"
#include "FixAllVertices.h"
#include "AuxSpherePerHinge.h"
#include "AuxBendingNormal.h"
#include "STVK.h"
#include "SDenergy.h"
#include "cuda_optimization_lib/Cuda_FixAllVertices.cuh"
#include "cuda_optimization_lib/Cuda_AuxSpherePerHinge.cuh"
#include "cuda_optimization_lib/Cuda_Minimizer.cuh"
#include <string>

class TotalObjective : public ObjectiveFunction
{
public:
	std::vector<std::shared_ptr<ObjectiveFunction>> objectiveList;
	
	TotalObjective();
	~TotalObjective();


	virtual double value(Cuda::Array<double>& curr_x, const bool update);
	virtual void gradient(Cuda::Array<double>& X, const bool update);

};