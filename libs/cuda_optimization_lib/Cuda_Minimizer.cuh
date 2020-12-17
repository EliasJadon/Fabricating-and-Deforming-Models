#pragma once
#include "Cuda_Basics.cuh"
#include "Cuda_AuxBendingNormal.cuh"
#include "Cuda_FixAllVertices.cuh"
#include "Cuda_AuxSpherePerHinge.cuh"
#include "Cuda_FixChosenConstraints.cuh"

class Cuda_Minimizer {
public:
	Cuda::Array<double> X, p, g, curr_x, v_adam, s_adam;

	Cuda_Minimizer(const unsigned int size);
	~Cuda_Minimizer();
	void adam_Step();
	void gradientDescent_Step();
	void linesearch_currX(const double step_size);
	void TotalGradient(
		const double* g1, const double w1,
		const double* g2, const double w2,
		const double* g3, const double w3,
		const double* g4, const double w4);
};
