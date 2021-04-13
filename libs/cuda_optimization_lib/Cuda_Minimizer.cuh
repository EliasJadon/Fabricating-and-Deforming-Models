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
		const double* g4, const double w4,
		const double* g5, const double w5,
		const double* g6, const double w6,
		const double* g7, const double w7,
		const double* g8, const double w8,
		const double* g9, const double w9,
		const double* g10, const double w10,
		const double* g11, const double w11,
		const double* g12, const double w12,
		const double* g13, const double w13,
		const double* g14, const double w14);
};
