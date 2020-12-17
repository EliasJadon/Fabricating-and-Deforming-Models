#pragma once
#include "Cuda_Basics.cuh"
#include "Cuda_AuxBendingNormal.cuh"
#include "Cuda_FixAllVertices.cuh"
#include "Cuda_AuxSpherePerHinge.cuh"
#include "Cuda_FixChosenConstraints.cuh"

namespace Cuda {
	namespace Minimizer {
		extern Array<double> X, p, g, curr_x;
		extern Array<double> v_adam, s_adam;

		void AdamStep();
		void linesearch_currX(const double step_size);
		void TotalGradient(
			const double* g1, const double w1,
			const double* g2, const double w2,
			const double* g3, const double w3,
			const double* g4, const double w4);
	}
}
