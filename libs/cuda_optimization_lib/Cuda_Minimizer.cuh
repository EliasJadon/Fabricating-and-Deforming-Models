#pragma once
#include "Cuda_Basics.cuh"
#include "Cuda_AuxBendingNormal.cuh"
#include "Cuda_FixAllVertices.cuh"
#include "Cuda_SSSymmetricDirichlet.cuh"
#include "Cuda_AuxSpherePerHinge.cuh"

namespace Cuda {
	namespace Minimizer {
		extern Array<double> X, p, g, curr_x;
		void linesearch_currX(const double step_size);
		void TotalGradient(const double, const double, const double, const double);
	}
}
