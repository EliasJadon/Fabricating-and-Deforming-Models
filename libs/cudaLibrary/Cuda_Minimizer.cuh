#pragma once
#include "Cuda_Basics.cuh"

namespace Cuda {
	namespace Minimizer {
		extern Array<double> X, p, g, curr_x;
		void linesearch_currX(const double step_size);
	}
}
