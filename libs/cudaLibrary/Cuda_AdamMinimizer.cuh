#pragma once
#include "Cuda_Minimizer.cuh"

namespace Cuda {
	namespace AdamMinimizer {
		extern Array<double> v_adam, s_adam;
		void step(const double,const double,const double);
	}
}
