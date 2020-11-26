#include "Cuda_AdamMinimizer.cuh"

namespace Cuda {
	namespace AdamMinimizer {
		Array<double> v_adam, s_adam;

		__global__ void stepKernel(
			const double alpha_adam, 
			const double beta1_adam,
			const double beta2_adam,
			const double* g,
			double* v_adam,
			double* s_adam,
			double* p) 
		{
			int index = blockIdx.x;
			v_adam[index] = beta1_adam * v_adam[index] + (1 - beta1_adam) * g[index];
			double tmp = pow(g[index], 2);
			s_adam[index] = beta2_adam * s_adam[index] + (1 - beta2_adam) * tmp; // element-wise square
			tmp = sqrt(s_adam[index]) + 1e-8;
			p[index] = -v_adam[index] / tmp;
		}

		void step(
			const double alpha_adam, 
			const double beta1_adam, 
			const double beta2_adam) 
		{
			stepKernel << <v_adam.size, 1 >> > (
				alpha_adam,
				beta1_adam,
				beta2_adam,
				Minimizer::g.cuda_arr,
				v_adam.cuda_arr,
				s_adam.cuda_arr,
				Minimizer::p.cuda_arr);
			CheckErr(cudaDeviceSynchronize());
		}
	}
}

