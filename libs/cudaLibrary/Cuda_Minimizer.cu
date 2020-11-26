#include "Cuda_Minimizer.cuh"

namespace Cuda {
	namespace Minimizer {
		Array<double> X, p, g, curr_x;

		__global__ void currXKernel(
			double* curr_x,
			const double* X,
			const double* p,
			const double step_size)
		{
			int index = blockIdx.x;
			curr_x[index] = X[index] + step_size * p[index];
		}

		void linesearch_currX(const double step_size) {
			currXKernel << <curr_x.size, 1 >> > (
				curr_x.cuda_arr,
				X.cuda_arr,
				p.cuda_arr,
				step_size);
			CheckErr(cudaDeviceSynchronize());
		}
	}
}

