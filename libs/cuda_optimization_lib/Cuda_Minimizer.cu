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


		__global__ void TotalGradientKernel(
			const unsigned int size, double* total_g,
			const double* g1, const double w1,
			const double* g2, const double w2,
			const double* g3, const double w3,
			const double* g4, const double w4) 
		{
			unsigned int Global_idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (Global_idx < size) {
				total_g[Global_idx] =
					w1 * g1[Global_idx] +
					w2 * g2[Global_idx] +
					w3 * g3[Global_idx] +
					w4 * g4[Global_idx];
			}
		}

		void TotalGradient(
			const double* g1, const double w1,
			const double* g2, const double w2,
			const double* g3, const double w3,
			const double* g4, const double w4)
		{
			TotalGradientKernel << <ceil(g.size / (double)1024), 1024 >> > (
				g.size, g.cuda_arr,
				g1,	w1,
				g2,	w2,
				g3,	w3,
				g4,	w4);
			CheckErr(cudaDeviceSynchronize());
		}

	}
}

