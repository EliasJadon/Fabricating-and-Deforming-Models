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
			const unsigned int size,
			double* g,
			const double* g1, const double w1,
			const double* g2, const double w2,
			const double* g3, const double w3,
			const double* g4, const double w4,
			const double* g5, const double w5) 
		{
			unsigned int Global_idx = threadIdx.x + blockIdx.x * blockDim.x;
			if (Global_idx < size) {
				g[Global_idx] =
					w1 * g1[Global_idx] +
					w2 * g2[Global_idx] +
					w3 * g3[Global_idx] +
					w4 * g4[Global_idx] +
					w5 * g5[Global_idx];
			}
		}

		void TotalGradient(
			const double w_AuxBendingNormal,
			const double w_FixAllVertices,
			const double w_SymmetricDirichlet,
			const double w_AuxSpherePerHinge,
			const double w_FixChosenVertices)
		{
			TotalGradientKernel << <ceil(g.size / (double)1024), 1024 >> > (
				g.size, g.cuda_arr,
				AuxBendingNormal::grad.cuda_arr,	w_AuxBendingNormal,
				FixAllVertices::grad.cuda_arr,		w_FixAllVertices,
				SSSymmetricDirichlet::grad.cuda_arr,w_SymmetricDirichlet,
				AuxSpherePerHinge::grad.cuda_arr, w_AuxSpherePerHinge,
				FixChosenVertices::grad.cuda_arr, w_FixChosenVertices
			);
			CheckErr(cudaDeviceSynchronize());
		}

	}
}

