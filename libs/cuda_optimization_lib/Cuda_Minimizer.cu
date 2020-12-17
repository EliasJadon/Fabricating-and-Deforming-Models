#include "Cuda_Minimizer.cuh"
#define alpha_adam 1.0f
#define beta1_adam 0.90f
#define beta2_adam 0.9990f

__global__ void currXKernel(
	double* curr_x,
	const double* X,
	const double* p,
	const double step_size)
{
	int index = blockIdx.x;
	curr_x[index] = X[index] + step_size * p[index];
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
__global__ void AdamStepKernel(
	const double* g,
	double* v_adam,
	double* s_adam,
	double* p)
{
	int index = blockIdx.x;
	v_adam[index] = beta1_adam * v_adam[index] + (1 - beta1_adam) * g[index];
	double tmp = pow(g[index], 2);
	s_adam[index] = beta2_adam * s_adam[index] + (1 - beta2_adam) * tmp;
	tmp = sqrt(s_adam[index]) + 1e-8;
	p[index] = -v_adam[index] / tmp;
}


namespace Cuda {
	namespace Minimizer {
		Array<double> X, p, g, curr_x;
		Array<double> v_adam, s_adam;

		void AdamStep()
		{
			AdamStepKernel << <v_adam.size, 1 >> > (
				Minimizer::g.cuda_arr,
				v_adam.cuda_arr,
				s_adam.cuda_arr,
				Minimizer::p.cuda_arr);
			CheckErr(cudaDeviceSynchronize());
		}
		void linesearch_currX(const double step_size) {
			currXKernel << <curr_x.size, 1 >> > (
				curr_x.cuda_arr,
				X.cuda_arr,
				p.cuda_arr,
				step_size);
			CheckErr(cudaDeviceSynchronize());
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

